from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig # Import RunnableConfig
from langchain_core.output_parsers.string import StrOutputParser # Import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import logging
import json # Import json for formatting streamed data
from typing import AsyncGenerator # Import AsyncGenerator

from .config import GOOGLE_API_KEY, LLM_MODEL
from .tools import portfolio_retriever, agent_tools
from .prompts import PLANNER_PROMPT, EXECUTOR_PROMPT, REPLANNER_PROMPT
from .schemas import Plan, PlanExecute, ReplannerOutput 

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Mis ---
def clean_newlines(text: str) -> str:
    """Removes extra newline characters from a string."""
    return text.replace("\n\n", "\n")


class MultiStepAgent:
    def __init__(self):
        # --- LLM and Tools ---
        # Explicitly get the project ID from environment variables
        # project_id = GOOGLE_CLOUD_PROJECT
        # if not project_id:
        #     raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set!")
        
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
        self.tools = [portfolio_retriever] # Add other tools here if needed
        self.agent_executor = create_react_agent(self.llm, self.tools)

        # --- Prompts ---
        self.planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PLANNER_PROMPT,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        self.replanner_prompt = ChatPromptTemplate.from_template(
            REPLANNER_PROMPT,
        )

        # --- Chains ---
        self.planner = self.planner_prompt | self.llm.with_structured_output(Plan)
        self.replanner = self.replanner_prompt | self.llm.with_structured_output(ReplannerOutput)

        # --- Workflow Definition ---
        self.workflow = StateGraph(PlanExecute)
        self.workflow.add_node("planner", self._plan_step)
        self.workflow.add_node("agent", self._execute_step)
        self.workflow.add_node("replan", self._replan_step)
        self.workflow.add_edge(START, "planner")
        self.workflow.add_edge("planner", "agent")
        self.workflow.add_edge("agent", "replan")
        self.workflow.add_conditional_edges("replan", self._should_end, {"agent": "agent", END: END})
        self.app = self.workflow.compile()

        logger.info("Setting up agent executor and prompts...")
        self.tools = agent_tools
        if not isinstance(self.tools, list):
            logger.error(f"agent_tools imported from tools.py is not a list: {self.tools}")
            raise TypeError("agent_tools must be a list")
        
        self.agent_executor = create_react_agent(self.llm, self.tools)

        try:
            logger.info(f"Successfully initialized ChatGoogleGenerativeAI with Model: {LLM_MODEL}")
        except Exception as e:
            # Use logger for errors
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI. Check GOOGLE_API_KEY/model ('{LLM_MODEL}'). Error: {e}", exc_info=True)
            raise

        logger.info("MultiStepAgent initialized successfully.")

    # --- Workflow Nodes ---
    async def _plan_step(self, state: PlanExecute):
        logger.info(f"--- Planning Step --- Input: {state['input']}")
        plan = await self.planner.ainvoke({"messages": [("user", state["input"])]})
        logger.info(f"Generated Plan: {plan.steps}")
        return {"plan": plan.steps, "intermediate_responses": []}

    async def _execute_step(self, state: PlanExecute):
        plan = state["plan"]
        if not plan:
            logger.warning("Execute step called with empty plan.")
            return {"response": "No more steps in the plan."}

        task = plan[0]
        step_number = len(state.get("past_steps", [])) + 1
        logger.info(f"--- Executing Step {step_number}: {task} ---")
        
        try:
            # Pass only the current task to the agent executor
            agent_executor_input = {"messages": [("user", task)]}
            logger.debug(f"Agent Executor Input: {agent_executor_input}")
            
            agent_response = await self.agent_executor.ainvoke(agent_executor_input)
            
            logger.debug(f"Agent Executor Raw Response: {agent_response}")

            # Extract direct tool output (same logic as before)
            if 'output' in agent_response:
                tool_output = agent_response['output']
                logger.info(f"Extracted Tool Output for Step {step_number}: {tool_output}")
            else:
                logger.warning(f"'output' key not found in agent_executor response. Falling back to last message. Response keys: {agent_response.keys()}")
                tool_output = agent_response["messages"][-1].content
                logger.info(f"Execution Fallback (last message) for Step {step_number}: {tool_output}")

            return {
                "past_steps": state.get("past_steps", []) + [(task, tool_output)],
                "plan": plan[1:],
                "intermediate_responses": state.get("intermediate_responses", []) + [tool_output]
            }
        except Exception as e:
            logger.error(f"Error during execution of step '{task}': {e}", exc_info=True)
            error_message = f"Error executing step: {e}"
            return {
                "past_steps": state.get("past_steps", []) + [(task, error_message)],
                "plan": plan[1:], 
                "intermediate_responses": state.get("intermediate_responses", []) + [error_message]
            }

    async def _replan_step(self, state: PlanExecute):
        logger.info("--- Replanning Step ---")
        all_responses = "\n".join(state["intermediate_responses"])
        all_steps = "\n".join([f"{step}: {response}" for step, response in state["past_steps"]])
        context = f"Input Query: {state['input']}\n\nInfo from previous steps:\n{all_steps}\n\nDirect tool responses:\n{all_responses}"
        logger.info(f"Replanner Context:\n{context}")
        
        try:
            # Get structured output matching the new ReplannerOutput model
            output: ReplannerOutput = await self.replanner.ainvoke({**state, "input": context})
            logger.info(f"Replanner LLM structured output (ReplannerOutput object): {output}")

            # Parse the ReplannerOutput object
            if output and output.final_answer:
                cleaned_response = clean_newlines(output.final_answer)
                logger.info(f"Replanner decided on Final Response: {cleaned_response}")
                return {"response": cleaned_response}
            
            elif output and output.plan and output.plan.steps:
                new_steps = output.plan.steps
                logger.info(f"Replanner decided on New Plan Steps: {new_steps}")
                return {"plan": new_steps}
            
            else:
                # Handle cases where structured output failed or LLM didn't provide response/plan
                logger.warning(f"Replanner returned ReplannerOutput object without final_answer or plan.steps. Output: {output}")
                return {"response": "Agent could not determine a final response or next step after replanning."}

        except Exception as e:
             logger.error(f"Error during replanning LLM call or structured output parsing: {e}", exc_info=True)
             return {"response": f"An error occurred during the replanning phase: {e}"} 

    def _should_end(self, state: PlanExecute):
        logger.debug(f"Checking should_end: response in state = {'response' in state and state['response'] is not None}")
        return END if "response" in state and state["response"] is not None else "agent"

    # --- NEW: Streaming method --- 
    async def astream_events(self, user_input: str) -> AsyncGenerator[str, None]:
        """Runs the agent workflow and yields Server-Sent Events for each step."""
        logger.info(f"--- Starting agent stream for Input: {user_input} ---")
        # Use triple quotes for multiline f-string
        yield f"""event: start
data: {json.dumps({'input': user_input})}\n\n""" 
        
        config = {"recursion_limit": 50}
        last_event_for_error = None
        final_response_yielded = False

        try:
            async for event in self.app.astream(
                {"input": user_input, "plan": [], "past_steps": [], "response": None, "intermediate_responses": []},
                config=config,
            ):
                last_event_for_error = event # Keep track for error reporting
                logger.debug(f"Workflow Event: {event}")

                # Determine the event type based on the node that produced it
                if "planner" in event:
                    plan_steps = event["planner"].get("plan", [])
                    event_data = {"type": "plan", "steps": plan_steps}
                    # Use triple quotes for multiline f-string
                    yield f"""event: plan\ndata: {json.dumps(event_data)}\n\n"""
                
                elif "agent" in event: # Corresponds to _execute_step output
                    agent_data = event["agent"]
                    past_steps = agent_data.get("past_steps")
                    if past_steps:
                        last_task, last_result = past_steps[-1]
                        event_data = {"type": "tool_result", "task": last_task, "result": last_result}
                        # Use triple quotes for multiline f-string
                        yield f"""event: tool_result\ndata: {json.dumps(event_data)}\n\n"""

                elif "replan" in event:
                    replan_data = event["replan"]
                    if "response" in replan_data and replan_data["response"]:
                        final_response = replan_data["response"]
                        event_data = {"type": "final_response", "response": final_response}
                        # Use triple quotes for multiline f-string
                        yield f"""event: final_response\ndata: {json.dumps(event_data)}\n\n"""
                        final_response_yielded = True # Mark as yielded
                    elif "plan" in replan_data and replan_data["plan"]:
                        new_plan_steps = replan_data["plan"]
                        event_data = {"type": "new_plan", "steps": new_plan_steps}
                        # Use triple quotes for multiline f-string
                        yield f"""event: new_plan\ndata: {json.dumps(event_data)}\n\n"""
                    else: pass 

                elif END in event:
                    if not final_response_yielded and isinstance(event[END], dict) and "response" in event[END]:
                        final_response = event[END]["response"]
                        event_data = {"type": "final_response", "response": final_response}
                        # Use triple quotes for multiline f-string
                        yield f"""event: final_response\ndata: {json.dumps(event_data)}\n\n"""
                        final_response_yielded = True
                    logger.info("Workflow reached END state.")

        except Exception as e:
            logger.error(f"Exception during agent workflow stream: {e}", exc_info=True)
            logger.error(f"Last event before exception: {last_event_for_error}")
            error_data = {"type": "error", "message": str(e)}
            # Use triple quotes for multiline f-string
            yield f"""event: error\ndata: {json.dumps(error_data)}\n\n"""
        finally:
            logger.info(f"--- Agent stream finished for Input: {user_input} ---")
            # Use triple quotes for multiline f-string
            yield f"""event: end\ndata: {json.dumps({'message': 'Stream ended.'})}\n\n""" 
            

    # --- arun method (keep for non-streaming use cases or remove if unused) ---
    async def arun(self, user_input: str):
         # ... (existing arun implementation) ...
         # This method now exists alongside astream_events
         # It iterates through the stream internally but only returns the final result.
        logger.info(f"--- [arun] Starting agent run for Input: {user_input} ---")
        config = {"recursion_limit": 50}
        final_state = None
        last_event = None
        try:
            async for event in self.app.astream(
                {"input": user_input, "plan": [], "past_steps": [], "response": None, "intermediate_responses": []},
                config=config,
            ):
                last_event = event # Keep track of the last event for error reporting
                logger.debug(f"[arun] Workflow Event: {event}")
                if END in event:
                    final_state = event[END]
                    logger.info("[arun] Workflow reached END state.")
        except Exception as e:
             logger.error(f"[arun] Exception during agent workflow execution: {e}", exc_info=True)
             logger.error(f"[arun] Last event before exception: {last_event}")
             return f"An error occurred during agent execution: {e}"


        if final_state and "response" in final_state:
            logger.info(f"--- [arun] Agent run finished. Final Response: {final_state['response']} ---")
            return final_state['response']
        else:
            logger.warning(f"[arun] Agent finished without a final response state. Last event: {last_event}")
            last_state_info = "Unknown state"
            if last_event:
                try:
                     last_state_info = str(list(last_event.values())[0])
                except Exception:
                    pass 
            logger.warning(f"[arun] Last state info: {last_state_info}")
            return "Agent finished unexpectedly without providing a final response."
