from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated, List, Tuple, Optional
from typing_extensions import TypedDict
from .config import API_KEY, LLM_MODEL
from .prompts import PLANNER_PROMPT, EXECUTOR_PROMPT, REPLANNER_PROMPT
from .tools import portfolio_retriever
from .schemas import Plan, PlanExecute, Act, Response

# --- Mis ---
def clean_newlines(text: str) -> str:
    """Removes extra newline characters from a string."""
    return text.replace("\n\n", "\n")


class MultiStepAgent:
    def __init__(self):
        # --- LLM and Tools ---
        self.llm = ChatVertexAI(model_name=LLM_MODEL, temperature=0)
        self.tools = [portfolio_retriever] # Add other tools here if needed
        self.agent_executor = create_react_agent(self.llm, self.tools, state_modifier=EXECUTOR_PROMPT)

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
        self.replanner = self.replanner_prompt | self.llm.with_structured_output(Act)

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

    # --- Workflow Nodes ---
    async def _plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke({"messages": [("user", state["input"])]})
        # Note: Removed Chainlit message sending
        return {"plan": plan.steps, "intermediate_responses": []}

    async def _execute_step(self, state: PlanExecute):
        plan = state["plan"]
        if not plan:
            return {"response": "No more steps in the plan."}

        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        step_number = len(state.get("past_steps", [])) + 1
        task_formatted = f"""For the following plan: {plan_str}\n\nYou are tasked with executing step {step_number}, {task}."""

        # Note: Removed Chainlit Step/Message sending
        agent_response = await self.agent_executor.ainvoke({"messages": [("user", task_formatted)]})
        final_response = agent_response["messages"][-1].content

        return {
            "past_steps": state.get("past_steps", []) + [(task, final_response)],
            "plan": plan[1:],
            "intermediate_responses": state.get("intermediate_responses", []) + [final_response]
        }

    async def _replan_step(self, state: PlanExecute):
        all_responses = "\n".join(state["intermediate_responses"])
        all_steps = "\n".join([f"{step}: {response}" for step, response in state["past_steps"]])
        context = f"Here is the information gathered from the previous steps:\n{all_steps}\n\nHere are the direct responses from the tools:\n{all_responses}"

        output = await self.replanner.ainvoke({**state, "input": context})
        if output.response:
            cleaned_response = clean_newlines(output.response.response)
            # Note: Removed Chainlit Step/Message sending
            return {"response": cleaned_response}
        else:
            # Note: Replanning might need adjustments depending on how you handle intermediate steps without Chainlit
            # For now, just returning the new plan steps if any
            if output.plan:
                return {"plan": output.plan.steps}
            else:
                # Handle cases where neither response nor plan is returned if necessary
                return {"response": "Replanning resulted in no further action."}


    def _should_end(self, state: PlanExecute):
        return END if "response" in state and state["response"] is not None else "agent"

    async def arun(self, user_input: str):
        config = {"recursion_limit": 50}
        # Stream the execution and accumulate the final response
        final_state = None
        async for event in self.app.astream(
            {"input": user_input, "plan": [], "past_steps": [], "response": None, "intermediate_responses": []},
            config=config,
        ):

            if END in event:
              final_state = event[END]

        if final_state and "response" in final_state:
            return final_state["response"]
        else:
            last_event_key = list(event.keys())[0]
            last_state = event[last_event_key]
            print(f"Warning: Agent finished without a final response. Last state: {last_state}")
            return "Agent finished without a final response." # Or raise an error
