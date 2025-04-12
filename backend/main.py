from fastapi import FastAPI
from .agents import MultiStepAgent
from .schemas import QueryRequest, QueryResponse

app = FastAPI()
agent = MultiStepAgent()

@app.post("/invoke", response_model=QueryResponse)
async def invoke_agent(request: QueryRequest):
    """Receives a user query and returns the agent's response."""
    agent_response = await agent.arun(request.query)
    return {"response": agent_response}

@app.get("/")
async def read_root():
    # Keep the root endpoint for basic checks if needed
    return {"message": "Planning Agent API is running."}