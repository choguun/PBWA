from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Annotated
import operator
from typing_extensions import TypedDict

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Optional[str]
    intermediate_responses: List[str]

class Plan(BaseModel):
    """Plan to follow."""
    steps: List[str] = Field(description="Steps to follow, in order.")

class ReplannerOutput(BaseModel):
    """Output structure for the Replanner LLM call."""
    final_answer: Optional[str] = Field(default=None, description="The final answer to the user's query, if determined.")
    plan: Optional[Plan] = Field(default=None, description="A new plan with next steps, if further action is needed.")

class SendEthInput(BaseModel):
    to_address: str = Field(description="The recipient Ethereum address.")
    amount_eth: float = Field(description="The amount of ETH to send.")