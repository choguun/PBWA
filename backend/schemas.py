from pydantic import BaseModel
from typing import List, Annotated, Tuple, Optional
import operator
from typing_extensions import TypedDict
from pydantic import Field

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class Plan(BaseModel):
    """Plan to follow."""
    steps: List[str] = Field(description="Steps to follow, in order.")

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Optional[str]
    intermediate_responses: List[str]

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform (for replanning)."""
    response: Optional[Response] = None
    plan: Optional[Plan] = None