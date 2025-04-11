from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import AIMessage
import os
import json
import re

@tool
def portfolio_retriever(prompt: str) -> str:
    """Retrieves portfolio information. Information returned must be information on the portfolio."""
    print("Using Portfolio Retriever tool now")
    return "0"
