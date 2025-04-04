from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
import os
from langchain_groq.chat_models import ChatGroq
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import initialize_agent,load_tools
import logging

router = APIRouter()

# # Load environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

# Initialize Chat LLM and tools
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    api_key=groq_api_key,
    groq_proxy=None
)

@tool
def chatgroq_tool(prompt: str) -> str:
    """
    Utilize the ChatGroq model to generate a response based on the given prompt.
    """
    return llm.generate(prompt)

# search = DuckDuckGoSearchRun()

tools = load_tools(["google-serper"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

# Define request model
class QueryRequest(BaseModel):
    message: str


@router.post("/ask/")
async def ask(request: QueryRequest):
    try:
        response = agent.invoke(request.message)
        print(response)
        return {"response": response['output']}
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Health check endpoint
@router.get("/")
async def root():
    return {"message": "Welcome to Insura"}
