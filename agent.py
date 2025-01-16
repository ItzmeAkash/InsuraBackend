from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_groq.chat_models import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import initialize_agent

# Initialize FastAPI ap
app = FastAPI()

# Load environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

# Initialize LLM and tools
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

search = DuckDuckGoSearchRun()

tools = [chatgroq_tool, search]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define API endpoint
@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        response = agent.run(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "ChatGroq API is running"}
