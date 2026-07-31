import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent.graph import agent_app

# 1. Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Order Tracking Agent API",
    description="Backend API for the Order Tracking AI Agent with Logging",
    version="1.0.0"
)

class HealthResponse(BaseModel):
    status: str
    message: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 2. Add a middleware to log all incoming HTTP requests and their processing time
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Time: {process_time:.4f}s")
    return response

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint to ensure the API is running."""
    return {"status": "ok", "message": "Order Tracking Agent API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Takes a user message, passes it to the LangGraph agent, 
    and returns the AI's response.
    """
    logger.info(f"Received chat request: {request.message}")
    
    try:
        input_message = HumanMessage(content=request.message)
        
        # The agent invoke will now automatically be traced by LangSmith 
        # because of the environment variables in .env
        result = agent_app.invoke({"messages": [input_message]})
        
        final_message = result["messages"][-1].content
        logger.info("Successfully generated agent response.")
        
        return {"response": final_message}
        
    except Exception as e:
        logger.error(f"Error during agent execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))