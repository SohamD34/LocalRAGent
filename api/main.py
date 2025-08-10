from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from config.settings import Settings
from src.workflow.workflow_builder import RAGWorkflowBuilder
from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging("INFO")

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="A hybrid RAG system with semantic search, keyword search, and reranking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow app
workflow_app = None

class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 1000

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list = []

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG workflow on startup"""
    global workflow_app
    try:
        logger.info("Initializing RAG workflow...")
        settings = Settings()
        workflow_builder = RAGWorkflowBuilder(settings)
        workflow_app = workflow_builder.build_workflow()
        logger.info("RAG workflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize workflow: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG system is running"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG system"""
    if not workflow_app:
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        inputs = {"question": request.question}
        
        # Get the final output
        final_output = None
        for output in workflow_app.stream(inputs):
            for key, value in output.items():
                final_output = value
        
        if not final_output or "generation" not in final_output:
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
        # Extract sources if available
        sources = []
        if "documents" in final_output and final_output["documents"]:
            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                for doc in final_output["documents"][:3]  # Top 3 sources
            ]
        
        return QuestionResponse(
            question=request.question,
            answer=final_output["generation"],
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the RAG System API",
        "docs": "/docs",
        "health": "/health",
        "ask_endpoint": "/ask"
    }
