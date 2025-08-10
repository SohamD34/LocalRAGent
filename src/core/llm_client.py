import logging
from langchain_community.chat_models import ChatOllama
from src.utils.exceptions import RAGSystemError

class LLMClient:
    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize different LLM clients for different tasks"""
        try:
            # JSON format LLM for grading tasks
            self.json_llm = ChatOllama(
                model=self.model,
                format='json',
                temperature=self.temperature
            )
            
            # Regular LLM for generation tasks
            self.regular_llm = ChatOllama(
                model=self.model,
                temperature=self.temperature
            )
            
            self.logger.info(f"Initialized LLM clients with model: {self.model}")
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM clients: {e}")
            raise RAGSystemError(f"Failed to initialize LLM: {e}")
    
    def get_json_llm(self) -> ChatOllama:
        """Get LLM configured for JSON output"""
        return self.json_llm
    
    def get_regular_llm(self) -> ChatOllama:
        """Get regular LLM for text generation"""
        return self.regular_llm