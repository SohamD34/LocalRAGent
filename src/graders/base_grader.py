from abc import ABC, abstractmethod
import logging
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from src.utils.exceptions import GradingError

class BaseGrader(ABC):
    def __init__(self, llm: ChatOllama, prompt_template: str, input_variables: list):
        self.llm = llm
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables
        )
        self.grader = self.prompt | self.llm | JsonOutputParser()
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def grade(self, **kwargs) -> dict:
        """Grade the input and return score"""
        pass
    
    def _safe_grade(self, **kwargs) -> dict:
        """Safely perform grading with error handling"""
        try:
            result = self.grader.invoke(kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Grading error: {e}")
            raise GradingError(f"Failed to grade: {e}")