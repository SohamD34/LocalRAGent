from src.graders.base_grader import BaseGrader
from config.prompts import PromptTemplates

class RelevanceGrader(BaseGrader):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            prompt_template=PromptTemplates.RETRIEVAL_GRADER,
            input_variables=["question", "document"]
        )
    
    def grade(self, question: str, document: str) -> dict:
        """Grade document relevance to question"""
        return self._safe_grade(question=question, document=document)