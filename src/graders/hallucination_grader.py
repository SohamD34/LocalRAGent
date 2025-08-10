from src.graders.base_grader import BaseGrader
from config.prompts import PromptTemplates

class HallucinationGrader(BaseGrader):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            prompt_template=PromptTemplates.HALLUCINATION_GRADER,
            input_variables=["documents", "generation", "question"]
        )
    
    def grade(self, documents: str, generation: str, question: str) -> dict:
        """Grade if generation is grounded in documents"""
        return self._safe_grade(
            documents=documents, 
            generation=generation, 
            question=question
        )