from src.graders.base_grader import BaseGrader
from config.prompts import PromptTemplates

class AnswerGrader(BaseGrader):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            prompt_template=PromptTemplates.ANSWER_GRADER,
            input_variables=["generation", "question"]
        )
    
    def grade(self, generation: str, question: str) -> dict:
        """Grade if answer is useful for the question"""
        return self._safe_grade(generation=generation, question=question)
