import logging
from typing import Dict, Any
from src.agents.router_agent import RouterAgent
from src.graders.hallucination_grader import HallucinationGrader
from src.graders.answer_grader import AnswerGrader
from src.utils.document_utils import format_docs

class WorkflowEdges:
    def __init__(
        self,
        router_agent: RouterAgent,
        hallucination_grader: HallucinationGrader,
        answer_grader: AnswerGrader
    ):
        self.router_agent = router_agent
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.logger = logging.getLogger(__name__)
    
    def route_question(self, state: Dict[str, Any]) -> str:
        """Route question to web search or RAG"""
        self.logger.info("---ROUTE QUESTION---")
        question = state["question"]
        
        datasource = self.router_agent.route_question(question)
        
        if datasource == 'web_search':
            self.logger.info("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        else:
            self.logger.info("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    
    def decide_to_generate(self, state: Dict[str, Any]) -> str:
        """Determine whether to generate answer or search web"""
        self.logger.info("---ASSESS GRADED DOCUMENTS---")
        web_search = state.get("web_search", "No")
        
        if web_search == "Yes":
            self.logger.info("---DECISION: DOCUMENTS NOT RELEVANT, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            self.logger.info("---DECISION: GENERATE---")
            return "generate"
    
    def grade_generation_v_documents_and_question(self, state: Dict[str, Any]) -> str:
        """Grade generation against documents and question"""
        self.logger.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Format documents for grading
        documents_text = format_docs(documents)
        
        # Check for hallucinations
        hallucination_score = self.hallucination_grader.grade(
            documents=documents_text,
            generation=generation,
            question=question
        )
        hallucination_grade = hallucination_score.get('score', 'no')
        
        if hallucination_grade.lower() == "yes":
            self.logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            
            # Check if answer addresses the question
            self.logger.info("---GRADE GENERATION vs QUESTION---")
            answer_score = self.answer_grader.grade(
                question=question,
                generation=generation
            )
            answer_grade = answer_score.get('score', 'no')
            
            if answer_grade.lower() == "yes":
                self.logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                self.logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            self.logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"