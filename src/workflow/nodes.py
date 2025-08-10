import logging
from typing import Dict, Any
from src.agents.rag_agent import RAGAgent
from src.agents.web_search_agent import WebSearchAgent
from src.agents.router_agent import RouterAgent
from src.graders.relevance_grader import RelevanceGrader
from src.graders.hallucination_grader import HallucinationGrader
from src.graders.answer_grader import AnswerGrader
from src.core.hybrid_retriever import HybridRetriever
from src.utils.document_utils import format_docs

class WorkflowNodes:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        rag_agent: RAGAgent,
        web_search_agent: WebSearchAgent,
        router_agent: RouterAgent,
        relevance_grader: RelevanceGrader,
        hallucination_grader: HallucinationGrader,
        answer_grader: AnswerGrader
    ):
        self.hybrid_retriever = hybrid_retriever
        self.rag_agent = rag_agent
        self.web_search_agent = web_search_agent
        self.router_agent = router_agent
        self.relevance_grader = relevance_grader
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve documents from vectorstore using hybrid search"""
        self.logger.info("---RETRIEVE---")
        question = state["question"]
        
        # Use hybrid retrieval with reranking
        documents = self.hybrid_retriever.retrieve_and_rerank(
            query=question,
            top_k=10,
            final_k=5
        )
        
        return {"documents": documents, "question": question}
    
    def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using RAG on retrieved documents"""
        self.logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Generate answer using RAG agent
        generation = self.rag_agent.generate_answer(question, documents)
        
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation
        }
    
    def grade_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Grade document relevance and determine if web search is needed"""
        self.logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        web_search = "No"
        
        for doc in documents:
            score = self.relevance_grader.grade(
                question=question, 
                document=doc.page_content
            )
            grade = score.get('score', 'no')
            
            if grade.lower() == "yes":
                self.logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                self.logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
        
        return {
            "documents": filtered_docs, 
            "question": question, 
            "web_search": web_search
        }
    
    def web_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search based on the question"""
        self.logger.info("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])
        
        # Perform web search
        web_result = self.web_search_agent.search(question)
        
        # Add web results to existing documents
        if documents:
            documents.append(web_result)
        else:
            documents = [web_result]
        
        return {"documents": documents, "question": question}
