import logging
from typing import List
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from config.prompts import PromptTemplates
from src.utils.document_utils import format_docs

class RAGAgent:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Setup RAG chain
        self.prompt = PromptTemplate(
            template=PromptTemplates.ANSWER_GENERATOR,
            input_variables=["context", "question"]
        )
        self.rag_chain = self.prompt | self.llm | StrOutputParser()
    
    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generate answer using RAG on retrieved documents"""
        try:
            self.logger.info(f"Generating answer for question: {question[:100]}...")
            
            # Format documents for context
            context = format_docs(documents)
            
            # Generate answer
            generation = self.rag_chain.invoke({
                "context": context, 
                "question": question
            })
            
            self.logger.info("Answer generated successfully")
            return generation
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
