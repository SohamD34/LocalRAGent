import logging
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from config.prompts import PromptTemplates

class RouterAgent:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Setup router chain
        self.prompt = PromptTemplate(
            template=PromptTemplates.QUESTION_ROUTER,
            input_variables=["question"]
        )
        self.router_chain = self.prompt | self.llm | JsonOutputParser()
    
    def route_question(self, question: str) -> str:
        """Route question to appropriate datasource"""
        try:
            self.logger.info(f"Routing question: {question[:100]}...")
            
            result = self.router_chain.invoke({"question": question})
            datasource = result.get('datasource', 'vectorstore')
            
            self.logger.info(f"Routed to: {datasource}")
            return datasource
            
        except Exception as e:
            self.logger.error(f"Error routing question: {e}")
            return "vectorstore"  # Default fallback
