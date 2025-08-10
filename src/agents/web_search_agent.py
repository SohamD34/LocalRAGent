import logging
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.docstore.document import Document

class WebSearchAgent:
    def __init__(self, tavily_api_key: str, k: int = 3):
        self.web_search_tool = TavilySearchResults(api_key=tavily_api_key, k=k)
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str) -> Document:
        """Perform web search and return results as a document"""
        try:
            self.logger.info(f"Performing web search for: {query[:100]}...")
            
            # Perform web search
            search_results = self.web_search_tool.invoke({"query": query})
            
            # Combine results into a single document
            web_content = "\n".join([result["content"] for result in search_results])
            web_document = Document(page_content=web_content)
            
            self.logger.info(f"Web search completed, found {len(search_results)} results")
            return web_document
            
        except Exception as e:
            self.logger.error(f"Error in web search: {e}")
            return Document(page_content="Web search failed.")
