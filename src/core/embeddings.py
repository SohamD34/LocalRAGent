import logging
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from src.utils.exceptions import EmbeddingError

class EmbeddingManager:
    def __init__(self, embedding_model: str = "gpt4all", persist_directory: str = None):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.embeddings = GPT4AllEmbeddings()
        self.logger = logging.getLogger(__name__)
    
    def create_vectorstore(
        self, 
        documents: List[Document], 
        collection_name: str = "rag-chroma"
    ) -> Chroma:
        """Create and populate vector store"""
        try:
            self.logger.info(f"Creating vectorstore with {len(documents)} documents...")
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            self.logger.info("Vectorstore created successfully")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error creating vectorstore: {e}")
            raise EmbeddingError(f"Failed to create vectorstore: {e}")
    
    def load_vectorstore(self, collection_name: str = "rag-chroma") -> Chroma:
        """Load existing vectorstore"""
        try:
            self.logger.info(f"Loading existing vectorstore: {collection_name}")
            
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            self.logger.info("Vectorstore loaded successfully")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error loading vectorstore: {e}")
            raise EmbeddingError(f"Failed to load vectorstore: {e}")
