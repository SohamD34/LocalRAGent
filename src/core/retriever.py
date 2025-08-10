import logging
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import numpy as np

class HybridRetriever:
    def __init__(
        self, 
        vectorstore: Chroma, 
        documents: List[Document],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.vectorstore = vectorstore
        self.documents = documents
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.logger = logging.getLogger(__name__)
        
        # Initialize retrievers
        self.semantic_retriever = vectorstore.as_retriever()
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        
        # Initialize ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.keyword_retriever],
            weights=[semantic_weight, keyword_weight]
        )
        
        # Initialize reranker
        self.reranker = CrossEncoder(rerank_model)
        self.logger.info(f"Initialized hybrid retriever with semantic_weight={semantic_weight}")
    
    def retrieve_and_rerank(
        self, 
        query: str, 
        top_k: int = 10, 
        final_k: int = 5
    ) -> List[Document]:
        """Retrieve documents using hybrid approach and rerank them"""
        try:
            # Get initial retrieval results
            self.logger.info(f"Retrieving top {top_k} documents for query: {query[:100]}...")
            
            # Use ensemble retriever to get diverse results
            retrieved_docs = self.ensemble_retriever.get_relevant_documents(query)[:top_k]
            
            if not retrieved_docs:
                self.logger.warning("No documents retrieved")
                return []
            
            # Rerank documents
            self.logger.info(f"Reranking {len(retrieved_docs)} documents...")
            reranked_docs = self._rerank_documents(query, retrieved_docs, final_k)
            
            self.logger.info(f"Returning top {len(reranked_docs)} reranked documents")
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to semantic retrieval only
            return self.semantic_retriever.get_relevant_documents(query)[:final_k]
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[Document]:
        """Rerank documents using cross-encoder"""
        try:
            # Prepare query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get relevance scores
            scores = self.reranker.predict(pairs)
            
            # Sort documents by relevance score
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k documents
            reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
            
            # Log scores for debugging
            for i, (doc, score) in enumerate(doc_score_pairs[:top_k]):
                self.logger.debug(f"Rank {i+1}: Score {score:.4f} - {doc.page_content[:100]}...")
            
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Error in reranking: {e}")
            # Fallback to original order
            return documents[:top_k]