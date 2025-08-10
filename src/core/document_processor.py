import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain.docstore.document import Document
from src.utils.document_utils import filter_complex_metadata
from src.utils.exceptions import DocumentProcessingError

class DocumentProcessor:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 0, firecrawl_api_key: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.firecrawl_api_key = firecrawl_api_key
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.logger = logging.getLogger(__name__)
    
    def crawl_urls(self, urls: List[str]) -> List[Document]:
        """Crawl and load documents from URLs"""
        try:
            self.logger.info(f"Crawling {len(urls)} URLs...")
            docs = []
            for url in urls:
                loader = FireCrawlLoader(
                    api_key=self.firecrawl_api_key, 
                    url=url, 
                    mode="scrape"
                )
                doc_batch = loader.load()
                docs.extend(doc_batch)
            
            self.logger.info(f"Successfully crawled {len(docs)} documents")
            return docs
            
        except Exception as e:
            self.logger.error(f"Error crawling URLs: {e}")
            raise DocumentProcessingError(f"Failed to crawl URLs: {e}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            self.logger.info(f"Splitting {len(documents)} documents...")
            doc_splits = self.text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(doc_splits)} document chunks")
            return doc_splits
        except Exception as e:
            self.logger.error(f"Error splitting documents: {e}")
            raise DocumentProcessingError(f"Failed to split documents: {e}")
    
    def filter_metadata(self, documents: List[Document]) -> List[Document]:
        """Clean document metadata"""
        try:
            self.logger.info("Filtering document metadata...")
            filtered_docs = filter_complex_metadata(documents)
            self.logger.info(f"Filtered {len(filtered_docs)} documents")
            return filtered_docs
        except Exception as e:
            self.logger.error(f"Error filtering metadata: {e}")
            raise DocumentProcessingError(f"Failed to filter metadata: {e}")
