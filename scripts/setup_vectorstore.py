import logging
from config.settings import Settings
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import EmbeddingManager
from src.utils.logging_config import setup_logging

def main():
    """Initialize vector store with documents"""
    logger = setup_logging("INFO")
    logger.info("Setting up vectorstore...")
    
    try:
        settings = Settings()
        
        # Initialize components
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            firecrawl_api_key=settings.firecrawl_api_key
        )
        
        embedding_manager = EmbeddingManager(
            persist_directory=settings.vectorstore_path
        )
        
        # Process documents
        logger.info(f"Processing {len(settings.default_urls)} URLs...")
        documents = processor.crawl_urls(settings.default_urls)
        doc_splits = processor.split_documents(documents)
        filtered_docs = processor.filter_metadata(doc_splits)
        
        # Create vectorstore
        vectorstore = embedding_manager.create_vectorstore(
            documents=filtered_docs,
            collection_name=settings.collection_name
        )
        
        logger.info(f"Vectorstore setup complete with {len(filtered_docs)} documents")
        
    except Exception as e:
        logger.error(f"Vectorstore setup failed: {e}")
        raise

if __name__ == "__main__":
    main()