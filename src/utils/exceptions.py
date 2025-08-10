class RAGSystemError(Exception):
    """Base exception for RAG system"""
    pass

class DocumentProcessingError(RAGSystemError):
    """Raised when document processing fails"""
    pass

class EmbeddingError(RAGSystemError):
    """Raised when embedding operations fail"""
    pass

class GradingError(RAGSystemError):
    """Raised when grading operations fail"""
    pass