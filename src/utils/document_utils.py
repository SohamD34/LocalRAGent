from typing import List
from langchain.docstore.document import Document

def format_docs(docs: List[Document]) -> str:
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def filter_complex_metadata(documents: List[Document]) -> List[Document]:
    """Filter out complex metadata that can't be serialized"""
    filtered_docs = []
    for doc in documents:
        if isinstance(doc, Document) and hasattr(doc, 'metadata'):
            cleaned_metadata = {
                k: v for k, v in doc.metadata.items() 
                if isinstance(v, (str, int, float, bool))
            }
            filtered_docs.append(Document(
                page_content=doc.page_content, 
                metadata=cleaned_metadata
            ))
    return filtered_docs