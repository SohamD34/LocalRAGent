import logging
from langgraph.graph import END, StateGraph
from config.settings import Settings
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import EmbeddingManager
from src.core.llm_client import LLMClient
from src.core.hybrid_retriever import HybridRetriever
from src.agents.rag_agent import RAGAgent
from src.agents.web_search_agent import WebSearchAgent
from src.agents.router_agent import RouterAgent
from src.graders.relevance_grader import RelevanceGrader
from src.graders.hallucination_grader import HallucinationGrader
from src.graders.answer_grader import AnswerGrader
from src.workflow.graph_state import GraphState
from src.workflow.nodes import WorkflowNodes
from src.workflow.edges import WorkflowEdges

class RAGWorkflowBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all components"""
        self.logger.info("Initializing RAG workflow components...")
        
        # Core components
        self.document_processor = DocumentProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            firecrawl_api_key=self.settings.firecrawl_api_key
        )
        
        self.embedding_manager = EmbeddingManager(
            persist_directory=self.settings.vectorstore_path
        )
        
        self.llm_client = LLMClient(
            model=self.settings.llm_model,
            temperature=self.settings.temperature
        )
        
        # Setup vectorstore and hybrid retriever
        self._setup_vectorstore()
        
        # Agents
        self.rag_agent = RAGAgent(self.llm_client.get_regular_llm())
        self.web_search_agent = WebSearchAgent(self.settings.tavily_api_key)
        self.router_agent = RouterAgent(self.llm_client.get_json_llm())
        
        # Graders
        json_llm = self.llm_client.get_json_llm()
        self.relevance_grader = RelevanceGrader(json_llm)
        self.hallucination_grader = HallucinationGrader(json_llm)
        self.answer_grader = AnswerGrader(json_llm)
        
        # Workflow components
        self.nodes = WorkflowNodes(
            hybrid_retriever=self.hybrid_retriever,
            rag_agent=self.rag_agent,
            web_search_agent=self.web_search_agent,
            router_agent=self.router_agent,
            relevance_grader=self.relevance_grader,
            hallucination_grader=self.hallucination_grader,
            answer_grader=self.answer_grader
        )
        
        self.edges = WorkflowEdges(
            router_agent=self.router_agent,
            hallucination_grader=self.hallucination_grader,
            answer_grader=self.answer_grader
        )
        
        self.logger.info("All components initialized successfully")
    
    def _setup_vectorstore(self):
        """Setup vectorstore and hybrid retriever"""
        try:
            # Try to load existing vectorstore
            vectorstore = self.embedding_manager.load_vectorstore(
                collection_name=self.settings.collection_name
            )
            self.logger.info("Loaded existing vectorstore")
            
            # For hybrid retriever, we need the original documents
            # In production, you'd store these separately or recreate them
            documents = self._get_documents_for_hybrid_search()
            
        except Exception as e:
            self.logger.info("Creating new vectorstore...")
            
            # Process documents
            documents = self.document_processor.crawl_urls(self.settings.default_urls)
            doc_splits = self.document_processor.split_documents(documents)
            filtered_docs = self.document_processor.filter_metadata(doc_splits)
            
            # Create vectorstore
            vectorstore = self.embedding_manager.create_vectorstore(
                documents=filtered_docs,
                collection_name=self.settings.collection_name
            )
            
            documents = filtered_docs
        
        # Setup hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            documents=documents,
            semantic_weight=self.settings.semantic_weight,
            keyword_weight=self.settings.keyword_weight
        )
    
    def _get_documents_for_hybrid_search(self):
        """Get documents for hybrid search - in production, store these separately"""
        # For now, recreate documents - in production, you'd cache these
        try:
            documents = self.document_processor.crawl_urls(self.settings.default_urls)
            doc_splits = self.document_processor.split_documents(documents)
            filtered_docs = self.document_processor.filter_metadata(doc_splits)
            return filtered_docs
        except Exception as e:
            self.logger.warning(f"Could not recreate documents for hybrid search: {e}")
            return []
    
    def build_workflow(self):
        """Build and compile the LangGraph workflow"""
        self.logger.info("Building LangGraph workflow...")
        
        # Create workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("websearch", self.nodes.web_search)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        
        # Set entry point
        workflow.set_conditional_entry_point(
            self.edges.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        
        # Add edges
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.edges.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.edges.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )
        
        # Compile workflow
        app = workflow.compile()
        self.logger.info("Workflow compiled successfully")
        
        return app