# LocalRAGent 🤖

RAG (Retrieval-Augmented Generation) system built with LangChain, LangGraph, and Llama 3.1, featuring hybrid search, document grading, and routing capabilities.


## 📁 Directory Structure

```
LocalRAGent/
├── api/                        # FastAPI REST API
│   ├── main.py                # API entry point
│   ├── models/                # Pydantic models
│   │   └── schemas.py         # Request/response schemas
│   └── routers/               # API route handlers
│       ├── chat.py            # Chat endpoints
│       └── health.py          # Health check endpoints
├── config/                     # Configuration files
│   ├── settings.py            # Application settings
│   └── prompts.py             # LLM prompts
├── data/                       # Data storage
│   ├── processed/             # Processed documents
│   ├── raw/                   # Raw input data
│   ├── static/                # Static assets
│   └── vectorstore/           # ChromaDB vector storage
├── src/                        # Core application code
│   ├── agents/                # Intelligent agents
│   │   ├── rag_agent.py       # RAG generation agent
│   │   ├── router_agent.py    # Query routing agent
│   │   └── web_search_agent.py # Web search agent
│   ├── core/                  # Core functionality
│   │   ├── document_processor.py # Document processing
│   │   ├── embeddings.py      # Embedding management
│   │   ├── llm_client.py      # LLM client wrapper
│   │   └── retriever.py       # Hybrid retrieval system
│   ├── graders/               # Quality assessment
│   │   ├── answer_grader.py   # Answer quality grading
│   │   ├── hallucination_grader.py # Hallucination detection
│   │   └── relevance_grader.py # Document relevance grading
│   ├── utils/                 # Utility functions
│   │   ├── document_utils.py  # Document helpers
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── logging_config.py  # Logging configuration
│   └── workflow/              # LangGraph workflow
│       ├── edges.py           # Workflow edge logic
│       ├── graph_state.py     # State management
│       ├── nodes.py           # Workflow nodes
│       └── workflow_builder.py # Workflow construction
├── scripts/                    # Utility scripts
│   └── setup_vectorstore.py  # Vector store initialization
├── tests/                      # Test suite
├── logs/                       # Application logs
├── main.py                     # CLI entry point
└── requirements.txt           # Python dependencies
```

### Prerequisites
- Python 3.8+
- Ollama (for local Llama 3.1 model)
- API keys for FireCrawl and Tavily

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SohamD34/LocalRAGent.git
cd LocalRAGent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
LANGCHAIN_API_KEY=your_langchain_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
TAVILY_API_KEY=your_tavily_api_key
```

4. **Install and start Ollama**
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.1 model
ollama pull llama3.1:8b
```

5. **Initialize the vector store**
```bash
python scripts/setup_vectorstore.py
```

### Usage Options

#### 1. **Command Line Interface**
```bash
python main.py
```

#### 2. **REST API Server**
```bash
# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is prompt engineering?"}'
```

#### 3. **Python Integration**
```python
from config.settings import Settings
from src.workflow.workflow_builder import RAGWorkflowBuilder

# Initialize the system
settings = Settings()
builder = RAGWorkflowBuilder(settings)
app = builder.build_workflow()

# Query the system
inputs = {"question": "How to reduce LLM costs?"}
for output in app.stream(inputs):
    print(output)
```
