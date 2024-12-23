from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from utils import format_docs
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

urls = [
        "https://www.ai-jason.com/learning-ai/how-to-reduce-llm-cost",
        "https://www.ai-jason.com/learning-ai/gpt5-llm",
        "https://www.ai-jason.com/learning-ai/how-to-buid-ai-agent-tutorial-3",
]

# Web crawling of documents

docs = [FireCrawlLoader(api_key=firecrawl_api_key, url=url, mode="scrape").load() for url in urls]
doc_list = [items for sublist in docs for items in sublist]


# Splitting docs

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(doc_list)


# Filtering out complex metadata

filtered_docs = []
for doc in doc_splits:
    
    if isinstance(doc, Document) and hasattr(doc, 'metadata'):
        cleaned_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
        filtered_docs.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))


# Vectorizing docs and adding to ChromaDB (vectorstore)

vectorstore = Chroma.from_documents(
    documents=filtered_docs,
    collection_name='rag-chroma',
    embedding=GPT4AllEmbeddings(),
)

retriever = vectorstore.as_retriever()


# Chat with Ollama

llm = ChatOllama(model='llama3.1:8b', format='json', temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader, assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The gial is to filter out erronous retrievals.\n
    Give a binary score - 'yes' or 'no' - to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eat_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "How to save llm cost?"
docs = retriever.invoke(question)
doc_text = docs[1].page_content

print(retrieval_grader.invoke({"question":question, "document":doc_text}))


# GENERATE ANSWERS

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. Use the following pices of retrieved context
    to answer the questions. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum ans keep the answet concise. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["context","question"]
)

llm = ChatOllama(model='llama3.1:8b', format='json', temperature=0)

rag_chain = prompt | llm | StrOutputParser()

question = "How to save LLM cost?"
docs = retriever.invoke(question)
generation = rag_chain.invoke({"question":question, "context":format_docs(docs)})
print(generation)


# HALLUCINATION GRADER

grader_llm = ChatOllama(model='llama3.1:8b', format='json', temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader, assessing whether an answer is usefyl to resolve a question. Give a binary 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
    Probide the binary score as a  JSON with a single key 'score' ans no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "question"])


hallucination_grader = prompt | grader_llm | JsonOutputParser()
hallucination_grader.invoke({"question":question, "generation":generation})


# ANSWER GRADER

llm = ChatOllama(model='llama3.1:8b', format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     
    Here is the answer:
    {generation} 

    Here is the question: {question}
    """,
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({"question": question,"generation": generation})


### Router

# LLM
llm = ChatOllama(model='llama3.1:8b', format="json")

prompt = PromptTemplate(
    template="""You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
    prompt engineering, prompting, and adversarial attacks. You can also use words that are similar to those, 
    no need to have exactly those words. Otherwise, use web-search. 

    Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no preamble or explanation.
    
    Examples:
    Question: When will the Euro of Football take place?
    Answer: {{"datasource": "web_search"}}

    Question: What are the types of agent memory?
    Answer: {{"datasource": "vectorstore"}}

    Question: What are the basic approaches for prompt engineering?
    Answer: {{"datasource": "vectorstore"}}

    Question: What is prompt engineering?
    Answer: {{"datasource": "vectorstore"}}
    
    Question to route: 
    {question}""",
    input_variables=["question"],
)


question_router = prompt | llm | JsonOutputParser()

print(question_router.invoke({"question": "When will the Euro of Football take place?"}))
print(question_router.invoke({"question": "What are the types of agent memory?"})) ### Index

print(question_router.invoke({"question": "What are the basic approaches for prompt engineering?"})) 


# WEB SEARCH 

web_search_tool = TavilySearchResults(api_key=tavily_api_key, k=3)




# LangGraph - Setup States and Nodes

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

### Conditional edge

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})  
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

### Conditional edge

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    print(state)
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation, "question": question})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        print(f'question: {question}, documents: {documents}, generation: {generation}, grade: {grade}')
        return "not supported"

workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generate



# GRAPH BUILD

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)


# Compile
app = workflow.compile()

# Test
from pprint import pprint
inputs = {"question": "What is prompt engineering?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])