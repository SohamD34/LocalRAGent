class PromptTemplates:
    RETRIEVAL_GRADER = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader, assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score - 'yes' or 'no' - to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: {document}
    Here is the user question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    ANSWER_GENERATOR = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the questions. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    HALLUCINATION_GRADER = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader, assessing whether an answer is grounded in / supported by a set of retrieved facts. 
    Give a binary 'yes' or 'no' score to indicate whether the answer is supported by the retrieved facts.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts: {documents}
    Here is the answer: {generation}
    Here is the question: {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    ANSWER_GRADER = """You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     
    Here is the answer: {generation}
    Here is the question: {question}"""

    QUESTION_ROUTER = """You are an expert at routing a 
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
    
    Question to route: {question}"""