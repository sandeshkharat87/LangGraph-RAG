from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from duckduckgo_search import DDGS


import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*randpool'
)


### load llm
llm = ChatOllama(model="cow/gemma2_tools:2b")

### Embedding Model 
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")





urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750, chunk_overlap=150, length_function = len
)

doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
DB = FAISS.from_documents(
    doc_splits, embed_model)

db_retriever = DB.as_retriever()


def _format_docs(docs):
    info = [i.page_content for i in docs]
    return "\n\n".join(info)


# from langchain.tools.retriever import create_retriever_tool

# retriever_tool = create_retriever_tool(
#     db_retriever,
#     "retrieve_blog_posts",
#     "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
# )

# my_tools = [retriever_tool]


class grade(BaseModel):
    """Binary score for relevance check."""
    
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")



### Grader Prompt
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {context} \n\n
    Here is the user question: {question} \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
    input_variables=["context", "question"],
)


### RAG Prompt
template_rag = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
If answer not in the context just say I dont know. Dont try to makeup an answer
Question: {question} 
Context: {context} 
Answer:
"""

prompt_rag = PromptTemplate.from_template(template_rag)


### LLM's
llm_soutput = llm.with_structured_output(grade) 
llm_grader = (prompt | llm_soutput)
llm_rag = (prompt_rag | llm | StrOutputParser())



### Agents
class AgentState(TypedDict):
    question : str
    docs : str
    generated_answer :  str
    similarity : str
    web_question: str


def retrieve_docs(state):
    print("--- Retriever ---")

    docs = db_retriever.invoke(state["question"])
    docs = _format_docs(docs)

    state["docs"] = docs

    return state



def relevance_check(state):

    print("---- check relevance ---")

    question = state["question"]
    docs = db_retriever.invoke(question)[0]
    score = llm_grader.invoke({"question": question, "context": docs })
    state["similarity"] = score.binary_score
    print("similarity", score)
    
    return state


def ROUTER(state):
    score = state["similarity"]
    if score == "yes":
        return "yes"
    else:
        return "no"    


def find_answer(state):
    print("--- finding answers ---")
    question = state["question"]
    docs = state["docs"]
    answer = llm_rag.invoke({"question": question, "context": docs})
    print("answer")
    state["generated_answer"] = answer 

    return state

def better_question(state):
    print("--- Better Question ---")
    question = state["question"]
    _template = """
    Find better version of given question optimized for web search so question could be serach on a web.
    while doing this make sure you dont change the actual meaning of question.
    In return just reutun Better question in json format.
    key should be `better_question`
    example:
    `better_question : improved question`
    Question:{question}
    Better Question:  
    """
    _prompt = PromptTemplate.from_template(_template)
    _llm = (_prompt | llm | JsonOutputParser())
    web_question = _llm.invoke(question)
    print(web_question)
    web_question = web_question['better_question']
    state["web_question"] = web_question
    return state
    

def web_search_tool(state):
    try:
        print("--- web search tool ---")
        question = state["better_question"]
        result = DDGS().text(question,max_results=3)
        docs = [i["body"] for i in result]
        docs = "\n\n".join(docs)
        state["docs"] = docs
        return state
    except Exception as e:  
        print("Exception: ",e)  
        return state

def pretty_print(state):
    print("--- preety print ---")
    print("Question: ",state["question"])
    print("Answer: "  ,state["generated_answer"])