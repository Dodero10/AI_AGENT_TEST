from langgraph.graph import StateGraph, END, START
from pprint import pprint
from langchain.schema import Document
from typing_extensions import TypedDict
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

load_dotenv()

embeddings = OpenAIEmbeddings()
loader = CSVLoader("Agentic RAG/data/data.csv", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(k=5)

# Document Grade


class GradeDocuments(BaseModel):
    binary_grade: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'")


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt for the grader
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human",
         "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# run grader
question = "Các dự án về học tập gồm những gì"
docs = retriever.invoke(question)

# RAG Chain
template = """"
You are a helpful assistant that answers questions based on the following context.'
Use the provided context to answer the question.
Context: {context}
Question: {question}
Answer:

"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

# response
generation = rag_chain.invoke({"context": docs, "question": question})


# Web Search
web_search_tool = TavilySearchResults(k=3)


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]


# node function to retrieve documents


def retrieve_documents(state):
    print("---RETRIEVING DOCUMENTS---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": question}

# node function to generation


def generation_node(state):
    print("---GENERATION---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# node function to web search


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    web_search = web_search.invoke({"question": question})

# node function for check_relevance


def grade_documents(state):

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_grade
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# node function for web search
def web_search(state):

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

# node function for decision


def decide_to_generate(state):

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        print("---DECISION: WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve_documents)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generation_node)  # generatae
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search_node",
        "generate": "generate",
    },
)

workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# save graph
png_data = app.get_graph().draw_mermaid_png()
with open("crag_graph.png", "wb") as f:
    f.write(png_data)

# example 1 where documents are relevant

inputs = {"question": "how does interlibrary loan work"}
print("---EXAMPLE 1---")
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")


pprint(value["generation"])


# example 2 where documents are not relevant

inputs = {"question": "Các dự án về học tập gồm những gì?"}
print("---EXAMPLE 2---")
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value["generation"])
