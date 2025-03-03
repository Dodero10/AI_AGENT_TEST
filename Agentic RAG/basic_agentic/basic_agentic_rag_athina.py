from athina_client.keys import AthinaApiKey
from athina_client.datasets import Dataset
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import render_text_description_and_args
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.chains import RetrievalQA
from langchain.tools import Tool
# Import LangSmith for tracing
from langsmith import Client
from langsmith.run_helpers import traceable

load_dotenv()

# Initialize LangSmith client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agentic-rag-project"  # You can change this project name

questions = [
    "What things you know about AI Engineer in Viettel Software Company?",
    "What address of Viettel Software Company?",
]

# Initialize empty lists for each query
responses = []
contexts_web = []
contexts_vector = []


loader = PyPDFLoader("../data/Viettel-Software-AI-Engineer.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# Vector store
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_db")

retriever = vector_store.as_retriever()


@traceable(run_type="tool")
@tool
def vector_search_tool(query: str) -> str:
    """Use this tool to search the vector database FAISS for relevant information"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
    result = qa_chain.invoke({"query": query})

    # Store the result for later use with Athina
    contexts_vector.append(str(result))

    return result


@traceable(run_type="tool")
@tool
def web_search_tool(query: str) -> str:
    """Use this tool to search the web for relevant information"""
    web_search = TavilySearchResults(k=2, api_key=os.getenv("TAVILY_API_KEY"))
    answer = web_search.run(query)

    # Store the result for later use with Athina
    contexts_web.append(str(answer))

    return answer


tools = [
    Tool(
        name="vector_search_tool",
        description="Use this tool to search the vector database FAISS for relevant information",
        func=vector_search_tool
    ),
    Tool(
        name="web_search_tool",
        description="Use this tool to search the web for relevant information",
        func=web_search_tool
    )
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
Always try the \"VectorStoreSearch\" tool first. Only use \"WebSearch\" if the vector store does not contain the required information.
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:"
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""
human_prompt = """{input}
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

prompt = prompt.partial(
    tools=render_text_description_and_args(tools),
    tool_names=", ".join([t.name for t in tools])
)


chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

agent_output = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

AthinaApiKey.set_key(os.environ['ATHINA_API_KEY'])

# Process each query with rate limiting and proper error handling
for query in questions:
    query_response = []
    query_contexts_web = []
    query_contexts_vector = []

    # Reset global lists for this query
    contexts_web = []
    contexts_vector = []

    try:
        time.sleep(2)

        # Add a trace name for LangSmith
        with Client().trace(
            name=f"Agentic RAG Query: {query[:50]}...",
            project_name=os.getenv("LANGCHAIN_PROJECT")
        ) as tracer:
            result = agent_output.invoke({"input": query})
            response_str = str(result["output"])

        # Copy the collected contexts
        query_contexts_web = [str(item) for item in contexts_web]
        query_contexts_vector = [str(item) for item in contexts_vector]

        # Create a row for Athina with proper data types
        row = {
            'query': query,
            'context_vector': query_contexts_vector,
            'context_web': query_contexts_web,
            'response': response_str  
        }

        try:
            Dataset.add_rows(
                dataset_id='aa77e83a-147b-47c5-91ad-ca344189b8d9',
                rows=[row]
            )
            print(f"Successfully added data for query: {query}")
        except Exception as e:
            print(f"Failed to add rows to Athina: {e}")
            print(
                f"Data types: query={type(query)}, "
                f"context_vector={type(query_contexts_vector)}, "
                f"context_web={type(query_contexts_web)}, "
                f"response={type(response_str)}"
            )

    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        if "ResourceExhausted" in str(e) or "429" in str(e):
            print("Rate limit hit. Waiting 60 seconds before continuing...")
            time.sleep(60)
