import os

import fitz  # PyMuPDF
import streamlit as st
from chain_of_agent import ChainOfAgents
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default model is now "gpt-4o-mini"
model = "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY")  # Fetch the API key from .env file

if not api_key:
    st.error("API key not found. Make sure to add it to your .env file.")
    st.stop()

# Streamlit app
st.title("Compare RAG and Chain of Agents")

# PDF file upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Extract text from PDFs
long_input = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                long_input += page.get_text()

# Input fields
query = st.text_input("Enter your query:")
context_window_size = st.number_input("Context window size:", min_value=1, value=100)

def simple_rag(query, documents):
    # Simple keyword-based retrieval
    relevant_docs = [doc for doc in documents if query.lower() in doc.lower()]
    # Concatenate relevant documents
    context = " ".join(relevant_docs)
    # Simulate a response generation
    response = f"Based on the retrieved context, the answer to your query '{query}' is: [Simulated Response]"
    return response

if st.button("Process"):
    if long_input and query:
        # Split the long input into documents (e.g., by paragraphs)
        documents = long_input.split("\n\n")
        
        # Perform simple RAG
        rag_output = simple_rag(query, documents)
        
        # Perform Chain of Agents
        chain_of_agents = ChainOfAgents(
            long_input=long_input,
            context_window_size=context_window_size,
            model=model,
            api_key=api_key,
            query=query
        )
        chain_output = chain_of_agents.process()
        
        # Display results side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("RAG Output")
            st.write(rag_output)
        
        with col2:
            st.header("Chain of Agents Output")
            st.write(chain_output)
    else:
        st.error("Please provide both the long input text and the query.")