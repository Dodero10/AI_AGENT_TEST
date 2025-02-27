import os
import time  # Add this import

import fitz  # PyMuPDF
import streamlit as st
from chain_of_agent import ChainOfAgents
from dotenv import load_dotenv
from rag import RAG

load_dotenv()

model = "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API key not found. Make sure to add it to your .env file.")
    st.stop()

st.title("Compare RAG and Chain of Agents")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

long_input = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                long_input += page.get_text()

    st.info(f"Total length of input text: {len(long_input)} characters")

query = st.text_input("Enter your query:")
context_window_size = st.number_input("Context window size:", min_value=1, value=100)

if st.button("Process"):
    if long_input and query:
        documents = long_input.split("\n\n")
        
        rag_start_time = time.time()
        rag = RAG()
        rag_output = rag.generate_answer(query, long_input, context_window_size)
        rag_time = time.time() - rag_start_time
        
        chain_start_time = time.time()
        chain_of_agents = ChainOfAgents(
            long_input=long_input,
            context_window_size=context_window_size,
            model=model,
            api_key=api_key,
            query=query
        )
        chain_output = chain_of_agents.process()
        chain_time = time.time() - chain_start_time
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("RAG Output")
            st.write(rag_output)
            st.info(f"RAG processing time: {rag_time:.2f} seconds")
        
        with col2:
            st.header("Chain of Agents Output")
            st.write(chain_output)
            st.info(f"Chain of Agents processing time: {chain_time:.2f} seconds")
    else:
        st.error("Please provide both the long input text and the query.")