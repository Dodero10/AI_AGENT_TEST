import os

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

class RAG:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set it in the .env file.")
        
        self.client = OpenAI(api_key=api_key)
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-ada-002"
        )

    def create_vector_store(self, documents: list):
        embeddings = self.embeddings.embed_documents(documents)
        
        embedding_matrix = np.array(embeddings).astype('float32')

        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        self.index = index
        self.documents = documents

    def split_text_into_chunks(self, text: str, chunk_size: int) -> list:
        tokens = text.split()
        chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        return chunks

    def get_similar_chunks(self, query: str, top_k: int = 3):
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)

        distances, indices = self.index.search(query_vector, top_k)

        similar_chunks = [self.documents[i] for i in indices[0]]
        return similar_chunks

    def generate_answer(self, query: str, long_text: str, context_window_size: int) -> str:
        chunks = self.split_text_into_chunks(long_text, context_window_size)

        self.create_vector_store(chunks)

        similar_chunks = self.get_similar_chunks(query)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Information:\n{''.join(similar_chunks)}\n\nAnswer:"}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()
        return answer

if __name__ == "__main__":
    rag = RAG()

    documents = [
        "France is a country in Western Europe. Its capital and largest city is Paris.",
        "Paris is known for its fashion, art, and culture. It is also a major center of commerce and finance.",
        "The Eiffel Tower is one of the most famous landmarks in Paris, attracting millions of tourists every year."
    ]

    query = "What is the capital of France?"

    answer = rag.generate_answer(query, documents, 100)
    print("Answer:", answer)
