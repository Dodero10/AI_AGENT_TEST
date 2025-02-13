import os

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class RAG:
    def __init__(self):
        # Thiết lập API key cho OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set it in the .env file.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Use the cheapest embedding model
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-ada-002"
        )

    def create_vector_store(self, documents: list):
        # Chuyển danh sách văn bản thành embeddings
        embeddings = self.embeddings.embed_documents(documents)
        
        # Chuyển embeddings thành một numpy array
        embedding_matrix = np.array(embeddings).astype('float32')

        # Tạo và lưu trữ vector vào FAISS index
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # Tạo FAISS index (dùng phương pháp L2 distance)
        index.add(embedding_matrix)  # Thêm dữ liệu vào index
        self.index = index
        self.documents = documents  # Lưu lại văn bản cho việc truy xuất
        return index

    def get_similar_chunks(self, query: str, top_k: int = 3):
        # Chuyển query thành embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)

        # Tìm các văn bản gần nhất (nearest neighbors)
        distances, indices = self.index.search(query_vector, top_k)

        # Lấy các văn bản tương ứng với các index đã tìm được
        similar_chunks = [self.documents[i] for i in indices[0]]
        return similar_chunks

    def generate_answer(self, query: str, long_text: list) -> str:
        # Tạo vector store từ văn bản dài
        self.create_vector_store(long_text)

        # Tìm các chunk văn bản phù hợp
        similar_chunks = self.get_similar_chunks(query)

        # Kết hợp các chunk với query để tạo prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Information:\n{''.join(similar_chunks)}\n\nAnswer:"}
        ]

        # Gửi yêu cầu đến OpenAI GPT-4o-mini để trả lời câu hỏi
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )

        # Trả về câu trả lời từ OpenAI
        answer = response.choices[0].message.content.strip()
        return answer

# Sử dụng class RAG
if __name__ == "__main__":
    rag = RAG()

    # Ví dụ về văn bản dài
    documents = [
        "France is a country in Western Europe. Its capital and largest city is Paris.",
        "Paris is known for its fashion, art, and culture. It is also a major center of commerce and finance.",
        "The Eiffel Tower is one of the most famous landmarks in Paris, attracting millions of tourists every year."
    ]

    # Ví dụ câu hỏi
    query = "What is the capital of France?"

    # Gọi hàm generate_answer
    answer = rag.generate_answer(query, documents)
    print("Answer:", answer)
