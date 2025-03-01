# Agentic RAG System

This project implements an Agentic Retrieval-Augmented Generation (RAG) system that combines generative language models with AI agents to retrieve up-to-date information and generate accurate, context-aware responses.

## Features

- **Vector Store Retrieval**: Retrieves relevant information from a database of pre-indexed documents.
- **Web Search**: Fetches up-to-date information from the web when the required data is not available in the Vector Store.
- **Agentic Decision Making**: Dynamically decides which tool to use based on the query.
- **Batch Processing**: Supports batch processing of multiple queries.
- **Evaluation**: Integration with Athina for evaluation of RAG performance.

## Project Structure

```
basic_agentic/
├── .env                  # Environment variables
├── main.py               # Main entry point
├── pyproject.toml        # Project dependencies
├── README.md             # This file
└── src/
    └── agentic_rag/      # Main package
        ├── __init__.py   # Package initialization
        ├── agent.py      # Main agent class
        ├── config.py     # Configuration handling
        ├── document_loader.py  # Document loading utilities
        ├── language_model.py   # Language model integration
        ├── vector_store.py     # Vector store functionality
        └── web_search.py       # Web search functionality
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd basic_agentic
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ATHINA_API_KEY=your_athina_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GOOGLE_API_KEY=your_google_api_key
   GROK_API_KEY=your_grok_api_key
   ```

## Usage

### Basic Usage

```python
from src.agentic_rag import AgenticRAG

# Initialize the system
rag = AgenticRAG()
rag.initialize()

# Load a PDF file
rag.load_pdf("path/to/your/pdf/file.pdf")

# Query the system
result = rag.query("What is the total automotive revenue for Q3?")
print(f"Response: {result['output']}")
```

### Batch Processing

```python
queries = [
    "What milestones did the Shanghai factory achieve in Q3?",
    "Tesla stock market summary for 2024?"
]
batch_results = rag.batch_query(queries)
```

### Evaluation with Athina

```python
rag.connect_to_athina("your-dataset-id")
```

### Interactive Mode

Run the main script to enter interactive mode:

```
python main.py
```

## Dependencies

- langchain
- langchain_community
- langchain-google-genai
- langchain-huggingface
- pypdf
- faiss-gpu
- athina-client
- python-dotenv

## License

This project is licensed under the MIT License - see the LICENSE file for details.
