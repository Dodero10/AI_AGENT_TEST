"""Example script for using the retrieval agent with proper configuration."""

import asyncio
import os
from dotenv import load_dotenv

from retrieval_graph import graph, index_graph
from retrieval_graph.config_helper import get_default_config

# Load environment variables
load_dotenv()

async def index_document(text):
    """Index a document with the proper configuration."""
    config = get_default_config()
    print(f"Indexing document with user_id: {config['configurable']['user_id']}")
    result = await index_graph.ainvoke({"docs": text}, config)
    return result

async def query_agent(query_text):
    """Query the agent with the proper configuration."""
    config = get_default_config()
    print(f"Querying agent with user_id: {config['configurable']['user_id']}")
    result = await graph.ainvoke(
        {"messages": [("user", query_text)]},
        config,
    )
    return result

async def main():
    """Run the example."""
    # 1. First, let's index a simple document
    document_text = """
    Artificial Intelligence (AI) is transforming industries worldwide.
    Machine Learning, a subset of AI, enables systems to learn from data.
    Natural Language Processing allows computers to understand human language.
    Computer Vision helps machines interpret and make decisions based on visual data.
    """
    
    # Index the document
    index_result = await index_document(document_text)
    print("Document indexed:", index_result)
    
    # 2. Now, let's query the agent
    query = "What are the main components of AI mentioned in the document?"
    response = await query_agent(query)
    
    # 3. Print the response
    print("\nQuery:", query)
    print("\nResponse:")
    for message in response["messages"]:
        if message.type == "ai":
            print(message.content)

if __name__ == "__main__":
    asyncio.run(main()) 