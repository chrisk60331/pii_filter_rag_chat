import logging
import os
from typing import Any, Dict, List, Union

import boto3
import dotenv
from botocore.config import Config
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from aws_rag_quickstart.LLM import ChatLLM, Embeddings
from aws_rag_quickstart.opensearch import (
    get_opensearch_connection,
    query_opensearch_with_score,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
if int(os.getenv("LOCAL", "0")):
    dotenv.load_dotenv()

client_config = Config(max_pool_connections=50)


@tool
def os_similarity_search(context: Dict[str, Any]) -> Any:
    """
    Perform a similarity search on OpenSearch.

    Args:
        context

    Returns:
        dict: The results of the search query.

    """
    unique_ids, question = context["unique_ids"], context["question"]
    
    # Build filter query for the unique IDs
    should_queries = [{"term": {"unique_id": uid}} for uid in unique_ids]
    # Get OpenSearch connection
    os_client = get_opensearch_connection(os.getenv("AOSS_HOST"), os.getenv("AOSS_PORT"))
    
    # Use the query_opensearch_with_score function
    results = query_opensearch_with_score(
        client=os_client,
        index_name=os.getenv("INDEX_NAME"),
        query_text=question,
        k=100,
    )
    
    # Format the response similar to the original function
    response = {
        "hits": {
            "hits": [
                {
                    "_source": doc,
                    "_score": doc.pop("score", 0)
                } for doc in results
            ]
        }
    }
    
    return response


def summarize_documents(
    event: Dict[str, Union[str, List[str]]], *args: Any, **kwargs: Any
) -> str:
    question = (
        "Describe each of these webpages from the website. What is happening "
        "on each page?"
    )
    return main(
        {
            "unique_ids": event.get("unique_ids"),
            "question": question,
        }
    )


def process_query(
    query: str, metadata_list: list[Dict[str, Any]], llm: ChatLLM
) -> str:
    """
    Process the query using the metadata and LLM
    """
    logging.info(f"Processing query: {query}")
    logging.info(f"Found {len(metadata_list)} matching documents")
    
    # Build prompt with context from metadata
    context = ""
    for i, metadata in enumerate(metadata_list):
        context += f"\nDocument {i+1}:\n"
        context += f"Source: {metadata.get('file_path', 'Unknown')}\n"
        context += f"Page: {metadata.get('page_number', 'Unknown')}\n"
        context += f"Content: {metadata.get('llm_generated', '')}\n"
    
    # Prompt for the LLM
    prompt = f"""
    You are a helpful assistant that answers questions based on the provided documents.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    Please provide a comprehensive answer based only on the information in the provided documents.
    If the answer cannot be determined from the context, politely say so.
    """
    
    from langchain.schema import HumanMessage
    
    message = HumanMessage(content=prompt)
    response = llm.llm.invoke([message])
    
    return response.content


def main(event: Dict[str, Any], *args: Any, **kwargs: Any) -> str:
    """
    Main entry point for the lambda function
    """
    # Get model ID from event or use default
    model_id = event.get("model_id", os.getenv("CHAT_MODEL"))
    
    # Initialize the chat model with selected model
    llm = ChatLLM(model_id=model_id)
    
    # Get query and document IDs from the event
    query = event.get("question", "")
    os_client = get_opensearch_connection()
    search_results = query_opensearch_with_score(
        client=os_client,
        index_name=os.getenv("INDEX_NAME"),
        query_text=query,
    )
    
    # Process query with search results
    response = process_query(query, search_results, llm)
    
    return response
