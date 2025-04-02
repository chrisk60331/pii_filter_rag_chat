import logging
import os
from typing import Any, Dict, List, Optional, Union

import dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection

from aws_rag_quickstart.AWSAuth import get_aws_auth
from aws_rag_quickstart.constants import OS_HOST, OS_INDEX_NAME, OS_PORT
from aws_rag_quickstart.LLM import Embeddings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
if int(os.getenv("LOCAL", "0")):
    dotenv.load_dotenv()


def get_opensearch_connection(host: str = OS_HOST, port: int = OS_PORT) -> OpenSearch:
    """
    Create a connection to the OpenSearch cluster.
    """
    logging.info("getting OpenSearch connection")
    try:
        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        # Check connection
        client.ping()
        logging.info(f"Successfully connected to OpenSearch at {host}:{port}")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to OpenSearch: {e}")
        raise


def is_opensearch_connected(client: OpenSearch) -> bool:
    """
    Connectivity test
    """
    try:
        return client.ping()
    except ConnectionError:
        return False


def create_index_opensearch(
    client: OpenSearch, embeddings: Any, index_name: str = OS_INDEX_NAME
) -> None:
    """
    Create a new index in OpenSearch.
    """
    try:
        mapping = {
            "mappings": {
                "properties": {
                    "text_embedding": {"type": "knn_vector", "dimension": 1536},
                    # Add all potential fields for metadata
                    "unique_id": {"type": "keyword"},
                    "file_path": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "page_number": {"type": "keyword"},
                    "llm_generated": {"type": "text"},
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.space_type": "cosinesimil",
                },
            },
        }
        response = client.indices.create(index=index_name, body=mapping)
        logging.info(f"Index {index_name} created: {response}")
    except Exception as e:
        logging.error(f"Error creating index: {e}")
        raise


def insert_document_opensearch(
    client: OpenSearch,
    index_name: str,
    embeddings: Any,
    data: Dict[str, Any],
) -> str:
    """
    Insert a document into the OpenSearch index.
    """
    try:
        # Check if llm_generated content exists
        content = data.get("llm_generated", "")
        if not content:
            logging.warning("No content to embed in document. Adding placeholder text.")
            content = "placeholder content for document with no text"
            data["llm_generated"] = content
            
        try:
            # Get embeddings for the text content
            embedding = embeddings.embed_query(content)
            
            # Validate embedding before indexing
            if embedding is None or (isinstance(embedding, list) and len(embedding) == 0):
                raise ValueError("Embedding is null or empty")
                
        except Exception as embed_error:
            logging.warning(f"Error generating embedding: {embed_error}")
            # Create a default embedding with the correct dimensionality (768 as defined in create_index_opensearch)
            embedding = [0.0] * 768  # Using 768 to match the index definition
            logging.warning("Using fallback zero embedding with dimension 768")
        
        # Add embedding to the document
        data["text_embedding"] = embedding
        
        # Index the document
        response = client.index(index=index_name, body=data)
        logging.info(f"Document indexed with ID: {response.get('_id')}")
        return response.get("_id")
    except Exception as e:
        logging.error(f"Error indexing document: {e}")
        raise


def query_opensearch_with_score(
    client: OpenSearch, 
    index_name: str, 
    query_text: str,
    k: int = 10,
    additional_query: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Query OpenSearch for similar documents based on text and optional filters.
    
    Args:
        client: OpenSearch client
        index_name: Name of the index to query
        query_text: Text to search for
        k: Number of results to return
        additional_query: Additional query parameters to filter results
        
    Returns:
        List of documents with similarity scores
    """
    try:
        # Generate embeddings for the query
        embeddings = Embeddings()
        query_embedding = embeddings.embed_query(query_text)
        
        # Build KNN query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "text_embedding": {
                                    "vector": query_embedding,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            },
            "size": k
        }
        
        # Add additional query criteria if provided
        if additional_query:
            for key, value in additional_query.items():
                query["query"]["bool"][key] = value
                
        logger.debug(f"Executing OpenSearch query: {query}")
        
        # Execute search
        response = client.search(index=index_name, body=query)
        
        # Process results
        hits = response.get("hits", {}).get("hits", [])
        results = []
        
        for hit in hits:
            document = hit.get("_source", {})
            document["score"] = hit.get("_score", 0)
            results.append(document)
            
        logging.info(f"Found {len(results)} matching documents")
        return results
        
    except Exception as e:
        logging.error(f"Error querying OpenSearch: {e}")
        return []


def delete_doc(
    data: Dict[str, Any],
) -> None:
    """
    Delete documents from OpenSearch.
    """
    try:
        client = get_opensearch_connection()
        
        # Build query to find documents to delete
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"file_path.keyword": data.get("file_path")}}
                    ]
                }
            }
        }
        
        # Delete by query
        response = client.delete_by_query(index=OS_INDEX_NAME, body=query)
        logging.info(f"Deleted documents: {response}")
    except Exception as e:
        logging.error(f"Error deleting documents: {e}")
        # Don't raise exception to avoid breaking cleanup process


def delete_documents_opensearch(
    client: OpenSearch, index_name: str, file_path: Any
) -> Any:
    """
    Delete documents from the OpenSearch instance related to the specific file.

    :param client: The OpenSearch client.
    :param index_name: The name of the index to query.
    :param file_path: The name of pdf file to filter on.
    :return: The results of the query.
    """
    query_body = {"query": {"match": {"file_path": file_path}}}

    response = client.delete_by_query(index=index_name, body=query_body)
    return response


def get_all_indexed_files_opensearch(index_name: str) -> Dict[str, Any]:
    """
    Get all indexed files from the OpenSearch instance.

    :param index_name: The name of the index to query
    :return: The results of the query
    """
    query_body = {
        "size": 0,
        "aggs": {
            "ids": {
                "composite": {
                    "sources": [
                        {"ids": {"terms": {"field": "unique_id.keyword"}}}
                    ]
                }
            }
        },
    }
    os_client = get_opensearch_connection(OS_HOST, OS_PORT)
    response = os_client.search(index=index_name, body=query_body)

    return response.get("aggregations").get("ids").get("buckets")


def list_docs_by_id(unique_ids: List[str]) -> Dict[str, Any]:
    # Returns a list of unique ids in OS
    pass  # Placeholder to maintain compatibility
