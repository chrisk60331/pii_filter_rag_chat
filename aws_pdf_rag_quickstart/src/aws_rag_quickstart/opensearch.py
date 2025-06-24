import logging
import os
from typing import Any, Dict, List, Optional, Union

import dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection

from aws_rag_quickstart.AWSAuth import get_aws_auth

from aws_rag_quickstart.LLM import Embeddings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
if int(os.getenv("LOCAL", "0")):
    dotenv.load_dotenv()


def get_opensearch_connection(host: str = os.getenv("AOSS_HOST"), port: int = os.getenv("AOSS_PORT")) -> OpenSearch:
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
    client: OpenSearch, embeddings: Any, index_name: str = os.getenv("INDEX_NAME")
) -> None:
    """
    Create a new index in OpenSearch.
    """
    try:
        # First try with k-NN vector support (if plugin is available)
        mapping_with_knn = {
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
        
        try:
            response = client.indices.create(index=index_name, body=mapping_with_knn)
            logging.info(f"Index {index_name} created with k-NN support: {response}")
            return
        except Exception as knn_error:
            logging.warning(f"Failed to create index with k-NN support: {knn_error}")
            logging.info("Falling back to dense_vector type...")
        
        # Fallback: Create index without k-NN plugin (use dense_vector)
        mapping_fallback = {
            "mappings": {
                "properties": {
                    "text_embedding": {"type": "dense_vector", "dims": 1536},
                    # Add all potential fields for metadata
                    "unique_id": {"type": "keyword"},
                    "file_path": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "page_number": {"type": "keyword"},
                    "llm_generated": {"type": "text"},
                }
            },
        }
        
        response = client.indices.create(index=index_name, body=mapping_fallback)
        logging.info(f"Index {index_name} created with dense_vector fallback: {response}")
        
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
            # Create a default embedding with the correct dimensionality (1536 as defined in create_index_opensearch)
            embedding = [0.0] * 1536  # Using 1536 to match the index definition
            logging.warning("Using fallback zero embedding with dimension 1536")
        
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
        
        # First try k-NN query (if supported)
        knn_query = {
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
        
        # Fallback: Use script_score query for dense_vector
        fallback_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": k
        }
        
        # Add additional query criteria if provided (for k-NN query)
        if additional_query:
            for key, value in additional_query.items():
                knn_query["query"]["bool"][key] = value
                
        logger.debug(f"Executing OpenSearch query: {knn_query}")
        
        # Execute search - try k-NN first, fall back to script_score
        try:
            response = client.search(index=index_name, body=knn_query)
        except Exception as knn_error:
            logger.warning(f"k-NN query failed: {knn_error}, falling back to script_score")
            logger.debug(f"Executing fallback query: {fallback_query}")
            response = client.search(index=index_name, body=fallback_query)
        
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
