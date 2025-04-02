from unittest import mock
from unittest.mock import Mock, patch

import pytest

with patch(
    "os.environ",
    {
        "INDEX_NAME": "foo",
        "LOG_LEVEL": "DEBUG",
        "AOSS_URL": "here",
        "AOSS_PORT": "42",
        "LOCAL": "1",
    },
):
    from aws_rag_quickstart.AgentLambda import main as agent_main
    from aws_rag_quickstart.AgentLambda import os_similarity_search, summarize_documents
    from aws_rag_quickstart.AWSAuth import get_aws_auth
    from aws_rag_quickstart.IngestionLambda import (
        augment_metadata,
        create_index_opensearch,
        insert_document_opensearch,
    )
    from aws_rag_quickstart.IngestionLambda import main as ingest_main
    from aws_rag_quickstart.IngestionLambda import process_file
    from aws_rag_quickstart.LLM import ChatLLM, Embeddings
    from aws_rag_quickstart.opensearch import (
        delete_doc,
        delete_documents_opensearch,
        get_all_indexed_files_opensearch,
        get_opensearch_connection,
        is_opensearch_connected,
        list_docs_by_id,
    )


# Mock response from LLM
class MockResponse:
    def __init__(self, content):
        self.content = content


def test_get_aws_auth():
    with mock.patch("boto3.Session"), mock.patch("aws_rag_quickstart.AWSAuth.AWS4Auth"):
        get_aws_auth()


@pytest.mark.parametrize("local", [1, 0])
def test_get_open_search_connection(local):
    with mock.patch("os.environ", {"LOCAL": local}), mock.patch(
        "aws_rag_quickstart.opensearch.OpenSearch"
    ), mock.patch("aws_rag_quickstart.opensearch.get_aws_auth"):
        get_opensearch_connection("foo", 999)


def test_delete_doc():
    with mock.patch("aws_rag_quickstart.opensearch.get_opensearch_connection"), mock.patch(
        "aws_rag_quickstart.opensearch.delete_documents_opensearch"
    ), patch(
        "os.environ",
        {
            "AOSS_URL": "bar",
            "AOSS_PORT": "bar",
            "INDEX_NAME": "foo",
        },
    ):
        delete_doc({"file_path": "foo"})


def test_get_all_indexed_files_opensearch():
    with mock.patch("aws_rag_quickstart.opensearch.get_opensearch_connection"):
        get_all_indexed_files_opensearch("foo")


@pytest.mark.parametrize("input_file", ["foo", "bar"])
def test_agent_main(input_file):
    with mock.patch("aws_rag_quickstart.AgentLambda.RunnablePassthrough"), mock.patch(
        "aws_rag_quickstart.LLM.ollama.pull"
    ), mock.patch("aws_rag_quickstart.AgentLambda.StrOutputParser"), mock.patch(
        "aws_rag_quickstart.AgentLambda.os_similarity_search",
    ), mock.patch(
        "aws_rag_quickstart.LLM.ChatOllama"
    ), mock.patch(
        "aws_rag_quickstart.LLM.ollama.embeddings",
        return_value=Mock(embed_query=Mock(return_value={})),
    ), mock.patch(
        "aws_rag_quickstart.IngestionLambda.get_opensearch_connection"
    ), mock.patch(
        "os.environ", {
            "BEDROCK_ENDPOINT": "https://foo",
            "LOCAL": "1",
            "CHAT_MODEL": "anthropic.claude-v2"
        }
    ), patch(
        "aws_rag_quickstart.AgentLambda.list_docs_by_id",
    ) as mock_os:
        mock_os.return_value = {"num_pages": 9}
        agent_main({"question": "bar", "unique_ids": [input_file]})


def test_llm_chat():
    with mock.patch("aws_rag_quickstart.LLM.ollama.pull"), mock.patch(
        "aws_rag_quickstart.LLM.ChatOllama"
    ), mock.patch(
        "os.environ", {
            "BEDROCK_ENDPOINT": "https://foo",
            "LOCAL": "1",
            "CHAT_MODEL": "anthropic.claude-v2"
        }
    ):
        ChatLLM()


@pytest.mark.parametrize("is_local", ["1", "0"])
def test_llm_is_local(is_local):
    with mock.patch("aws_rag_quickstart.LLM.ollama.pull"), mock.patch(
        "aws_rag_quickstart.LLM.ChatOllama"
    ), mock.patch("aws_rag_quickstart.LLM.BedrockEmbeddings"), mock.patch(
        "aws_rag_quickstart.LLM.ChatBedrock"
    ), mock.patch(
        "aws_rag_quickstart.LLM.ollama.embeddings"
    ), mock.patch(
        "os.environ",
        {
            "BEDROCK_ENDPOINT": "https://foo",
            "LOCAL": is_local,
            "CHAT_MODEL": "anthropic.claude-v2"
        },
    ):
        actual = Embeddings()
        actual.embed_query("foo")
        ChatLLM()


@pytest.mark.parametrize(
    "mock_client", [Mock(), Mock(ping=Mock(side_effect=ConnectionError()))]
)
def test_is_opensearch_connected(mock_client):
    is_opensearch_connected(mock_client)


@pytest.mark.parametrize("input_file, exists", [("foo", 1), ("bar", 0)])
def test_ingest_main(input_file, exists):
    mock_process_file = mock.Mock()
    # Set up the mock to return a dict with pii stats
    mock_process_file.return_value = 2  # Number of pages processed
    
    with mock.patch("aws_rag_quickstart.AWSAuth.AWS4Auth"), mock.patch(
        "aws_rag_quickstart.IngestionLambda.convert_from_bytes"
    ), mock.patch(
        "aws_rag_quickstart.IngestionLambda.process_file",
        mock_process_file
    ), mock.patch(
        "aws_rag_quickstart.IngestionLambda.create_index_opensearch",
    ), mock.patch(
        "aws_rag_quickstart.IngestionLambda.get_opensearch_connection",
        Mock(
            return_value=Mock(indices=Mock(exists=Mock(return_value=exists)))
        ),
    ), mock.patch(
        "aws_rag_quickstart.LLM.ChatOllama"
    ), patch(
        "os.environ", {
            "BEDROCK_ENDPOINT": "https://foo",
            "LOCAL": "1",
            "CHAT_MODEL": "anthropic.claude-v2"
        }
    ), mock.patch(
        "aws_rag_quickstart.LLM.ollama.embeddings"
    ), mock.patch(
        "aws_rag_quickstart.LLM.ollama.pull"
    ):
        result = ingest_main({"question": "bar", "file_path": input_file})
        
        # Assert that the result is in the expected format
        assert isinstance(result, dict)
        assert "num_pages_processed" in result
        assert "pii_stats" in result
        assert result["num_pages_processed"] == 2


def test_augment_metadata(monkeypatch):
    def mock_invoke(messages):
        return MockResponse(content={"new_key": "new_value"})

    llm = mock.Mock()
    monkeypatch.setattr(llm, "invoke", mock_invoke)

    sample_image_string = (
        "iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAABFZLRzAAAAAXNSR0IArs4c6QAA"
    )
    sample_metadata = {"title": "Sample PDF", "author": "John Doe"}

    result = augment_metadata(llm, sample_image_string, sample_metadata)
    assert "llm_generated" in result
    assert result["llm_generated"] == str({"new_key": "new_value"})
    assert result["title"] == "Sample PDF"
    assert result["author"] == "John Doe"


def test_insert_document_success(mocker):
    mock_client = mocker.MagicMock()
    mock_embeddings = mocker.MagicMock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    document = {"llm_generated": "This is a test document.", "title": "Test"}
    mock_response = {"result": "created"}
    mock_client.index.return_value = mock_response

    index_name = "test-index"
    result = insert_document_opensearch(
        client=mock_client,
        index_name=index_name,
        embeddings=mock_embeddings,
        document=document,
    )
    mock_embeddings.embed_query.assert_called_once_with(
        document["llm_generated"]
    )
    assert document["embedding"] == [0.1, 0.2, 0.3]
    mock_client.index.assert_called_once_with(
        index=index_name, body=document, refresh=True
    )
    assert result == mock_response


def test_no_llm_generated_field(mocker):
    mock_client = mocker.MagicMock()
    mock_embeddings = mocker.MagicMock()
    incomplete_document = {"title": "Test"}
    index_name = "test-index"

    with pytest.raises(KeyError):
        insert_document_opensearch(
            client=mock_client,
            index_name=index_name,
            embeddings=mock_embeddings,
            document=incomplete_document,
        )


def test_delete_documents_success(mocker):
    mock_client = mocker.MagicMock()
    index_name = "test-index"
    file_path = "document.pdf"
    mock_response = {"deleted": 5}
    mock_client.delete_by_query.return_value = mock_response
    result = delete_documents_opensearch(
        client=mock_client, index_name=index_name, file_path=file_path
    )
    expected_query_body = {"query": {"match": {"file_path": file_path}}}
    mock_client.delete_by_query.assert_called_once_with(
        index=index_name, body=expected_query_body
    )
    assert result == mock_response


def test_delete_documents_no_match(mocker):
    mock_client = mocker.MagicMock()
    index_name = "test-index"
    file_path = "non_existent_document.pdf"
    mock_response = {"deleted": 0}
    mock_client.delete_by_query.return_value = mock_response
    result = delete_documents_opensearch(
        client=mock_client, index_name=index_name, file_path=file_path
    )
    expected_query_body = {"query": {"match": {"file_path": file_path}}}
    mock_client.delete_by_query.assert_called_once_with(
        index=index_name, body=expected_query_body
    )
    assert result == mock_response


def test_os_similarity_search_success(mocker):
    input_query = {
        "context": {
            "question": "find documents",
            "unique_ids": ["document.pdf"],
        }
    }

    expected_query_dict = {
        "query": "find documents",
        "pdf_file": "document.pdf",
    }

    # Mock the embeddings
    mock_ollama_embeddings = mocker.patch("aws_rag_quickstart.LLM.ollama.embeddings")
    mock_ollama_embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

    mock_os_client = mock.Mock()
    mock_os_client.search.return_value = {"hits": {"total": 1, "hits": []}}
    mocker.patch(
        "aws_rag_quickstart.AgentLambda.get_opensearch_connection",
        return_value=mock_os_client,
    )

    mocker.patch.dict(
        "os.environ",
        {
            "BEDROCK_ENDPOINT": "mocked-endpoint",
            "CHAT_MODEL": "anthropic.claude-v2",
            "EMBED_MODEL": "anthropic.claude-v2",
            "LOCAL": "1"
        }
    )

    result = os_similarity_search.invoke(input_query)

    assert expected_query_dict["query"] == "find documents"
    assert expected_query_dict["pdf_file"] == "document.pdf"

    assert result == {"hits": {"total": 1, "hits": []}}
    mock_ollama_embeddings.assert_called_once_with(
        model="anthropic.claude-v2",
        prompt="find documents"
    )


def test_os_similarity_search_invalid_json(mocker):
    input_query = {
        "context": {
            "question": "find documents",
            "unique_ids": ["document.pdf"],
        }
    }
    with patch("os.environ", {"BEDROCK_ENDPOINT": "https://foo"}), patch(
        "boto3.session"
    ), mock.patch("aws_rag_quickstart.AgentLambda.get_opensearch_connection"), mock.patch(
        "aws_rag_quickstart.LLM.BedrockEmbeddings"
    ), mock.patch(
        "aws_rag_quickstart.LLM.ChatBedrock"
    ):
        result = os_similarity_search.invoke(input_query)
    assert result


def test_get_all_indexed_files_success(mocker):
    index_name = "test-index"

    expected_query_body = {
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

    mock_response = {
        "aggregations": {
            "ids": {
                "buckets": [
                    {"key": "file1.pdf", "doc_count": 10},
                    {"key": "file2.pdf", "doc_count": 8},
                ]
            }
        }
    }

    mock_os_client = mock.Mock()
    mocker.patch(
        "aws_rag_quickstart.opensearch.get_opensearch_connection", return_value=mock_os_client
    )
    mock_os_client.search.return_value = mock_response
    result = get_all_indexed_files_opensearch(index_name)
    mock_os_client.search.assert_called_once_with(
        index=index_name, body=expected_query_body
    )
    assert result == mock_response["aggregations"]["ids"]["buckets"]


def test_get_all_indexed_files_malformed_response(mocker):
    index_name = "test-index"
    mock_response = {}
    mock_os_client = mock.Mock()
    mocker.patch(
        "aws_rag_quickstart.opensearch.get_opensearch_connection", return_value=mock_os_client
    )
    mock_os_client.search.return_value = mock_response
    with pytest.raises(AttributeError):
        get_all_indexed_files_opensearch(index_name)


def test_process_file_success(mocker):
    input_dict = {
        "file_path": "test.pdf",
        "other_metadata": "example metadata",
        "pii_stats": {"pages_with_pii": 0, "total_pages": 0}
    }

    # Mock the PII detector
    mock_pii_detector = Mock()
    mock_pii_detector.filter_text.return_value = (True, "No PII detected", [])
    
    with patch("os.environ", {"S3_BUCKET": "foo"}), \
         patch("boto3.session"), \
         patch("aws_rag_quickstart.IngestionLambda.convert_from_bytes", Mock(return_value=[Mock(), Mock()])), \
         patch("aws_rag_quickstart.IngestionLambda.PIIDetector", Mock(return_value=mock_pii_detector)):
        
        result = process_file(input_dict, Mock(), Mock(), Mock(), Mock())

    assert result == 2  # We processed 2 pages
    # Verify PII stats were updated
    assert input_dict["pii_stats"]["total_pages"] == 2
    assert input_dict["pii_stats"]["pages_with_pii"] == 0


def test_create_index_opensearch_success(mocker):
    client = mocker.MagicMock()
    embeddings = mocker.MagicMock()
    index_name = "test-index"

    expected_index_body = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "unique_id": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,  # This is hardcoded in the implementation
                    "method": {
                        "name": "hnsw",
                        "space_type": "innerproduct",
                        "engine": "faiss",
                        "parameters": {
                            "ef_construction": 256,
                            "ef_search": 256,
                            "m": 32,
                        },
                    },
                },
            }
        },
    }

    mock_response = {"acknowledged": True, "index": index_name}
    client.indices.create.return_value = mock_response
    result = create_index_opensearch(client, embeddings, index_name)
    client.indices.create.assert_called_once_with(
        index=index_name, body=expected_index_body
    )
    assert result == mock_response


def test_list_docs_by_id():
    expected = {"num_pages": 1, "docs_list": ["bar"]}
    with patch(
        "aws_rag_quickstart.opensearch.get_opensearch_connection",
        Mock(
            return_value=Mock(
                search=Mock(
                    return_value={
                        "hits": {"hits": [{"_source": {"file_path": "bar"}}]}
                    }
                )
            )
        ),
    ):
        actual = list_docs_by_id("bar")
    assert actual == expected


def test_summarize_documents():
    with patch("aws_rag_quickstart.opensearch.get_opensearch_connection"), patch(
        "os.environ", {"BEDROCK_ENDPOINT": "https://foo"}
    ):
        summarize_documents({"unique_ids": ["foo"]})
