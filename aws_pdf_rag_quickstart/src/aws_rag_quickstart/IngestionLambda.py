import base64
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict

import boto3
import dotenv
from langchain.schema import HumanMessage
from opensearchpy import OpenSearch
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

from aws_rag_quickstart.constants import OS_HOST, OS_INDEX_NAME, OS_PORT
from aws_rag_quickstart.LLM import ChatLLM, Embeddings
from aws_rag_quickstart.opensearch import (
    create_index_opensearch,
    get_opensearch_connection,
    insert_document_opensearch,
)
from aws_rag_quickstart.pii_detector import PIIDetector

logging.basicConfig(level=os.environ["LOG_LEVEL"])
if int(os.getenv("LOCAL", "0")):
    dotenv.load_dotenv()


def augment_metadata(
    llm: ChatLLM, text_content: str, general_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Augment metadata using text content from PDF page
    
    :param llm: LLM instance for augmentation
    :param text_content: Text content from the PDF page
    :param general_metadata: Existing metadata
    :return: Augmented metadata
    """
    logging.info("Starting LLM metadata augmentation...")
    start_time = time.time()
    
    # Check if PII was detected in the text
    if general_metadata.get("pii_warning", ""):
        raise(Exception("\n\nWARNING: {pii_warning}. Do not include or refer to this specific PII in your metadata."))
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Add to the metadata of a PDF file based on this page content of the file. "
                "The metadata you generate will "
                "be indexed into an opensearch instance. Put all descriptive data into the values "
                f"section of the metadata. The existing metadata is {general_metadata}. "
                f"The content of the page is: \n\n{text_content}\n\n"
                "Only return a JSON object with the additional keys and values.",
            }
        ],
    )
    
    try:
        # Set a timeout for the LLM call - adjust as needed (30 seconds)
        response = llm.invoke([message])
        result = general_metadata.copy()
        result["llm_generated"] = str(response.content)
        
        elapsed_time = time.time() - start_time
        logging.info(f"LLM metadata augmentation completed in {elapsed_time:.2f} seconds")
        
        return result
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"LLM metadata augmentation failed after {elapsed_time:.2f} seconds: {str(e)}")
        
        # Return basic metadata without LLM enrichment if it fails
        result = general_metadata.copy()
        result["llm_generated"] = "Error: LLM processing timed out or failed"
        return result


def process_file(
    input_dict: Dict[str, Any],
    metadata_llm: ChatLLM,
    os_client: OpenSearch,
    os_index_name: str,
    os_embeddings: Any,
) -> int:
    """
    Process a file using the metadata. ONLY SUPPORTS PDF FILES FOR NOW
    We will examine each page of the pdf and build up metadata for each page.
    The metadata will be written to an opensearch instance

    :param input_dict: input_dict.
    :param metadata_llm: llm used to generate metadata.
    :param os_client: OpenSearchClient.
    :param os_index_name: OpenSearch index name .
    :param os_embeddings: embeddings function.
    :return: number of pages processed
    """
    file_path = input_dict.get("file_path")
    use_local_storage = input_dict.get("use_local_storage", False)
    
    logging.info(f"Processing file {file_path} with local_storage={use_local_storage}")
    start_time = time.time()
    
    # Initialize PII detector
    pii_detector = PIIDetector()
    
    # Initialize PII statistics tracking
    pii_stats = input_dict.get("pii_stats", {"pages_with_pii": 0, "total_pages": 0})
    
    try:
        # Handle local or S3 storage
        if use_local_storage:
            logging.info(f"Reading PDF from local storage: {file_path}")
            with open(file_path, 'rb') as f:
                pdf_file = f.read()
        else:
            logging.info(f"Reading PDF from S3: {file_path}")
            session = boto3.session.Session()
            s3 = session.client("s3")
            pdf_file = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=file_path)[
                "Body"
            ].read()
        
        logging.info(f"PDF file retrieved, size: {len(pdf_file)} bytes")
        
        # Use PyPDF2 to extract text instead of processing images
        pdf_reader = PdfReader(BytesIO(pdf_file))
        num_pages = len(pdf_reader.pages)
        logging.info(f"PDF has {num_pages} pages")

        i = 0
        for i in range(num_pages):
            page_start_time = time.time()
            page = pdf_reader.pages[i]
            text_content = page.extract_text()
            
            logging.info(f"Processing page {i+1}..")
            logging.info(f"Page {i+1} extracted text, size: {len(text_content)} bytes")
            
            # Update total pages stat
            pii_stats["total_pages"] += 1
            
            # Apply PII filtering to the text content
            is_safe, filtered_content, detected_entities = pii_detector.filter_text(text_content)
            
            if not is_safe:
                # Update pages with PII stat
                pii_stats["pages_with_pii"] += 1
                
                logging.warning(f"PII detected in page {i+1}. Detected entities: {detected_entities}")
                logging.info("Skipping indexing of page with PII")
                # Log the detected PII entities
                pii_warnings = [f"{entity['word']} ({entity['entity_group']})" for entity in detected_entities]
                logging.info(f"PII found: {', '.join(pii_warnings)}")
                
                # Skip indexing this page
                page_elapsed_time = time.time() - page_start_time
                logging.info(f"Page {i+1} processing stopped due to PII detection in {page_elapsed_time:.2f} seconds")
                continue
            
            metadata = augment_metadata(metadata_llm, text_content, input_dict)
            metadata["page_number"] = f"page_{i+1}"
            
            # Add PII detection flag to metadata (should always be false here since we skip pages with PII)
            metadata["contains_pii"] = False
            
            insert_document_opensearch(
                os_client, os_index_name, os_embeddings, metadata
            )
            
            page_elapsed_time = time.time() - page_start_time
            logging.info(f"Page {i+1} processed and indexed in {page_elapsed_time:.2f} seconds")
        
        # Update the PII stats in the input dict for reporting
        input_dict["pii_stats"] = pii_stats
            
        total_elapsed_time = time.time() - start_time
        logging.info(f"Indexed {num_pages} pages in {total_elapsed_time:.2f} seconds.")
        return num_pages
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Error processing file {file_path} after {elapsed_time:.2f} seconds: {str(e)}")
        raise


def main(event: Dict[str, Any], *args: Any, **kwargs: Any) -> int:
    logging.info(f"Starting ingestion process for event: {event}")
    start_time = time.time()
    
    metadata_llm = ChatLLM().llm
    os_embeddings = Embeddings()
    os_client = get_opensearch_connection(OS_HOST, OS_PORT)

    # create index if it does not exist
    if not os_client.indices.exists(index=OS_INDEX_NAME):
        create_index_opensearch(os_client, os_embeddings, OS_INDEX_NAME)

    # Add PII statistics tracking to event
    event["pii_stats"] = {"pages_with_pii": 0, "total_pages": 0}
    
    # process input pdf
    num_pages_processed = process_file(
        event, metadata_llm, os_client, OS_INDEX_NAME, os_embeddings
    )
    
    # Log PII statistics
    pii_stats = event.get("pii_stats", {})
    if pii_stats.get("total_pages", 0) > 0:
        pii_percentage = (pii_stats.get("pages_with_pii", 0) / pii_stats.get("total_pages", 0)) * 100
        logging.info(f"PII detection statistics: {pii_stats.get('pages_with_pii', 0)} of {pii_stats.get('total_pages', 0)} pages had PII ({pii_percentage:.1f}%)")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Ingestion completed: {num_pages_processed} pages processed in {elapsed_time:.2f} seconds")

    # Return both page count and PII stats
    return {
        "num_pages_processed": num_pages_processed,
        "pii_stats": pii_stats
    }
