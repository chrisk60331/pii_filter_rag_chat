"""
Chainlit application with integrated PII filtering for AWS RAG-based chatbot.
Supports both Bedrock and OpenAI models with real-time PII detection.
"""

import logging
import os
from typing import Dict, List
import uuid

import chainlit as cl
from chainlit.message import Message
from chainlit.types import AskFileResponse
import boto3

from aws_rag_quickstart.AgentLambda import main as agent_handler
from aws_rag_quickstart.IngestionLambda import main as ingest_handler
from aws_rag_quickstart.opensearch import delete_doc, list_docs_by_id
from aws_rag_quickstart.pii_detector import PIIDetector
from aws_rag_quickstart.LLM import ChatLLM, Embeddings

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize the PII detector
pii_detector = PIIDetector()

# Track uploaded document IDs
user_documents = {}

# Check if using local LLM or AWS Bedrock
IS_LOCAL = bool(int(os.getenv("LOCAL", "0")))
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    logger.info("Starting new chat session")
    
    # Set initial user documents list
    cl.user_session.set("document_ids", [])
    
    # Send a welcome message
    await cl.Message(
        content="""Welcome to the AWS RAG chatbot with PII protection. 
        You can ask questions about your documents after uploading them.
        Note: All inputs are screened for PII and will be rejected if detected."""
    ).send()
    
    # Setup file upload UI
    await setup_file_upload()


async def setup_file_upload():
    """Setup the file upload UI elements"""
    await cl.Message(content="Please upload PDF documents to begin.").send()
    files = await cl.AskFileMessage(
        content="Upload PDF files",
        accept=["application/pdf"],
        max_size_mb=10,
        max_files=5,
        timeout=180,
    ).send()
    
    if not files:
        await cl.Message(content="No files uploaded. Upload files to continue.").send()
        return await setup_file_upload()
    
    await process_uploaded_files(files)


async def process_uploaded_files(files: List[AskFileResponse]):
    """Process uploaded files and ingest them into the RAG system"""
    # Show loading message
    processing_msg = cl.Message(content=f"Processing {len(files)} uploaded files...")
    await processing_msg.send()
    
    # Get current unique ID for user or create new one
    user_id = cl.user_session.get("user_id", str(uuid.uuid4()))
    cl.user_session.set("user_id", user_id)
    
    # Process each file
    document_ids = cl.user_session.get("document_ids", [])
    
    # Initialize the LLM models for metadata generation and embeddings
    try:
        logger.info(f"Initializing LLM (local={IS_LOCAL}, profile={AWS_PROFILE})")
        metadata_llm = ChatLLM().llm
        embeddings = Embeddings()
        logger.info("LLM initialization successful")
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        await cl.Message(content=f"Error initializing LLM: {str(e)}").send()
        return
    
    # Check if S3 storage is available or use local storage
    use_local_storage = False
    local_storage_dir = "/tmp/pdf_storage"
    s3_bucket_name = os.getenv('S3_BUCKET', 'aoss-qa-dev-data-bucket')
    
    # Create S3 client and test connection
    try:
        # Use the same AWS profile as configured for Bedrock
        session = boto3.Session(profile_name=AWS_PROFILE)
        s3 = session.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT_URL'))
        # Try to list the bucket to check if it exists
        s3.head_bucket(Bucket=s3_bucket_name)
        logger.info(f"S3 bucket {s3_bucket_name} is accessible using profile {AWS_PROFILE}")
    except Exception as e:
        logger.warning(f"S3 bucket {s3_bucket_name} is not accessible: {str(e)}")
        logger.info("Falling back to local storage")
        use_local_storage = True
        # Create local storage directory if it doesn't exist
        os.makedirs(local_storage_dir, exist_ok=True)
    
    for i, file in enumerate(files):
        # Update processing message for each file
        processing_msg.content = f"Processing file {i+1}/{len(files)}: {file.name}..."
        await processing_msg.update()
        logger.info(f"Processing file: {file.name}, path: {file.path}")
        
        try:
            # Read file content
            with open(file.path, "rb") as source_file:
                file_content = source_file.read()
            
            # Create a unique file key
            file_key = f"{user_id}_{file.name}"
            
            if use_local_storage:
                # Store locally instead of S3
                processing_msg.content = f"Storing {file.name} locally..."
                await processing_msg.update()
                
                local_file_path = os.path.join(local_storage_dir, file_key)
                with open(local_file_path, 'wb') as local_file:
                    local_file.write(file_content)
                logger.info(f"Stored {file.name} locally at {local_file_path}")
                
                processing_msg.content = f"File stored. Ingesting {file.name} (this may take a minute)..."
                await processing_msg.update()
                
                # Create a modified event for local storage
                event = {
                    "unique_id": user_id,
                    "file_path": local_file_path,
                    "use_local_storage": True  # Flag to indicate local storage
                }
            else:
                # Upload to S3
                processing_msg.content = f"Uploading {file.name} to S3..."
                await processing_msg.update()
                
                try:
                    s3.put_object(
                        Bucket=s3_bucket_name,
                        Key=file_key,
                        Body=file_content
                    )
                    logger.info(f"Uploaded {file.name} to S3 bucket {s3_bucket_name}")
                    
                    processing_msg.content = f"File uploaded. Ingesting {file.name} (this may take a minute)..."
                    await processing_msg.update()
                    
                    # Create event for S3 storage
                    event = {
                        "unique_id": user_id,
                        "file_path": file_key,
                        "use_local_storage": False
                    }
                except Exception as e:
                    logger.error(f"Error uploading to S3: {e}")
                    await cl.Message(
                        content=f"Error uploading file {file.name} to S3: {str(e)}"
                    ).send()
                    continue  # Skip to next file
            
            # Process the file with the ingest handler
            try:
                # Show a processing message
                processing_msg.content = f"Processing {file.name}... This might take a moment."
                await processing_msg.update()
                
                # Prepare the event with necessary LLM information
                # ingest_handler will use our configured LLM from LLM.py
                try:
                    # Call the ingestion handler which will use the LLM configured through env vars
                    response = ingest_handler(event)
                    
                    document_ids.append(file_key)
                    
                    # Check if response is the new dictionary format
                    if isinstance(response, dict) and "pii_stats" in response:
                        num_pages = response.get("num_pages_processed", 0)
                        pii_stats = response.get("pii_stats", {})
                        
                        # Log the PII stats
                        pages_with_pii = pii_stats.get("pages_with_pii", 0)
                        total_pages = pii_stats.get("total_pages", 0)
                        
                        if total_pages > 0:
                            pii_percentage = (pages_with_pii / total_pages) * 100
                            logger.info(f"PII detection in {file.name}: {pages_with_pii} of {total_pages} pages had PII ({pii_percentage:.1f}%)")
                            
                            # Add PII warning to the processing message if PII was found
                            if pages_with_pii > 0:
                                document_ids = []
                                raise Exception(f"{file.name}: Error: PII was detected in {pages_with_pii} of {total_pages} pages - NOT indexed.")
                            else:
                                processing_msg.content = f"Successfully ingested {file.name}. No PII detected."
                        else:
                            processing_msg.content = f"Successfully ingested {file.name}."
                    else:
                        # Handle the old format (just a page count)
                        logger.info(f"Ingested document: {file_key}, processed {response} pages")
                        processing_msg.content = f"Successfully ingested {file.name}."
                    
                    await processing_msg.update()
                except Exception as e:
                    logger.error(f"Error in ingest_handler: {str(e)}")
                    raise
                
            except Exception as e:
                logger.error(f"Error ingesting document {file_key}: {e}")
                processing_msg.content = f"Error processing file {file.name}: {str(e)}"
                await processing_msg.update()
        except Exception as e:
            logger.error(f"Error reading file {file.name}: {e}")
            await cl.Message(
                content=f"Error reading file {file.name}: {str(e)}"
            ).send()
    
    # Update session with document IDs
    cl.user_session.set("document_ids", document_ids)
    
    # Final update to the processing message
    if document_ids:
        processing_msg.content = f"Successfully processed {len(files)} files. You can now ask questions!"
        await processing_msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages"""
    user_input = message.content
    document_ids = cl.user_session.get("document_ids", [])
    user_id = cl.user_session.get("user_id", "")
    
    # Check if user has uploaded documents
    if not document_ids:
        await cl.Message(
            content="Please upload PDF documents before asking questions."
        ).send()
        return await setup_file_upload()
    
    # Filter for PII in the user input
    is_safe, message_content, detected_entities = pii_detector.filter_text(user_input)
    
    if not is_safe:
        # PII detected, reject the message
        await cl.Message(
            content=message_content,
            author="PII Filter"
        ).send()
        await setup_file_upload()
        return
    
    # No PII detected, process the message
    try:
        # Show thinking message
        thinking_msg = cl.Message(content="Thinking...")
        await thinking_msg.send()
        
        # Process the query through the RAG system
        event = {
            "unique_ids": document_ids,  # These are now S3 keys
            "question": user_input
        }
        
        response = agent_handler(event)
        
        # Update the thinking message with the response
        thinking_msg.content = response
        await thinking_msg.update()
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await cl.Message(
            content="An error occurred while processing your query. Please try again."
        ).send()

    finally:
        await setup_file_upload()


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat session ends"""
    document_ids = cl.user_session.get("document_ids", [])
    user_id = cl.user_session.get("user_id", "")
    
    # Clean up documents from the RAG system if needed
    if document_ids and user_id:
        logger.info(f"Cleaning up {len(document_ids)} documents for user {user_id}")
        
        # Create S3 client for cleanup
        try:
            session = boto3.session.Session()
            s3 = session.client('s3', endpoint_url=os.getenv('AWS_ENDPOINT_URL'))
            s3_bucket_name = os.getenv('S3_BUCKET', 'aoss-qa-dev-data-bucket')
            s3_available = True
        except Exception as e:
            logger.warning(f"Unable to connect to S3: {str(e)}")
            s3_available = False
        
        local_storage_dir = "/tmp/pdf_storage"
        
        for doc_id in document_ids:
            try:
                # Try to delete from OpenSearch first
                try:
                    delete_doc({"file_path": doc_id})
                    logger.info(f"Deleted document from OpenSearch: {doc_id}")
                except Exception as e:
                    logger.error(f"Error deleting document from OpenSearch {doc_id}: {e}")
                
                # Check if it's a local file path
                if os.path.exists(doc_id):
                    # Delete local file
                    try:
                        os.remove(doc_id)
                        logger.info(f"Deleted local file: {doc_id}")
                    except Exception as e:
                        logger.error(f"Error deleting local file {doc_id}: {e}")
                elif s3_available:
                    # Delete from S3
                    try:
                        s3.delete_object(Bucket=s3_bucket_name, Key=doc_id)
                        logger.info(f"Deleted S3 object: {doc_id}")
                    except Exception as e:
                        logger.error(f"Error deleting S3 object {doc_id}: {e}")
                
                logger.info(f"Cleaned up document: {doc_id}")
            except Exception as e:
                logger.error(f"Error cleaning up document {doc_id}: {e}")
        
        # Also clean up any files in the local storage directory that match the user ID pattern
        try:
            if os.path.exists(local_storage_dir):
                for filename in os.listdir(local_storage_dir):
                    if filename.startswith(f"{user_id}_"):
                        try:
                            os.remove(os.path.join(local_storage_dir, filename))
                            logger.info(f"Cleaned up additional local file: {filename}")
                        except Exception as e:
                            logger.error(f"Error deleting additional local file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up local storage directory: {e}")


if __name__ == "__main__":
    # This code runs when the script is executed directly
    import chainlit as cl
    cl.run() 