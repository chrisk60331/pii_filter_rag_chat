"""
Chainlit application with integrated PII filtering for AWS RAG-based chatbot.
Supports both Bedrock and OpenAI models with real-time PII detection.
"""

import logging
import os
from typing import Dict, List
import uuid

import chainlit as cl
from chainlit.input_widget import Select

from aws_rag_quickstart.AgentLambda import main as agent_handler
from aws_rag_quickstart.IngestionLambda import main as ingest_handler
from aws_rag_quickstart.pii_detector import PIIDetector
from aws_rag_quickstart.bedrock_llm import BedrockLLM
from aws_rag_quickstart.constants import ALL_MODELS

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

# Initialize Bedrock client for model listing
bedrock_client = BedrockLLM() if not IS_LOCAL else None

async def get_available_models() -> List[Dict[str, str]]:
    """Get list of available Bedrock and OpenAI models"""
    if IS_LOCAL:
        return []
    
    try:
        return ALL_MODELS
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []


@cl.on_settings_update
async def on_settings_update(settings):
    selected_model = settings.get("model")
    cl.user_session.set("selected_model", selected_model)
    logger.info(f"Updated model selection: {selected_model}")


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    logger.info("Starting new chat session")
    
    # Set initial user documents list
    cl.user_session.set("document_ids", [])
    # Send a welcome message
    await cl.Message(
        content="""Welcome to the AWS RAG chatbot with PII protection. 
        You can ask questions about your documents after uploading them using the file attachment button.
        Note: All inputs are screened for PII and will be rejected if detected."""
    ).send()

    models = await get_available_models()
    initial_index = 12  # Default index
    # Create model selection dropdown
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Select Model",
                values=models,
                initial_value=models[initial_index],
            )
        ]
    ).send()
    cl.user_session.set("selected_model", settings["model"])


async def process_uploaded_files(files: List[cl.File]):
    """Process uploaded files and ingest them into the RAG system"""
    # Show loading message
    processing_msg = cl.Message(content=f"Processing {len(files)} uploaded files...")
    await processing_msg.send()
    
    # Get current unique ID for user or create new one
    user_id = cl.user_session.get("user_id", str(uuid.uuid4()))
    cl.user_session.set("user_id", user_id)
    
    # Process each file
    document_ids = cl.user_session.get("document_ids", [])
    local_storage_dir = "/tmp/pdf_storage"
    os.makedirs(local_storage_dir, exist_ok=True)
    
    for i, file in enumerate(files):
        processing_msg.content = f"Processing file {i+1}/{len(files)}: {file.name}..."
        await processing_msg.update()
        logger.info(f"Processing file: {file.name}, path: {file.path}")
        file_key = f"{user_id}_{file.name}"
        processing_msg.content = f"Storing {file.name} locally..."
        await processing_msg.update()
        logger.info(f"Stored {file.name} locally at {file.path}")
        event = {
            "unique_id": user_id,
            "file_path": file.path,
            "use_local_storage": True  # Flag to indicate local storage
        }

        try:
            processing_msg.content = f"Processing {file.name}... This might take a moment."
            await processing_msg.update()
            response = ingest_handler(event)
            document_ids.append(file_key)
            logger.info(f"Ingested document: {file_key}, processed {response} pages")
            processing_msg.content = f"Successfully ingested {file.name}."                
            await processing_msg.update()

        except Exception as e:
            logger.error(f"Error ingesting document {file_key}: {e}")
            processing_msg.content = f"Error processing file {file.name}: {str(e)}"
            await processing_msg.update()
    
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

    models = await get_available_models()
    logger.info(f"Available models: {models}")
    
    selected_model = cl.user_session.get("selected_model", os.getenv("CHAT_MODEL"))
    logger.info(f"Selected model: {selected_model}"),

    if message.elements:
        images = [file for file in message.elements]
        await process_uploaded_files(images)
    
    # Filter for PII in the user input
    is_safe, message_content, detected_entities = pii_detector.filter_text(user_input)
    logger.info(f"PII detection in user input: {detected_entities} {is_safe} { message_content }")
    if not is_safe:
        # PII detected, reject the message
        await cl.Message(
            content=message_content,
            author="PII Filter"
        ).send()
        return
    
    # No PII detected, process the message
    try:
        # Show thinking message
        thinking_msg = cl.Message(content="Thinking...")
        await thinking_msg.send()
        
        # Get selected model from session
        # Process the query through the RAG system
        event = {
            "unique_ids": document_ids,  # These are now S3 keys
            "question": user_input,
            "model_id": selected_model  # Pass selected model to handler
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


if __name__ == "__main__":
    # This code runs when the script is executed directly
    import chainlit as cl
    cl.run() 