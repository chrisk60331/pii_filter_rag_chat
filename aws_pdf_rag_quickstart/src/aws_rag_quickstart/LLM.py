import logging
import os
from typing import Any

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_openai import ChatOpenAI
import boto3

from aws_rag_quickstart.constants import BEDROCK_MODELS, OPENAI_MODELS

IS_LOCAL = bool(int(os.getenv("LOCAL", "0")))
TEMPERATURE = os.getenv("MODEL_TEMP", "0.7")
REGION_NAME = os.getenv("AWS_REGION", "us-west-2")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
# Default timeout in seconds
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARN"))


def is_bedrock_model(model_id: str) -> bool:
    """Check if the given model ID is a Bedrock model."""
    return model_id in BEDROCK_MODELS


def is_openai_model(model_id: str) -> bool:
    """Check if the given model ID is an OpenAI model."""
    return model_id in OPENAI_MODELS


class LLM:
    @property
    def is_local_llm(self: Any) -> bool:
        return bool(int(os.getenv("LOCAL", "0")))


class ChatLLM(LLM):
    def __init__(self, model_id: str = None) -> None:
        self.chat_model = model_id or os.getenv("CHAT_MODEL")
        if not self.chat_model:
            raise ValueError("Model ID must be provided either in constructor or CHAT_MODEL env var")
        
        # Initialize based on model type, not local/remote distinction
        if is_bedrock_model(self.chat_model):
            try:
                # Use the specified AWS profile
                session = boto3.Session(profile_name=AWS_PROFILE)
                logging.info(f"Using AWS profile: {AWS_PROFILE}")
                
                # Create a client with the timeout configuration
                bedrock_client = session.client(
                    'bedrock-runtime', 
                    region_name=REGION_NAME,
                    config=boto3.session.Config(
                        connect_timeout=DEFAULT_TIMEOUT,
                        read_timeout=DEFAULT_TIMEOUT
                    )
                )
                
                self.llm = ChatBedrock(
                    model_id=self.chat_model,
                    region_name=REGION_NAME,
                    model_kwargs={"temperature": float(TEMPERATURE)},
                    endpoint_url=os.environ.get("BEDROCK_ENDPOINT"),
                    client=bedrock_client,
                )
                logging.info(f"Bedrock model {self.chat_model} initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing Bedrock model: {str(e)}")
                raise
        elif is_openai_model(self.chat_model):
            try:
                self.llm = ChatOpenAI(
                    model=self.chat_model,
                    temperature=float(TEMPERATURE),
                    request_timeout=DEFAULT_TIMEOUT,
                )
                logging.info(f"OpenAI model {self.chat_model} initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing OpenAI model: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported model: {self.chat_model}. Supported models: {BEDROCK_MODELS + OPENAI_MODELS}")


class Embeddings(LLM):
    def __init__(self) -> None:
        self.prompt = None
        self.embed_model = os.getenv("EMBED_MODEL")

    def embed_query(self, prompt: str) -> Any:
        self.prompt = prompt
        # Always use Bedrock for embeddings (no more Ollama dependency)
        logging.info("Using Bedrock for embeddings with profile: %s", AWS_PROFILE)
        # Use the specified AWS profile
        session = boto3.Session(profile_name=AWS_PROFILE)
        
        # Create a client with the timeout configuration
        bedrock_client = session.client(
            'bedrock-runtime',
            region_name=REGION_NAME,
            config=boto3.session.Config(
                connect_timeout=DEFAULT_TIMEOUT,
                read_timeout=DEFAULT_TIMEOUT
            )
        )
        
        result = BedrockEmbeddings(
            region_name=REGION_NAME,
            endpoint_url=os.environ.get("BEDROCK_ENDPOINT"),
            client=bedrock_client,
        ).embed_query(self.prompt)
        logging.info("Embed length: %s", len(result))
        return result
