import logging
import os
from typing import Any

import ollama
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_ollama import ChatOllama
import boto3

IS_LOCAL = bool(int(os.getenv("LOCAL", "0")))
TEMPERATURE = os.getenv("MODEL_TEMP", "0.7")
REGION_NAME = os.getenv("AWS_REGION", "us-west-2")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
# Default timeout in seconds
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARN"))


class LLM:
    @property
    def is_local_llm(self: Any) -> bool:
        return bool(int(os.getenv("LOCAL", "0")))


class ChatLLM(LLM):
    def __init__(self) -> None:
        self.chat_model = os.getenv("CHAT_MODEL")
        if self.is_local_llm:
            try:
                logging.info(f"Initializing Ollama with model {self.chat_model}")
                ollama.pull(self.chat_model)
                self.llm = ChatOllama(
                    model=self.chat_model,
                    temperature=TEMPERATURE,
                    num_predict=4096,
                    seed=42,
                    request_timeout=DEFAULT_TIMEOUT,  # Add timeout
                )
                logging.info(f"Ollama model {self.chat_model} initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing Ollama model: {str(e)}")
                raise
        else:
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


class Embeddings(LLM):
    def __init__(self) -> None:
        self.prompt = None
        self.embed_model = os.getenv("EMBED_MODEL")

    def embed_query(self, prompt: str) -> Any:
        self.prompt = prompt
        if self.is_local_llm:
            try:
                logging.info(f"Getting embeddings from Ollama with model {self.embed_model}")
                # Pull the model first to ensure it's available
                logging.info(f"Pulling Ollama embedding model {self.embed_model} if needed")
                ollama.pull(self.embed_model)
                response = ollama.embeddings(
                    model=self.embed_model, 
                    prompt=self.prompt
                )
                return response.get("embedding")
            except Exception as e:
                logging.error(f"Error getting embeddings from Ollama: {str(e)}")
                raise
        else:
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
