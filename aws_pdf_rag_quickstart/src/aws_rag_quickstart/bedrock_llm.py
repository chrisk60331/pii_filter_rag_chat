import logging
import os
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set up logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logger = logging.getLogger(__name__)

# Default settings from environment variables
REGION_NAME = os.getenv("AWS_REGION", "us-west-2")
BEDROCK_ENDPOINT = os.getenv("BEDROCK_ENDPOINT", None)
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMP", "0.7"))

# Client configuration with reasonable defaults
client_config = Config(
    max_pool_connections=50,
    retries={"max_attempts": 3, "mode": "standard"},
    connect_timeout=5,
    read_timeout=DEFAULT_TIMEOUT,
)


class BedrockLLM:
    """
    Enhanced AWS Bedrock LLM client with additional capabilities.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        streaming: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        max_tokens: Optional[int] = None,
        region_name: str = REGION_NAME,
        endpoint_url: Optional[str] = BEDROCK_ENDPOINT,
    ):
        """
        Initialize the Bedrock LLM client.
        
        Args:
            model_id: The Bedrock model ID to use (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            temperature: Controls randomness in responses (0.0-1.0)
            streaming: Whether to stream responses
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            region_name: AWS region name
            endpoint_url: Custom Bedrock endpoint URL
        """
        self.model_id = model_id or os.getenv("CHAT_MODEL")
        if not self.model_id:
            raise ValueError("Model ID must be provided either in constructor or CHAT_MODEL env var")
            
        self.temperature = temperature
        self.streaming = streaming
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        
        # Set up model kwargs based on provider
        self.model_kwargs = {"temperature": self.temperature}
        if max_tokens:
            if "anthropic" in self.model_id.lower():
                self.model_kwargs["max_tokens"] = max_tokens
            elif "amazon" in self.model_id.lower():
                self.model_kwargs["maxTokenCount"] = max_tokens
            else:
                # Default fallback for other models
                self.model_kwargs["max_tokens"] = max_tokens
        
        # Initialize the LangChain Bedrock chat model
        self._init_chat_model()
    
    def _init_chat_model(self):
        """Initialize the underlying LangChain Bedrock chat model."""
        logger.info(f"Initializing Bedrock LLM with model {self.model_id}")
        try:
            self.llm = ChatBedrock(
                model_id=self.model_id,
                region_name=self.region_name,
                model_kwargs=self.model_kwargs,
                endpoint_url=self.endpoint_url,
                streaming=self.streaming,
                # timeout=self.timeout,
            )
            logger.info(f"Successfully initialized Bedrock LLM with model {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock LLM: {str(e)}")
            raise
    
    def chat(
        self, 
        message: str, 
        system_prompt: Optional[str] = None, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Send a chat message to the LLM.
        
        Args:
            message: The user message to send
            system_prompt: Optional system prompt to guide the model
            chat_history: Optional chat history as list of dicts with "role" and "content" keys
            
        Returns:
            The model's response as a string
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                if msg["role"].lower() == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"].lower() == "assistant":
                    messages.append(HumanMessage(content=msg["content"]))
        
        # Add the current message
        messages.append(HumanMessage(content=message))
        
        # Invoke the model
        response = self.llm.invoke(messages)
        return response.content
    
    def create_prompt_template(
        self,
        template: str,
        input_variables: List[str],
        include_chat_history: bool = False
    ) -> ChatPromptTemplate:
        """
        Create a reusable prompt template.
        
        Args:
            template: The prompt template string
            input_variables: List of input variable names in the template
            include_chat_history: Whether to include a chat history placeholder
            
        Returns:
            A ChatPromptTemplate object
        """
        messages = [
            ("system", template),
        ]
        
        if include_chat_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
            
        messages.append(("human", "{input}"))
        
        return ChatPromptTemplate.from_messages(
            messages=messages,
            input_variables=input_variables + ["input"]
        )
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available Bedrock models in the current region.
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Create a Bedrock client
            bedrock_client = boto3.client(
                service_name="bedrock",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                config=client_config,
            )
            
            # Get foundation models
            response = bedrock_client.list_foundation_models()
            return response.get("modelSummaries", [])
        except Exception as e:
            logger.error(f"Error listing Bedrock models: {str(e)}")
            return [] 