"""
Constants for the AWS RAG application.
"""

REGION_NAME = "us-east-1"
# model to generate metadata per image
TEMPERATURE = 0
# Bedrock models
BEDROCK_MODELS = [
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0", 
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

# OpenAI models
OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
]

# Combined list of all available models
ALL_MODELS = BEDROCK_MODELS + OPENAI_MODELS 