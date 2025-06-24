"""
Constants for the AWS RAG application.
"""

REGION_NAME = "us-east-1"
# model to generate metadata per image
TEMPERATURE = 0
# Bedrock models
BEDROCK_MODELS = [
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0", 
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-opus-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.deepseek.r1-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "us.meta.llama3-1-8b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-2-3b-instruct-v1:0",
    "us.meta.llama3-2-90b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "us.mistral.pixtral-large-2502-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-premier-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.writer.palmyra-x4-v1:0",
    "us.writer.palmyra-x5-v1:0",
]

# OpenAI models
OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "chatgpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-o3",
    "gpt-o3-pro",
    "gpt-o3-mini",
    "gpt-o1-mini"
    "gpt-o1-pro"
]

# Combined list of all available models
ALL_MODELS = BEDROCK_MODELS + OPENAI_MODELS 