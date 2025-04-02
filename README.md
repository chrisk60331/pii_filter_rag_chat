# AWS RAG Chatbot with PII Filtering


## Features

- Real-time PII detection using Hugging Face NER models
- Automatic rejection of messages containing PII
- Support for AWS Bedrock and OpenAI models
- PDF document ingestion with RAG-based query processing
- User-friendly Chainlit interface

## Setup and Installation

1. Clone this repository:


2. Set up AWS credentials and configuration:
   - Configure AWS CLI with `aws configure` or `saml2aws`
   - Create a `.env` file based on the sample provided

3. Run the Chainlit application using Docker Compose:
   ```
   docker-compose up -d --build
   ```

## Environment Variables

Create a `.env` file with the following variables:

```
AWS_REGION=us-west-2
LOCAL=0
MODEL_TEMP=0.7
CHAT_MODEL=anthropic.claude-3-sonnet-20240229-v1:0  # For AWS Bedrock
EMBED_MODEL=amazon.titan-embed-text-v1  # For AWS Bedrock
BEDROCK_ENDPOINT=https://bedrock-runtime.us-west-2.amazonaws.com
LOG_LEVEL=INFO
```

For OpenAI models, set the following:
```
OPENAI_API_KEY=your_api_key_here
```

## Running with Different Configurations

### Using AWS Bedrock (Default)
This runs the application using AWS Bedrock for AI services:

```bash
# Make sure .env.bedrock is properly configured
cp .env.bedrock .env
# Run the application
COMPOSE_PROFILES=default docker-compose up -d --build
```

### Using Ollama (Local LLM)
This runs the application with the Ollama local LLM:

```bash
# Make sure .env.ollama is properly configured 
cp .env.ollama .env
# Run with Ollama
COMPOSE_PROFILES=ollama docker-compose up -d --build
```

## PII Detection

The PII detection system uses a Hugging Face Named Entity Recognition (NER) model to identify potentially sensitive information in user messages. By default, it uses the `dslim/bert-base-NER` model, which can detect:

- Person names
- Organizations
- Locations
- Other identifying information

## Customization

### Using Different PII Detection Models

You can use different Hugging Face models by modifying the `model_name` parameter in the `PIIDetector` initialization:

```python
# Use a different model
pii_detector = PIIDetector(model_name="your-preferred-model")
```

### Adjusting Detection Sensitivity

The detection threshold can be adjusted in the `filter_text` method call:

```python
# More strict (lower threshold catches more potential PII)
is_safe, message_content, detected_entities = pii_detector.filter_text(user_input, threshold=0.6)

# More lenient (higher threshold allows more through)
is_safe, message_content, detected_entities = pii_detector.filter_text(user_input, threshold=0.9)
```

## Architecture

1. User inputs are sent to the Chainlit interface
2. PII detector checks for sensitive information
3. If PII is detected, the message is rejected with feedback
4. If safe, the message is processed by the RAG system
5. Responses are generated using either AWS Bedrock or OpenAI models

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
