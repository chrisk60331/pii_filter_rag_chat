# AWS RAG Chatbot with PII Protection

## Features

- Real-time PII detection using Hugging Face models
- Integrated AWS Bedrock and OpenAI model support
- PDF document ingestion and RAG-based queries
- Automatic rejection of messages containing PII

## How it works

1. Upload your PDF documents
2. Ask questions about your documents
3. All inputs are automatically screened for PII
4. If PII is detected, the message will be rejected
5. Otherwise, the system will use RAG to answer your question

## About PII Protection

This chatbot uses a small language model to detect potentially sensitive personal information. The model identifies common PII entity types such as:

- Person names
- Organizations
- Locations
- Miscellaneous identifying information

If detected, your message will not be processed, and you'll receive feedback about what triggered the detection. 