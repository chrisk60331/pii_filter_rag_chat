# Define custom function directory
ARG FUNCTION_DIR="/function"

FROM python:3.12.7-slim-bullseye as build-image
# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ${FUNCTION_DIR}/
# Create directory structure and copy source files properly
RUN mkdir -p ${FUNCTION_DIR}/src/aws_rag_quickstart
RUN touch ${FUNCTION_DIR}/src/aws_rag_quickstart/__init__.py
# Install dependencies using uv
RUN uv venv
RUN uv pip install -e .

# Install chainlit and required dependencies using pip for better PATH integration
RUN uv pip install --no-cache-dir chainlit>=0.7.0 transformers>=4.36.0 torch>=2.1.0


FROM build-image as local-stack-setup

RUN apt-get update
RUN apt-get install -y curl unzip less
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip && ./aws/install
RUN uv pip install awscli-local
ENV AWS_ENDPOINT_URL=http://local-stack:4566
COPY data ${FUNCTION_DIR}/data

# Create a startup script that ensures the bucket exists
RUN echo '#!/bin/bash \n\
set -e \n\
echo "Creating S3 bucket: aoss-qa-dev-data-bucket" \n\
awslocal s3api create-bucket --bucket aoss-qa-dev-data-bucket --region us-west-1 \
--create-bucket-configuration '\''{"LocationConstraint": "us-west-1"}'\'' || true \n\
echo "Verifying bucket exists..." \n\
awslocal s3api head-bucket --bucket aoss-qa-dev-data-bucket \n\
echo "Copying data files to S3..." \n\
for i in data/*; do \n\
  echo "Uploading $i to S3" \n\
  awslocal s3 cp $i s3://aoss-qa-dev-data-bucket/ || echo "Failed to copy $i" \n\
done \n\
echo "S3 setup complete!" \n\
# Keep container running \n\
tail -f /dev/null' > /start.sh

RUN chmod +x /start.sh
CMD ["/start.sh"]

FROM build-image as bedrock

# Include global arg in this stage of the build
ARG FUNCTION_DIR
RUN pip install awslambdaric

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
# Pass the name of the function handler as an argument to the runtime
CMD [ "src.aws_rag_quickstart.AgentLambda.main" ]

FROM build-image as opensearch

# Include global arg in this stage of the build
ARG FUNCTION_DIR
RUN pip install awslambdaric

RUN apt-get update -y
RUN apt-get install -y poppler-utils
# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
# Pass the name of the function handler as an argument to the runtime
CMD [ "src.aws_rag_quickstart.IngestionLambda.main" ]

FROM build-image as ecsopensearch

# Include global arg in this stage of the build
ARG FUNCTION_DIR
RUN apt-get update -y
RUN apt-get install -y curl poppler-utils
# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Copy Chainlit app and configuration
COPY chainlit_app.py ${FUNCTION_DIR}/
COPY chainlit.md ${FUNCTION_DIR}/

# Install local package
WORKDIR ${FUNCTION_DIR}
COPY src/aws_rag_quickstart ${FUNCTION_DIR}/src/aws_rag_quickstart/


# Set PYTHONPATH to include the function directory
ENV PYTHONPATH=${FUNCTION_DIR}:${PYTHONPATH}

# Create a directory for temporary files
RUN mkdir -p /tmp
RUN chmod 777 /tmp

# Expose port for Chainlit
EXPOSE 80

# Pass the name of the function handler as an argument to the runtime
CMD [ "uv", "run", "python", "-m", "chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "80" ]

FROM build-image as ollamallm
RUN apt-get update -y
RUN apt-get install --no-install-recommends -y curl
RUN curl -L https://ollama.com/download/ollama-linux-arm64.tgz -o ollama-linux-arm64.tgz
RUN tar -C /usr -xzf ollama-linux-arm64.tgz
RUN pip install --no-cache-dir "huggingface_hub[cli]==0.26.2"
CMD ["ollama", "serve"]
