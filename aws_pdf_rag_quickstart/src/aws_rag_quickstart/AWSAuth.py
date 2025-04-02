import boto3
from requests_aws4auth import AWS4Auth

from aws_rag_quickstart.constants import REGION_NAME


def get_aws_auth() -> AWS4Auth:
    service = "es"  # must set the service as 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION_NAME,
        service,
        session_token=credentials.token,
    )
    return awsauth
