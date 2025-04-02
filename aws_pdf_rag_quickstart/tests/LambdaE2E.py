import json
import os
from json import JSONDecodeError

import requests


def lambda_handler(event, context):
    host = os.environ["ECS_HOST_IP"]
    delete = int(os.environ["DELETE"])  # boolean
    docs = []
    try:
        response = requests.post(
            "http://" + host + "/pdf_file",
            data=json.dumps({"event": {"unique_ids": ["sample.pdf"]}}),
        )

        docs = json.loads(response.content.decode()).get("docs_list")
    except JSONDecodeError:
        pass
    if not docs:
        response = requests.put(
            "http://" + host + "/pdf_file",
            data=json.dumps(
                {
                    "event": {
                        "unique_id": "sample.pdf",
                        "file_path": "sample.pdf",
                    }
                }
            ),
        )
        print(vars(response))
    response = requests.post(
        "http://" + host + "/chat",
        data=json.dumps(
            {
                "event": {
                    "unique_ids": ["sample.pdf"],
                    "question": "Describe whats happening in the docs",
                }
            }
        ),
    )
    print(vars(response))
    if delete:
        response = requests.delete(
            "http://" + host + "/pdf_file",
            data=json.dumps(
                {
                    "event": {
                        "unique_id": "sample.pdf",
                        "file_path": "sample.pdf",
                    }
                }
            ),
        )
        print(vars(response))
    return "ok"


if __name__ == "__main__":
    lambda_handler({}, {})
