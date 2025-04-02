import json
import os
import time

import requests


def lambda_handler(event, context):
    create = int(os.environ["CREATE"])  # boolean
    delete = int(os.environ["DELETE"])  # boolean
    host = os.environ["ECS_HOST_IP"]
    if create:
        response = requests.put(
            "http://" + host + "/bulk",
            data=json.dumps(
                {
                    "event": {
                        "unique_id": "JohnDeere",
                        "file_paths": [
                            "JohnDeere/John Deere US _ Products & Services Information.pdf",
                            "JohnDeere/Compact, Ag, 4WD Tractors _ John Deere US.pdf",
                        ],
                    }
                }
            ),
        )
        assert (
            vars(response)["_content"]
            == b'{"message":"Processing in the background"}'
        ), vars(response)["_content"]
        time.sleep(90)
    response = requests.post(
        "http://" + host + "/pdf_file",
        data=json.dumps({"event": {"unique_ids": ["JohnDeere", "apple"]}}),
    )
    expected = {
        "num_pages": 9,
        "docs_list": [
            "JohnDeere/John Deere US _ Products & Services Information.pdf",
            "JohnDeere/Compact, Ag, 4WD Tractors _ John Deere US.pdf",
        ],
    }
    print(vars(response)["_content"])
    actual = json.loads(vars(response)["_content"].decode())

    assert sorted(actual["docs_list"]) == sorted(
        expected["docs_list"]
    ), f"{sorted(actual['docs_list'])} == {sorted(expected['docs_list'])}"
    time.sleep(10)
    response = requests.get(
        "http://" + host + "/summary",
        data=json.dumps(
            {
                "event": {
                    "unique_ids": [
                        "JohnDeere",
                        "apple/www.apple.com_2024-08-27-22-43-46.pdf",
                    ]
                }
            }
        ),
    )
    print(vars(response))
    if delete:
        response = requests.delete(
            "http://" + host + "/bulk",
            data=json.dumps(
                {
                    "event": {
                        "unique_id": "JohnDeere",
                        "file_paths": [
                            "JohnDeere/Compact, Ag, 4WD Tractors _ John Deere US.pdf",
                            "John Deere US _ Products & Services Information.pdf",
                        ],
                    }
                }
            ),
        )
        print(vars(response))
    return "ok"


if __name__ == "__main__":
    lambda_handler({}, {})
