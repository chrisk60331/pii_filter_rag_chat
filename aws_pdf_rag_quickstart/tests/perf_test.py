import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

limit = 6
host = "localhost"


def make_request():
    start = time.time()
    response = requests.get(f"http://{host}/summary?unique_id=JohnDeere")
    end = time.time()
    return response.status_code, end - start


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=limit) as executor:
        futures = [executor.submit(make_request) for _ in range(limit)]

        for future in as_completed(futures):
            status, text = future.result()
            print(f"Status: {status}, Response: {text}")
