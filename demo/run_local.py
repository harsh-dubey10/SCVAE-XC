"""Backup for class: run the whole demo on this laptop (no Colab needed).

    pip install -r requirements.txt
    python run_local.py            # opens the Gradio page in the browser; prints the FastAPI docs URL
    python run_local.py --share    # also prints a public gradio.live link (needs internet)

The FastAPI backend uses the first free port from 8765 upward, so it never clashes with other servers.
"""

import argparse
import os
import socket
import threading
import time

import requests
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true")
args = parser.parse_args()


def free_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


port = free_port()
url = f"http://127.0.0.1:{port}"
server = uvicorn.Server(uvicorn.Config("api:app", host="127.0.0.1", port=port, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(120):
    try:
        if requests.get(url + "/health", timeout=1).json().get("status") == "ok":
            break
    except (requests.RequestException, ValueError):
        pass
    time.sleep(0.5)
else:
    raise SystemExit("the FastAPI backend did not start; see the messages above")
print(f"FastAPI backend: {url}/docs")

os.environ["API_URL"] = url
import app  # noqa: E402

app.build().launch(share=args.share, inbrowser=True)
