import requests
import json
import time
import base64
import json
import requests
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()

KEY = "<API_KEY>"

MODEL = "llama3-405b"
version = "v1"
URL = "fast-api.snova.ai"
NUM_SECONDS_TO_SLEEP = 5

eval_logger = logging.getLogger(__name__)
eval_logger.setLevel(logging.INFO)


def run_inference(messages):
    payload = {
        "messages": messages,
        "stop": ["<|eot_id|>"],
        "model": MODEL,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Basic {KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"https://{URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode("utf-8").split("data: ")[1])
                    if "choices" in json_response and len(json_response["choices"]) > 0:
                        content = (
                            json_response["choices"][0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            full_response += content
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue

        return full_response

    except requests.exceptions.HTTPError as e:
        if response.status_code in {401, 503, 504}:
            print(
                f"Attempt failed due to rate limit or gate timeout. Status code: {response.status_code}. Trying again in {NUM_SECONDS_TO_SLEEP} seconds..."
            )
            time.sleep(NUM_SECONDS_TO_SLEEP)
            return run_inference(messages)
        else:
            print(
                f"Request failed with status code: {response.status_code}. Error: {e}"
            )
            return ""


def main():
    user_prompt = "Hello, how are you?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_prompt},
    ]
    generated_text = run_inference(messages)
    print("\nFull response:", generated_text)


if __name__ == "__main__":
    main()
