# tools/rmse_scanner.py

import os
import requests
import time
from langfuse.decorators import observe


@observe()
async def check_malicious_url(url):
    hostname = os.environ.get("RMSE_HOSTNAME")

    # Authenticate with the hostname as the audience
    headers = {
        "x-api-key": os.environ.get("RMSE_API_KEY"),
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    response = requests.post(
        f"{hostname}/evaluate",
        json={"url": url, "source": "checkmate"},
        headers=headers,
    )

    if response.status_code != 200:
        return {"success": False, "error": response.content}
    else:
        results = response.json()
        if results.get("success", False):
            overall_result = results.get("overall_result", {})
            if overall_result:
                return {
                    "success": True,
                    "result": {
                        "classification": overall_result["classification"],
                        "score": overall_result["score"],
                    },
                }
            else:
                request_id = results.get("request_id")
                if not request_id:
                    return {
                        "success": False,
                        "error": results.get("message", "Request ID missing"),
                    }

                while not overall_result:
                    time.sleep(1)
                    evaluation_response = requests.get(
                        f"{hostname}/url/{request_id}/evaluation", headers=headers
                    )
                    if evaluation_response.status_code != 200:
                        return {"success": False, "error": evaluation_response.content}
                    evaluation_result = evaluation_response.json()
                    if evaluation_result:
                        overall_result = evaluation_result.get("overall_result", {})
                        if overall_result:
                            return {
                                "success": True,
                                "result": {
                                    "classification": overall_result["classification"],
                                    "score": overall_result["score"],
                                },
                            }
                        else:
                            return {"success": False, "error": "Overall result missing"}
        else:
            return {
                "success": False,
                "error": results.get("message", "An error occurred"),
            }


check_malicious_url_definition = dict(
    name="check_malicious_url",
    description="Runs a check on the provided URL to determine if it is malicious.\
      Returns either 'MALICIOUS', 'SUSPICIOUS' or 'BENIGN', as well as as maliciousness\
        score from 0-1. Note, while a malicious rating should be trusted, a benign rating \
            doesn't imply the absence of malicious behaviour, as there might be false negatives.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "url": {
                "type": "STRING",
                "description": "The URL of the website to check whether it is malicious.",
            },
        },
        "required": ["url"],
    },
)

check_malicious_url_tool = {
    "function": check_malicious_url,
    "definition": check_malicious_url_definition,
}
