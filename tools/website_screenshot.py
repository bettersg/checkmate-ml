# tools/website_screenshot.py

import requests
import os
from google.auth.transport.requests import Request
from google.oauth2.id_token import fetch_id_token


def get_identity_token(audience: str) -> str:
    """Fetch the identity token for Service B."""
    request = Request()
    id_token = fetch_id_token(request, audience)
    return id_token


async def get_website_screenshot(url):
    hostname = os.environ.get("SCREENSHOT_HOSTNAME")

    identity_token = get_identity_token(hostname)
    headers = {
        "Authorization": f"Bearer {identity_token}",
        "Content-Type": "application/json",
    }
    payload = {"url": url}
    response = requests.post(
        f"{hostname}/get-screenshot", json=payload, headers=headers
    )

    if response.status_code != 200:
        return {"success": False, "error": response.json()}

    return response.json()


screenshot_function_definition = dict(
    name="get_website_screenshot",
    description="Takes a screenshot of the url provided. Call this when you need to look at the web page.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "url": {
                "type": "STRING",
                "description": "The URL of the website to take a screenshot of.",
            },
        },
        "required": ["url"],
    },
)

get_screenshot_tool = {
    "function": get_website_screenshot,
    "definition": screenshot_function_definition,
}
