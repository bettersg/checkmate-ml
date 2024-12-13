import requests
import os
from dotenv import load_dotenv
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
        "Content-Type": "application/json"
    }
    load_dotenv()
    payload = {
                  "url": url
              }
    print("prijsdfkgsjdfg", f"{hostname}/get-screenshot")
    response = requests.post(f"{hostname}/get-screenshot", json=payload, headers=headers)

    if response.status_code != 200:
        return {"success": False, "error": response.json()}

    return response.json()
    