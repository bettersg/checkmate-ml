
import requests
import json
import dotenv
import os

dotenv.load_dotenv()

async def search_google(q):
  url = "https://google.serper.dev/search"
  headers = {
    'X-API-KEY': os.environ.get('SERPER_API_KEY'),
    'Content-Type': 'application/json'
  }
  payload = json.dumps({
    "q": q,
    "location": "Singapore",
    "gl": "sg"
  })
  response = requests.request("POST", url, headers=headers, data=payload)
  return {
      "result": response.json().get('organic'),
      "cost": 1/1000 #https://serper.dev/
  }

# if __name__ == "__main__":
#     import asyncio
#     q = "checkmate sg"
#     result = asyncio.run(search_google(q))
#     print(result)
