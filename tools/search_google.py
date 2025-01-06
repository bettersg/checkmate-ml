import requests
import json
import dotenv
import os

dotenv.load_dotenv()


async def search_google(q):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": os.environ.get("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = json.dumps({"q": q, "location": "Singapore", "gl": "sg"})
    response = requests.request("POST", url, headers=headers, data=payload)
    return {
        "result": response.json().get("organic"),
        "cost": 1 / 1000,  # https://serper.dev/
    }


search_function_definition = dict(
    name="search_google",
    description="Searches Google for the given query and returns organic search results using serper.dev. Call this when you need to retrieve information from Google search results.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "q": {
                "type": "STRING",
                "description": "The search query to use on Google.",
            },
        },
        "required": ["q"],
    },
)

search_google_tool = {
    "function": search_google,
    "definition": search_function_definition,
}


# if __name__ == "__main__":
#     import asyncio
#     q = "checkmate sg"
#     result = asyncio.run(search_google(q))
#     print(result)
