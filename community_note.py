import asyncio
from typing import Union
import json
from website_screenshot import get_website_screenshot
from tool_search_google import search_google
from utils import calculate_openai_api_cost, call_tool
import os
from openai import OpenAI

tool_dict = {}
tool_dict["get_website_screenshot"] = get_website_screenshot
tool_dict["search_google"] = search_google

MODEL = "gpt-4o"


client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),  # This is the default and can be omitted
)

tools = [
    {
        "type": "function",
        "strict": True,
        "function": {
            "name": "search_google",
            "description": "Searches Google for the given query and returns organic search results using serper.dev. Call this when you need to retrieve information from Google search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The search query to use on Google."
                    }
                },
                "required": ["q"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "strict": True,
        "function": {
            "name": "get_website_screenshot",
            "description": "Takes a screenshot of the url provided. Call this when you need look at the web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to take a screenshot of."
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "strict": True,
        "function": {
            "name": "submit_community_note",
            "description": "Submits a community note, which concludes the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "The content of the community note. This should be succinct and provide the user enough context to stay safe and informed."
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "uri",
                            "description": "A link from which you sourced your community note source for the note."
                        },
                        "description": "A list of links from which your community note is based."
                    }
                },
                "required": ["note", "sources"],
                "additionalProperties": False
            }
        }
    }
]

async def summary_note(messages, cost_tracker):
    """
    Generates a summary sentence with emoji for the final note.

    Args:
        messages (list): The conversation history.

    Returns:
        tuple: Sanity check result and summary sentence.
    """
    summary_prompt = (
"""Review the conversation history below. Based on the community note generated, generate a summary sentence that starts with an emoji that succinctly conveys the key message. Output the final community note, which starts with an emoji, then a summary sentence, then the rest of the succinct, clear note. (less than 50 words)
Guidelines for emoji at the start of summary sentence:
- Use 🚨 or ❌ for scams or highly harmful content.
- Use ⚠️ for content that warrants caution but is not outright harmful.
- Use ✅ for legitimate or safe content.
- If unsure, avoid adding an emoji.
Reminder: do not integrate the sources into the note."""
)

    prompt = f"{summary_prompt}\n\n{messages}"
    response = client.chat.completions.create(
        model=MODEL,  # Use your preferred model
        messages=[{"role": "system", "content": prompt}],
    )

    content = response.choices[0].message.to_dict()["content"]

    openai_cost = calculate_openai_api_cost(response, "gpt-4o")
    cost_tracker["total_cost"] += openai_cost
    cost_tracker["cost_trace"].append({
        "model": "final_openai_call",
        "cost": openai_cost
    })

    print("cost", openai_cost)

    print("Sanity Check and Summary Result:", content)

    messages.append(response.choices[0].message)

    return messages, content

import time

# Define the system prompt as a variable
system_prompt = (
"""You are an agent in Singapore that helps the user check information that they send in on WhatsApp, which could either be a text message, or an image.
Your task is to:
1. Strictly use the supplied tools to help you check the information.
2. Submit an X-style community note to conclude your task.
- For irrelevant messages such as 'hello,' 'thank you,' let the user know that there’s nothing to verify.
- For all other messages, evaluate the content using tools and provide a concise, straightforward summary of the issue or risk (50 words or less)."""
)

async def generate_community_note(data_type: str = "text", text: Union[str, None] = None, image_url: Union[str, None] = None, caption: Union[str, None] = None):
  """Generates a community note based on the provided data type (text or image).

  Args:
      data_type: The type of data provided, either "text" or "image".
      text: The text content of the community note (required if data_type is "text").
      image_url: The URL of the image (required if data_type is "image").
      caption: An optional caption for the image.

  Returns:
      A dictionary representing the community note.
  """
  start_time = time.time()  # Start the timer
  cost_tracker = {"total_cost": 0,
                   "cost_trace": []  # To store the cost details
                  }
  messages = []
  messages.append({"role": "system", "content": system_prompt})

  if data_type == "text":
    if text is None:
      raise ValueError("Text content is required when data_type is 'text'")
    content = f"User sent in: {text}"

  elif data_type == "image":
    if image_url is None:
      raise ValueError("Image URL is required when data_type is 'image'")
    content = [
      {
        "type": "text",
        "text": f"User sent in the following image with this caption: {caption}"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": image_url
        }
      }
    ]

  messages.append({"role": "user", "content": content})
  completed = False
  # Main loop to process messages and handle function calls
  try:
    while len(messages) < 20 and not completed:
      response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="required"
      )
      messages.append(response.choices[0].message)
      openai_cost = calculate_openai_api_cost(response, MODEL)
      cost_tracker["total_cost"] += openai_cost
      cost_tracker["cost_trace"].append({
          "model": "openai_api",
          "cost": openai_cost
      })
      # Check if the response contains tool calls
      tool_calls = response.choices[0].message.tool_calls
      if tool_calls:
        # Gather all tool call results asynchronously
        tool_call_promises = []
        for tool_call in tool_calls:
          tool_name = tool_call.function.name
          arguments = json.loads(tool_call.function.arguments)
          tool_call_id = tool_call.id
          if tool_name == "submit_community_note":
            final_messages, final_note = await summary_note(messages, cost_tracker)
            arguments["final_note"] = final_note
            if "note" in arguments:
                arguments["initial_note"] = arguments.pop("note")
            else:
                arguments["initial_note"] = "Error"
            arguments["sources"] = "\n".join(arguments.get("sources", []))
            arguments["trace"] = final_messages
            arguments["cost"] = cost_tracker["total_cost"]
            arguments["cost_trace"] = cost_tracker["cost_trace"]
            duration = time.time() - start_time  # Calculate duration
            arguments["time_taken"] = duration
            return arguments
          else:
            tool_call_promise = call_tool(tool_dict, tool_name, arguments, tool_call_id, cost_tracker)
            tool_call_promises.append(tool_call_promise)
        tool_results = await asyncio.gather(*tool_call_promises)
        flattened_results = [item for sublist in tool_results for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Separate user messages and tool messages
        user_messages = [msg for msg in flattened_results if msg.get("role") != "tool"]
        tool_messages = [msg for msg in flattened_results if msg.get("role") == "tool"]
        # Extend with tool messages first, then append user messages
        messages.extend(tool_messages)
        messages.extend(user_messages)

      else:
        # If no tool call is generated, prompt for more input
        messages.append({
          "role": "system",
          "content": "You should only be using the provided functions"
        })
    duration = time.time() - start_time  # Calculate duration
    return {
        "error": "Couldn't generate note within 20 turns",
        "cost": cost_tracker["total_cost"],
        "cost_trace": cost_tracker["cost_trace"],
        "time_taken": duration
    }
  except Exception as e:
    print(f"Error: {e}")
    for message in messages:
      print(message)
      print("\n\n")
    return {
        "error": str(e),
        "cost": cost_tracker["total_cost"],
        "cost_trace": cost_tracker["cost_trace"],
        "trace": messages,
        "time_taken": duration
    }


if __name__ == "__main__":
    import asyncio
    text = "WP is so much better than PAP"
    result = asyncio.run(generate_community_note(text=text))
    # prettify the result
    for key, value in result.items():
        print(f"{key}: {value}")
    # print(result)
