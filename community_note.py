import asyncio
from typing import Union
import json
from website_screenshot import get_website_screenshot
from tool_search_google import search_google
from utils import calculate_openai_api_cost, call_tool
import os
from openai import OpenAI
from langfuse.openai import openai
from langfuse import Langfuse
from dotenv import load_dotenv
from chinese_community_note import translate_to_chinese

# Load environment variables from .env file
load_dotenv()

# Langfuse setup
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

langfuse = Langfuse()

# Tool dictionary
tool_dict = {
    "get_website_screenshot": get_website_screenshot,
    "search_google": search_google,
}
MODEL = "gpt-4o"


# client = OpenAI(
#     api_key=os.environ.get('OPENAI_API_KEY'),  # This is the default and can be omitted
# )



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

async def summary_note(session_id, messages, cost_tracker):
    """
    Generates a summary sentence with emoji for the final note.

    Args:
        messages (list): The conversation history.

    Returns:
        tuple: Sanity check result and summary sentence.
    """
    summary_prompt = langfuse.get_prompt("summary_community_note", label="production")
    prompt = f"{summary_prompt.prompt}\n\n{messages}"
    response = openai.chat.completions.create(
        model=summary_prompt.config['model'],
        messages=[{"role": "system", "content": prompt}],
        session_id=session_id
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

async def generate_community_note(session_id, data_type: str = "text", text: Union[str, None] = None, image_url: Union[str, None] = None, caption: Union[str, None] = None):
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
  system_prompt = langfuse.get_prompt("community_note", label="production")
  messages.append({"role": "system", "content": system_prompt.prompt})
  # print("reached til here", messages)


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
      response = openai.chat.completions.create(
        model=system_prompt.config['model'],
        messages=messages,
        tools=tools,
        tool_choice="required",
        session_id=session_id
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
            final_messages, final_note = await summary_note(session_id, messages, cost_tracker)
            chinese_note = translate_to_chinese(final_note)
            # arguments["final_note"] = final_note
            # if "note" in arguments:
            #     arguments["initial_note"] = arguments.pop("note")
            # else:
            #     arguments["initial_note"] = "Error"
            # arguments["sources"] = "\n".join(arguments.get("sources", []))
            # arguments["trace"] = final_messages
            # arguments["cost"] = cost_tracker["total_cost"]
            # arguments["cost_trace"] = cost_tracker["cost_trace"]
            # duration = time.time() - start_time  # Calculate duration
            # arguments["time_taken"] = duration
            final_arguments = {
                "en": final_note,
                "cn": chinese_note,
                "sources": arguments.get("sources", [])
            }
            return final_arguments
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
    # print(f"Error: {e}")
    # for message in messages:
    #   print(message)
    #   print("\n\n")
    return {
        "error": str(e),
        "cost": cost_tracker["total_cost"],
        "cost_trace": cost_tracker["cost_trace"],
        "trace": messages,
        "time_taken": duration
    }


# if __name__ == "__main__":
#     import asyncio
#     text = "WP is so much better than PAP"
#     result = asyncio.run(generate_community_note(text=text))
#     # prettify the result
#     for key, value in result.items():
#         print(f"{key}: {value}")
#     # print(result)
