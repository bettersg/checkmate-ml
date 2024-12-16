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
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException
from utils import remove_user_links_from_sources

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

# final return item definitions
class CommunityNoteItem(BaseModel):
    en: str
    cn: str
    links: List[str]
    isControversial: bool = False
    isVideo: bool = False


# client = OpenAI(
#     api_key=os.environ.get('OPENAI_API_KEY'),  # This is the default and can be omitted
# )


# get tools from langfuse
search_google_tool_str = langfuse.get_prompt("search_google_tool", label="production").prompt
get_website_screenshot_tool_str = langfuse.get_prompt("get_website_screenshot_tool", label="production").prompt
submit_community_note_tool_str = langfuse.get_prompt("submit_community_note_tool", label="production").prompt

# convert to json
search_google_tool = json.loads(search_google_tool_str)
get_website_screenshot_tool = json.loads(get_website_screenshot_tool_str)
submit_community_note_tool = json.loads(submit_community_note_tool_str)

tools = [search_google_tool, get_website_screenshot_tool, submit_community_note_tool]
print(tools)


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

    openai_cost = calculate_openai_api_cost(response, summary_prompt.config['model'])
    cost_tracker["total_cost"] += openai_cost
    cost_tracker["cost_trace"].append({
        "type": "summarizer_openai_call",
        "model": summary_prompt.config['model'],
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
        temperature=0,
        tools=tools,
        tool_choice="required",
        session_id=session_id
      )
      messages.append(response.choices[0].message)
      openai_cost = calculate_openai_api_cost(response, system_prompt.config['model'])
      cost_tracker["total_cost"] += openai_cost
      cost_tracker["cost_trace"].append({
          "type": "main_openai_api",
          "model": system_prompt.config['model'],
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

            # summarise the note
            final_messages, final_note = await summary_note(session_id, messages, cost_tracker)

            # translate the note to chinese
            chinese_note, final_messages = translate_to_chinese(session_id, final_note, final_messages, cost_tracker)

            # remove user links from sources
            user_input = final_messages[1]["content"]
            final_sources = remove_user_links_from_sources(user_input, arguments.get("sources", []))
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
            final_note_items = CommunityNoteItem(en=final_note, 
                                                 cn=chinese_note, 
                                                 links=final_sources, 
                                                 isControversial=arguments.get("isControversial", False), 
                                                 isVideo=arguments.get("isVideo", False))
            return final_note_items
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
  
  except ValueError as e:
    raise HTTPException(
        status_code=400,
        detail={
            "error": "Bad Request",
            "message": str(e),
            "code": "INVALID_INPUT"
        }
    )
  except Exception as e:
    # print(f"Error: {e}")
    # for message in messages:
    #   print(message)
    #   print("\n\n")
    raise HTTPException(
        status_code=500,
        detail={
            "error": "Internal Server Error",
            "message": str(e),
            "code": "GENERATION_FAILED"
        }
    )


if __name__ == "__main__":
    import asyncio
    text = "https://go.gov.sg/hsg-enrol" 
    result = asyncio.run(generate_community_note("123test", text=text))
    # prettify the result
    print(result)
    # print(result)
