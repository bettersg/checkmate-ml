from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def calculate_openai_api_cost(response, model='gpt-4o'):
    """
    Calculate the cost of an OpenAI API call based on the response usage data and model pricing.

    Parameters:
    - response (dict): The API response containing the 'usage' field.
    - model (str): The model used for the API call. Default is 'gpt-4o-mini'.

    Returns:
    - float: The total cost of the API call in USD.
    """

    # Define pricing per 1,000 tokens for different models
    pricing_per_1k_tokens = {
        'gpt-4o-mini': {'prompt': 0.15 / 1000, 'completion': 0.60 / 1000},
        'gpt-4o': {'prompt': 2.50 / 1000, 'completion': 10.00 / 1000},
        # Add other models and their pricing as needed
    }

    # Ensure the model is recognized
    if model not in pricing_per_1k_tokens:
        raise ValueError(f"Pricing for model '{model}' is not defined.")

    # Extract token usage from the response
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # Retrieve pricing for the specified model
    model_pricing = pricing_per_1k_tokens[model]
    prompt_cost_per_token = model_pricing['prompt']
    completion_cost_per_token = model_pricing['completion']

    # Calculate costs
    prompt_cost = prompt_tokens / 1000 * prompt_cost_per_token
    completion_cost = completion_tokens / 1000 * completion_cost_per_token
    total_cost = prompt_cost + completion_cost

    return total_cost


async def call_tool(tool_dict, tool_name, arguments, tool_call_id, cost_tracker):
  try:
    result = await tool_dict[tool_name](**arguments)
    if "cost" in result:

      cost_tracker["total_cost"] += result["cost"] #calculate costs

      cost_tracker["cost_trace"].append({
          "tool_name": tool_name,
          "cost": result["cost"]
      })
    if tool_name == "get_website_screenshot":
      # print("printing results", result)
      url = arguments.get("url", "unknown URL")
      if not result["success"]:
        return {
            "role": "tool",
            "content": "error occured, screenshot could not be taken",
            "tool_call_id": tool_call_id
        }
      return [
          {
              "role": "tool",
              "content": "screenshot successfully taken",
              "tool_call_id": tool_call_id
          },
          {
              "role": "user",
              "content": [
                {"type": "text", "text": f"Here's the screenshot for {url}:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": result["result"],
                    }
                },
              ]
          },
      ]
    return {
        "role": "tool",
        "content": json.dumps(result["result"]),
        "tool_call_id": tool_call_id
    }
  except Exception as exc:
    return {
        "role": "tool",
        "content": f"Tool {tool_name} generated an exception: {exc}",
        "tool_call_id": tool_call_id
    }
