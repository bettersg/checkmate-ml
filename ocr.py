from paddleocr import PaddleOCR
import openai
import re
import requests
from io import BytesIO
from PIL import Image
import json
import os

#OCR Initialization
ocr = PaddleOCR(use_angle_cls=True, lang="ch")
openai.api_key = os.environ["OPENAI_API_KEY"]
left_determining_ratio = 0.2 #determines what's considered messages that start on the left
same_height_tolerance_ratio = 0.3 #proportion of the max of current and previous line heights, that is the tolerance for determining whether we assume current line height to be same as previous line
same_left_alignment_tolerance_ratio = 0.01 #proportion of total image width, that is the tolerance for determining 2 lines are both left aligned
threshold = 5 #1/threshold is the proportion of messages from top that are ignored when looking for timestamp
prompts = json.load(open("files/ocr_prompts.json"))

def perform_ocr(img_url):
  """
  Performs OCR on the image at the given url, and returns the OCR result and the image size
  """
  response = requests.get(img_url)
  img = Image.open(BytesIO(response.content))
  img_size = img.size
  result = ocr.ocr(response.content, cls=True)
  return result[0], img_size

def group_messages(ocr_result, image_size):
  """
  Groups the OCR result into messages, and returns a list of dictionaries, each dictionary representing a message
  """
  img_width = image_size[0]
  img_height = image_size[1]

  def get_snippet_features(img_snippet):
    left_edge = (img_snippet[0][0][0] + img_snippet[0][3][0]) / 2
    right_edge = (img_snippet[0][1][0] + img_snippet[0][2][0]) / 2
    top_edge = (img_snippet[0][0][1] + img_snippet[0][1][1]) / 2
    bottom_edge = (img_snippet[0][2][1] + img_snippet[0][3][1]) / 2
    height = bottom_edge - top_edge
    return left_edge, right_edge, top_edge, bottom_edge, height

  def replace_time_format(s):
      pattern = '([01]?[0-9]|2[0-3]):[0-5][0-9]'
      return re.sub(pattern, "", s).replace("AM","").replace("PM","") #i notice sometimes it puts out AMV/PMV instead of AM/PM

  message_groups = []
  message_group = {}
  previous_left_edge = None
  previous_right_edge = None
  previous_top_edge = None
  previous_bottom_edge = None
  previous_height = None
  for index, snippet in enumerate(ocr_result):
    left_edge, right_edge, top_edge, bottom_edge, height = get_snippet_features(snippet)
    is_left = left_edge <= img_width * left_determining_ratio
    is_right = not is_left
    text = snippet[1][0]
    if index == 0:
      pass
    else:
      is_same_height_as_previous = abs(height - previous_height) <= max(height, previous_height)*same_height_tolerance_ratio
      is_same_left_alignment = abs(left_edge - previous_left_edge) <= img_width * same_left_alignment_tolerance_ratio
      is_timestamp = (len(text) - len(replace_time_format(text))) >= 2
      if is_timestamp:
        continue

      if not ((is_left and is_previous_left and is_same_height_as_previous and is_same_left_alignment) or (is_right and is_previous_right and is_same_height_as_previous)):
        message_groups.append(message_group)
        message_group = {}

    message_group["bottom_edge"] = max(bottom_edge, message_group.get("bottom_edge",0))
    message_group["right_edge"] = max(right_edge, message_group.get("right_edge",0))
    message_group["left_edge"] = min(left_edge, message_group.get("left_edge",right_edge))
    message_group["top_edge"] = min(top_edge, message_group.get("top_edge",bottom_edge))
    message_group["is_left"] = is_left

    if ("text" in message_group and previous_bottom_edge and ((top_edge - previous_bottom_edge) > height)):
      message_group["text"] = message_group.get("text","") + "\n\n"

    message_group["text"] = message_group.get("text","") + " "+ text
    previous_left_edge = left_edge
    previous_right_edge = right_edge
    previous_top_edge = top_edge
    previous_bottom_edge = bottom_edge
    previous_height = height
    is_previous_left = is_left
    is_previous_right = is_right
  if message_group:
    message_groups.append(message_group)
  return message_groups

def add_spaces(message_groups):
  """
  calls openai to add spaces to the messages, returns a list of dictionaries, each dictionary representing a message
  """
  condensed_message_groups = json.dumps([{"is_left": message_group["is_left"], "text": message_group["text"]} for message_group in message_groups], indent=2)
  relevant_prompts = prompts.get("add-spaces",{})
  system_message_template = relevant_prompts.get("system","")
  assert system_message_template
  user_message_example = relevant_prompts.get("user-example","")
  assert user_message_example
  ai_message_example = relevant_prompts.get("ai-example","")
  assert ai_message_example
  MODEL = "gpt-3.5-turbo"
  response = openai.ChatCompletion.create(
      model=MODEL,
      messages=[
          {"role": "system", "content": system_message_template},
          {"role": "user", "content": user_message_example},
          {"role": "assistant", "content": ai_message_example},
          {"role": "user", "content": condensed_message_groups},
      ],
      temperature=0,
  )
  try:
    return json.loads(response.choices[0].message.content)
  except Exception as e:
    print("Error:", e)
    return condensed_message_groups

def extract_meaningful_groups(fixed_message_groups):
  """
  calls openai to add spaces to the messages, returns a list of dictionaries, each dictionary representing a message
  """
  condensed_message_groups = json.dumps(fixed_message_groups, indent=2)
  relevant_prompts = prompts.get("extract-convo",{})
  system_message_template = relevant_prompts.get("system","")
  assert system_message_template
  user_message_example = relevant_prompts.get("user-example","")
  assert user_message_example
  ai_message_example = relevant_prompts.get("ai-example","")
  assert ai_message_example
  MODEL = "gpt-3.5-turbo"
  response = openai.ChatCompletion.create(
      model=MODEL,
      messages=[
          {"role": "system", "content": system_message_template},
          {"role": "user", "content": user_message_example},
          {"role": "assistant", "content": ai_message_example},
          {"role": "user", "content": condensed_message_groups},
      ],
      temperature=0,
  )
  try:
    return json.loads(response.choices[0].message.content)
  except Exception as e:
    print("Error:", e)
    return condensed_message_groups

def check_convo(messages, threshold):
  pattern = r'^([01]?[0-9]|2[0-3]):[0-5][0-9](?:\s?[apAP][mM])?$'
  message_length = len(messages)
  cutoff = max(1,int(message_length/threshold)) #only starts finding timestamps from below here
  for message in messages[cutoff:]:
    if re.match(pattern, message):
      return True
  return False

def get_impt_message(final_output):
  try:
    if len(final_output["text_messages"]) > 0:
      longest_text_item = max(final_output["text_messages"], key=lambda x: len(x["text"]))
      return longest_text_item["text"]
    else:
      return ""
  except:
    return ""

def end_to_end(img_url):
  ocr_result, img_size = perform_ocr(img_url)
  message_groups = group_messages(ocr_result, img_size)
  fixed_message_groups = add_spaces(message_groups)
  output = extract_meaningful_groups(fixed_message_groups)
  ocr_output_processed = [item[1][0] for item in ocr_result]
  is_convo = check_convo(ocr_output_processed, threshold)
  important_message = get_impt_message(output)
  return output, is_convo, important_message

