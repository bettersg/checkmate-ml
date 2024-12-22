from langfuse.openai import openai
from langfuse import Langfuse

langfuse = Langfuse()

def redact(text):
    prompt = langfuse.get_prompt("message_redaction", label='prod')
    # Make use of prompts defined in Langfuse
    system_message = prompt.compile()
    prompt_messages = [{"role": "system", "content": system_message}]
    for example in prompt.config["examples"]:
        prompt_messages.append({"role": "user", "content": example["user"]})
        prompt_messages.append({"role": "assistant", "content": example["assistant"]})

    prompt_messages.append({"role": "user", "content": text})

    response = openai.chat.completions.create(
        model=prompt.config['model'],
        messages=prompt_messages,
        temperature=prompt.config['temperature'],
        seed=11,
        langfuse_prompt=prompt,
        user_id="pii_masking",
        name='redact',
        tags=['prod']
    )

    return response.choices[0].message.content, response.usage.total_tokens
