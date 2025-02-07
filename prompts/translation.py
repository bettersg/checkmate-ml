from langfuse import Langfuse


translation_system_prompt = """You are a professional translator specializing in English to {{language}} translations. Your task is to translate the user's text while ensuring:
1. The translation captures the meaning and context of the original text accurately.
2. The tone and style remain consistent with the original message.
3. Avoid direct transliteration where it might make the text awkward or unclear in {{language}}.
The output should only be the translated text, and should be fluent and grammatically correct.
"""


def compile_messages_array():
    prompt_messages = [{"role": "system", "content": translation_system_prompt}]
    prompt_messages.append(
        {
            "role": "user",
            "content": "{{text}}",
        }
    )
    return prompt_messages


config = {
    "model": "gpt-4o",
    "temperature": 0.0,
}

if __name__ == "__main__":
    langfuse = Langfuse()
    prompt_messages = compile_messages_array()
    langfuse.create_prompt(
        name="translation",
        type="chat",
        prompt=prompt_messages,
        labels=["production", "development", "uat"],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    print("Prompt created successfully.")
