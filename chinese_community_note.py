from deep_translator import GoogleTranslator

def translate_to_chinese(text):
    """
    Translates the given text from English to Simplified Chinese using deep-translator.

    Args:
        text (str): The English text to translate.

    Returns:
        str: The translated text in Simplified Chinese.
    """
    try:
        translated_text = GoogleTranslator(source='auto', target='zh-CN').translate(text)
        return translated_text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

# Example usage
# if __name__ == "__main__":
#     english_text = "✅ The website in the image appears to be legitimate. It aligns with the official Healthier SG website details, offering enrolment information for Singapore residents through official channels like HealthHub. However, always verify URLs and contact official sources when uncertain."
#     chinese_translation = translate_to_chinese(english_text)
#     print("Chinese Translation:", chinese_translation)