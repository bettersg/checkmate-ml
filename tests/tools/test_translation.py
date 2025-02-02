import pytest
from tools.translation import translate_text, SupportedLanguage


@pytest.mark.asyncio
async def test_translate_text_valid_language():
    text = "Hello, world!"
    language = SupportedLanguage.CN.value
    translated_text = await translate_text(text, language)
    assert translated_text is not None
    assert isinstance(translated_text, str)


@pytest.mark.asyncio
async def test_translate_text_invalid_language():
    text = "Hello, world!"
    language = "invalid_language"
    with pytest.raises(ValueError) as excinfo:
        await translate_text(text, language)
    assert "Unsupported language" in str(excinfo.value)


@pytest.mark.asyncio
async def test_translate_text_default_language():
    text = "Hello, world!"
    translated_text = await translate_text(text)
    assert translated_text is not None
    assert isinstance(translated_text, str)
