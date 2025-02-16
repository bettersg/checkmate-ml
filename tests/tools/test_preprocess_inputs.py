import pytest
from unittest.mock import patch, AsyncMock
from tools import get_screenshots_from_text, preprocess_inputs


@pytest.mark.asyncio
async def test_get_screenshots_from_text_no_urls():
    text = "This is a text without any URLs"
    result = await get_screenshots_from_text(text)
    assert result == []


@pytest.mark.asyncio
async def test_get_screenshots_from_text_single_url():
    with patch(
        "tools.preprocess_inputs.get_website_screenshot", new_callable=AsyncMock
    ) as mock_screenshot:
        mock_screenshot.return_value = {
            "success": True,
            "result": "https://example.com/screenshot1.png",
        }

        text = "Check out https://example.com"
        result = await get_screenshots_from_text(text)

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com"
        assert result[0]["image_url"] == "https://example.com/screenshot1.png"

        mock_screenshot.assert_called_once_with("https://example.com")


@pytest.mark.asyncio
async def test_get_screenshots_from_text_multiple_urls():
    with patch(
        "tools.preprocess_inputs.get_website_screenshot", new_callable=AsyncMock
    ) as mock_screenshot:
        mock_screenshot.side_effect = [
            {"success": True, "result": "https://example.com/screenshot1.png"},
            {"success": True, "result": "https://example.com/screenshot2.png"},
        ]

        text = "Check these: https://example1.com and https://example2.com"
        result = await get_screenshots_from_text(text)

        assert len(result) == 2
        assert result[0]["url"] == "https://example1.com"
        assert result[0]["image_url"] == "https://example.com/screenshot1.png"
        assert result[1]["url"] == "https://example2.com"
        assert result[1]["image_url"] == "https://example.com/screenshot2.png"

        assert mock_screenshot.call_count == 2


@pytest.mark.asyncio
async def test_get_screenshots_from_text_failed_screenshot():
    with patch(
        "tools.preprocess_inputs.get_website_screenshot", new_callable=AsyncMock
    ) as mock_screenshot:
        mock_screenshot.return_value = {
            "success": False,
            "error": "Failed to get screenshot",
        }

        text = "Check out https://example.com"
        result = await get_screenshots_from_text(text)

        assert len(result) == 0
        mock_screenshot.assert_called_once_with("https://example.com")


@pytest.mark.asyncio
async def test_multiple_screenshots():
    text = """
        Check these links:
        https://example.com
        https://google.com
        https://github.com
        """
    result = await get_screenshots_from_text(text)

    assert len(result) > 0
    for screenshot in result:
        assert "url" in screenshot
        assert "image_url" in screenshot
        assert isinstance(screenshot["url"], str)
        assert isinstance(screenshot["image_url"], str)
        assert screenshot["url"] in text


@pytest.mark.asyncio
async def test_preprocess_inputs():
    text = """
        Check these links:
        https://example.com
        https://google.com
        https://github.com
        """
    result = await preprocess_inputs(None, None, text)

    # We don't know exactly how many will succeed, but we should get some results
    assert len(result) > 0

    # Verify structure of results
    assert "result" in result
    assert "screenshots" in result

    result_json = result["result"]

    assert "is_access_blocked" in result_json
    assert "is_video" in result_json
    assert "reasoning" in result_json
    assert "intent" in result_json
