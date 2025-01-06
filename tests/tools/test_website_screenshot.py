# tests/tools/test_website_screenshot.py

import pytest
from tools import get_website_screenshot
from tests.utils import print_dict


@pytest.mark.asyncio
async def test_get_website_screenshot():

    url = "http://checkmate.sg"

    result = await get_website_screenshot(url)
    print_dict(result)
    assert "success" in result
    assert result["success"] is True
    assert "result" in result
