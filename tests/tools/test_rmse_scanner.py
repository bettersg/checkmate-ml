# tests/tools/test_rmse_scanner.py

import pytest
from tools import check_malicious_url
from tests.utils import print_dict


@pytest.mark.asyncio
async def test_check_url_scam_tool():
    url = "https://bbc.com"
    result = await check_malicious_url(url)
    print_dict(result)
    assert "success" in result
    assert "result" in result or "error" in result
