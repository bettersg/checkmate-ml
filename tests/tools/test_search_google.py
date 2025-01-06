# tests/tools/test_search_google.py

import pytest
from tools import search_google


@pytest.mark.asyncio
async def test_search_google():
    query = "checkmate sg"
    result = await search_google(query)
    assert "result" in result
    assert "cost" in result
    assert result["cost"] == 1 / 1000
