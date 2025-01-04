# tests/tools/test_review_report.py

import pytest
from tools import submit_report_for_review
from tests.utils import print_dict


@pytest.mark.asyncio
async def test_review_report_tool(capfd):
    report = "This is a test report."
    sources = ["http://example.com"]
    result = await submit_report_for_review(report, sources, False, False, False)
    # print the prettified json result
    print_dict(result)
    assert "result" in result
    assert "feedback" in result["result"]
    assert "passedReview" in result["result"]
