# Running Tests

To run the tests, you need to execute the `pytest` command from the root directory of the project.

## Running All Tests

To run all tests, use the following command:

```sh
pytest
```

## Running Individual Tests

To run an individual test file, specify the path to the test file. For example, to run the tests in `test_review_report.py`, use:

```sh
pytest tests/tools/test_review_report.py
```

You can also run a specific test function within a test file by appending `::test_function_name` to the file path. For example:

```sh
pytest tests/tools/test_review_report.py::test_review_report_tool
```

## Additional Options

To see print statements during test execution, use the `-s` option:

```sh
pytest -s
```
