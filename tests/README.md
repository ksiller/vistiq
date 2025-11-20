# Vistiq Test Suite

This directory contains unit tests for the vistiq package. The test structure mirrors the source code structure in `src/vistiq/`.

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vistiq --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test class
pytest tests/test_core.py::TestConfiguration

# Run specific test function
pytest tests/test_core.py::TestConfiguration::test_configuration_creation

# Run with verbose output
pytest -v

# Run with more detailed output
pytest -vv
```

### Using tox

```bash
# Run all test environments
tox

# Run specific environment
tox -e py312

# Run with specific pytest arguments
tox -- -v -k test_core
```

## Test Structure

- `conftest.py`: Shared pytest fixtures
- `test_core.py`: Tests for core configuration and processing classes
- `test_utils.py`: Tests for utility functions and classes
- `test_preprocess.py`: Tests for preprocessing operations
- `test_seg.py`: Tests for segmentation operations
- `test_analysis.py`: Tests for analysis operations (coincidence detection)
- `test_workflow.py`: Tests for workflow classes
- `test_workflow_builder.py`: Tests for workflow builder components
- `test_app.py`: Tests for CLI application functions

## Writing New Tests

When adding new functionality, please add corresponding tests:

1. Create a test file following the naming convention `test_<module_name>.py`
2. Organize tests into classes that mirror the structure of the code being tested
3. Use descriptive test function names starting with `test_`
4. Use fixtures from `conftest.py` for common test data
5. Aim for high code coverage while focusing on testing behavior, not implementation details

## Coverage

The test suite aims for high code coverage. Run coverage reports with:

```bash
pytest --cov=vistiq --cov-report=term-missing
```

This will show which lines are not covered by tests.

