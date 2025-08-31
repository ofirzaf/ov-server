"""
Pytest configuration file for OpenVINO GenAI Server tests.
"""

import pytest
from openai import OpenAI


def pytest_addoption(parser):
    """Add custom command line option for base URL."""
    parser.addoption(
        "--base-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL of the running server (default: http://localhost:8000)"
    )


@pytest.fixture(scope="session")
def base_url(request):
    """Fixture to get the base URL from command line or use default."""
    return request.config.getoption("--base-url")


@pytest.fixture(scope="session")
def client(base_url):
    """Fixture to create OpenAI client with the specified base URL."""
    return OpenAI(
        base_url=f"{base_url}/v1",
        api_key="test-key"  # Server doesn't validate API keys based on the code
    )
