from unittest.mock import MagicMock, patch

import pytest

from src.project.frontend import classify_prompt, get_backend_url


@patch("src.project.frontend.run_v2.ServicesClient")
def test_get_backend_url(mock_services_client):
    """Test get_backend_url function."""
    mock_service = MagicMock()
    mock_service.name.split.return_value = ["", "", "", "", "production-model"]
    mock_service.uri = "http://backend-url"
    mock_services_client().list_services.return_value = [mock_service]

    url = get_backend_url()
    if url != "http://backend-url":
        raise AssertionError(f"Expected 'http://backend-url', but got {url}")


@patch("src.project.frontend.get_backend_url")
@patch("src.project.frontend.requests.post")
def test_classify_prompt(mock_post, mock_get_backend_url):
    """Test classify_prompt function."""
    mock_get_backend_url.return_value = "http://backend-url"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_post.return_value = mock_response

    result = classify_prompt("test prompt")
    if result != {"result": "success"}:
        raise AssertionError(f'Expected \'{{"result": "success"}}\', but got {result}')

    mock_post.assert_called_once_with("http://backend-url/predict", files={"prompt:": "test prompt"}, timeout=5)


@patch("src.project.frontend.get_backend_url")
def test_classify_prompt_failure(mock_get_backend_url):
    """Test classify_prompt function when the request fails."""
    mock_get_backend_url.return_value = "http://backend-url"
    with patch("src.project.frontend.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="Failed to classify prompt: 500 Internal Server Error") as excinfo:
            classify_prompt("test prompt")
        if "Failed to classify prompt: 500 Internal Server Error" not in str(excinfo.value):
            msg = "Failed to classify prompt: 500 Internal Server Error not found in exception message"
            raise AssertionError(msg)
