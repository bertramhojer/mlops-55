from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from project.api import app, registry


@pytest.fixture(scope="session")
def mock_registry():
    """Mock the registry for testing."""
    # Create mock models
    registry.model = Mock()
    registry.tokenizer = Mock()

    # Configure mock outputs
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    registry.model.return_value = mock_output
    registry.tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    return registry


@pytest.fixture
def client(mock_registry):
    """Create a test client with mocked models."""
    with TestClient(app) as client:
        yield client


def test_predict_basic(client):
    """Test prediction function."""
    payload = {"query": "What color is the sky?", "choices": ["blue", "green", "red", "yellow"]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200  # noqa: S101
    data = response.json()
    assert "predictions" in data  # Updated from "probabilities" to "predictions" # noqa: S101
    assert len(data["predictions"]) == len(  # noqa: S101
        payload["choices"]
    )  # Updated from "probabilities" to "predictions" # noqa: S101
    assert data["choices"] == payload["choices"]  # noqa: S101
