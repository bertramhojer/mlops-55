import typing
from contextlib import asynccontextmanager

import fastapi
import torch
import uvicorn
from transformers import AutoTokenizer, BertForSequenceClassification


class Registry:
    """Models for the API."""

    binary: BertForSequenceClassification
    multiclass: BertForSequenceClassification
    tokenizer: AutoTokenizer

    def get(self, mode: str) -> BertForSequenceClassification:
        """Get the client."""
        return getattr(self, mode)


registry = Registry()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Lifespan for the FastAPI app."""
    print("Starting up...")
    registry.binary = BertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=4)
    registry.multiclass = BertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=4)
    registry.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    registry.binary.eval()
    registry.multiclass.eval()
    yield
    del registry.binary
    del registry.multiclass
    del registry.tokenizer
    print("Shutting down...")


app = fastapi.FastAPI(
    title="Multiple Choice API",
    description="API for the Multiple Choice project",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(query: str, choices: list[str], mode: typing.Literal["binary", "multiclass"] = "multiclass"):
    """Predict endpoint.

    Args:
        query: The question or prompt
        choices: List of possible answers
        mode: Mode to use from registry

    Returns:
        Dictionary containing probabilities for each choice
    """
    # Combine query with each choice
    model = registry.get(mode)

    if mode == "binary":
        inputs = [f"{query} [SEP] {choice}" for choice in choices]
    else:
        inputs = [f"{query} [SEP] {choice}" for choice in choices]

    # TODO: Try to fix type hint warning posed by Pyright
    encoded_inputs = registry.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")  # type: ignore  # noqa: PGH003

    # Run inference
    with torch.inference_mode():
        outputs = model(**encoded_inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Convert to list and return predictions
    predictions = probabilities.flatten().tolist()

    return {"probabilities": predictions, "choices": choices}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
