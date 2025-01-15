import typing
from contextlib import asynccontextmanager

import fastapi
import torch
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer, BertForSequenceClassification


class Registry:
    """Models for the API."""

    binary: BertForSequenceClassification
    multiclass: BertForSequenceClassification
    tokenizer: AutoTokenizer

    def get(self, mode: str) -> BertForSequenceClassification:
        """Get the client."""
        return getattr(self, mode)


class PredictionRequest(BaseModel):
    """Request for prediction."""

    query: str
    choices: list[str]
    mode: typing.Literal["binary", "multiclass"] = "multiclass"


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
def predict(request: PredictionRequest):
    """Predict endpoint."""
    model = registry.get(request.mode)

    # Combine query with each choice
    if request.mode == "binary":
        inputs = [f"{request.query} [SEP] {choice}" for choice in request.choices]
    else:
        inputs = [f"{request.query} [SEP] {choice}" for choice in request.choices]

    # Tokenize all inputs together
    encoded_inputs = registry.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

    # Run inference
    with torch.inference_mode():
        outputs = model(**encoded_inputs)
        # Take the diagonal of the output matrix to get one probability per choice
        logits = outputs.logits.diagonal()  # This gives us one logit per choice
        probabilities = torch.nn.functional.softmax(logits, dim=0)

    # Convert to list and return predictions
    predictions = probabilities.tolist()

    return {"probabilities": predictions, "choices": request.choices}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
