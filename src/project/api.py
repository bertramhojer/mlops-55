from contextlib import asynccontextmanager

import fastapi
import torch
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer, BertForSequenceClassification


class Registry:
    """Models for the API."""

    model: BertForSequenceClassification
    tokenizer: AutoTokenizer

    def get(self) -> BertForSequenceClassification:
        """Get the client."""
        return self.model


class TestRequest(BaseModel):
    """Request for testing."""

    query: str


class PredictionRequest(BaseModel):
    """Request for prediction."""

    query: str
    choices: list[str]


registry = Registry()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Lifespan for the FastAPI app."""
    print("Starting up...")
    registry.model = BertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=4)
    registry.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    registry.model.eval()
    print("Model loaded successfully")
    yield
    del registry.model
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


@app.post("/test")
def test(request: TestRequest):
    """Test endpoint."""
    return {"query": request.query, "response": "I'm not qualified to answer that (ANY) question."}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict endpoint."""
    model = registry.get()  # No arguments needed

    inputs = [f"{request.query} [SEP] {choice}" for choice in request.choices]

    # Tokenize all inputs together
    encoded_inputs = registry.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

    # Run inference
    with torch.inference_mode():
        outputs = model(**encoded_inputs)
        logits = outputs.logits.diagonal()  # This gives us one logit per choice
        probabilities = torch.nn.functional.softmax(logits, dim=0)

    # Convert to list and return predictions
    predictions = probabilities.tolist()

    return {
        "predictions": predictions,
        "choices": request.choices,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
