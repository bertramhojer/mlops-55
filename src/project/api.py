from contextlib import asynccontextmanager

import fastapi
import torch
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer

import wandb
from project.model import ModernBERTQA  # Make sure this import works


class Registry:
    """Models for the API."""

    model: ModernBERTQA
    tokenizer: AutoTokenizer

    def get(self) -> ModernBERTQA:
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

    # Initialize wandb in disabled mode (we just need to access artifacts)
    wandb.init(mode="disabled")

    # Get the artifact
    api = wandb.Api()
    artifact = api.artifact("mlops_55/ModernBERTQA/model-jvz6josi:v0", type="model")
    artifact_dir = artifact.download()

    # Load model and tokenizer
    registry.model = ModernBERTQA(
        model_name="answerdotai/ModernBERT-base", optimizer_cls=torch.optim.Adam, optimizer_params={"lr": 1e-5}
    )

    # Load the checkpoint
    checkpoint = torch.load(f"{artifact_dir}/model.ckpt", map_location="cpu", weights_only=False)
    registry.model.load_state_dict(checkpoint["state_dict"])

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


# Modify your predict endpoint to work with the new model
@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict endpoint."""
    model = registry.get()

    # Create input pairs for each option
    input_texts = [f"Question: {request.query} Answer: {choice}" for choice in request.choices]

    # Tokenize inputs
    encoded = registry.tokenizer(input_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Add dummy labels (required by model)
    encoded["labels"] = torch.zeros(len(request.choices), dtype=torch.long)

    # Run inference
    with torch.inference_mode():
        outputs = model(**encoded)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get probability of being correct (second column)
        correct_probs = probabilities[:, 1].tolist()

    return {
        "predictions": correct_probs,
        "choices": request.choices,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
