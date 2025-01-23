import logging
import os
import sys
from contextlib import asynccontextmanager

import fastapi
import torch
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer

import wandb
from project.model import ModernBERTQA

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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
    try:
        logger.info("Starting up application...")

        # Get environment variables
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb_entity = os.getenv("WANDB_ENTITY")
        wandb_project = os.getenv("WANDB_PROJECT")

        if not all([wandb_api_key, wandb_entity, wandb_project]):
            msg = "Missing required wandb environment variables"
            raise ValueError(msg)

        logger.info("Initializing wandb...")
        # Initialize wandb with explicit settings
        wandb.login()
        wandb.init(mode="disabled")

        # Get the artifact
        logger.info("Fetching wandb artifact...")
        try:
            api = wandb.Api()
            artifact = api.artifact(f"{wandb_entity}/ModernBERTQA/model-jvz6josi:v0", type="model")
            artifact_dir = artifact.download()
            logger.info(f"Artifact downloaded to {artifact_dir}")
        except Exception as e:
            logger.error(f"Error downloading artifact: {str(e)}")
            raise

        # Load model and tokenizer
        logger.info("Loading model...")
        registry.model = ModernBERTQA(
            model_name="answerdotai/ModernBERT-base", optimizer_cls=torch.optim.Adam, optimizer_params={"lr": 1e-5}
        )

        # Load the checkpoint
        logger.info("Loading checkpoint...")
        checkpoint_path = os.path.join(artifact_dir, "model.ckpt")
        if not os.path.exists(checkpoint_path):
            msg = "Checkpoint not found"
            raise FileNotFoundError(msg)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        registry.model.load_state_dict(checkpoint["state_dict"])

        logger.info("Loading tokenizer...")
        registry.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        registry.model.eval()
        logger.info("Startup complete - Model loaded successfully")

        yield

        logger.info("Shutting down...")
        del registry.model
        del registry.tokenizer

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise


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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")  # noqa: S104
