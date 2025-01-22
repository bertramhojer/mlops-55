import torch
from transformers import AutoTokenizer

import wandb
from project.model import ModernBERTQA

# Disable wandb sync and logging
wandb.init(mode="disabled")

# Get artifact without creating a run
api = wandb.Api()
artifact = api.artifact("mlops_55/ModernBERTQA/model-jvz6josi:v0", type="model")
artifact_dir = artifact.download()

# Load the model and tokenizer
model = ModernBERTQA(
    model_name="answerdotai/ModernBERT-base", optimizer_cls=torch.optim.Adam, optimizer_params={"lr": 1e-5}
)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# Load checkpoint
checkpoint = torch.load(f"{artifact_dir}/model.ckpt", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Prepare question and options
question = "What is the capital of France?"
options = ["sadgas", "Paris", "sadgasd", "sadgsad"]

# Create input pairs for each option
input_texts = [f"Question: {question} Answer: {option}" for option in options]

# Tokenize inputs
encoded = tokenizer(input_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Add dummy labels (0 for all options since we don't know the correct answer during inference)
encoded["labels"] = torch.zeros(len(options), dtype=torch.long)

# Run inference
with torch.no_grad():
    outputs = model(**encoded)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predictions = torch.argmax(outputs.logits, dim=1)

# Print results
print("\nInference Results:")
print("-----------------")
for i, option in enumerate(options):
    print(f"\nOption: {option}")
    print(f"Raw logits: {outputs.logits[i]}")
    print(f"Probabilities: {probabilities[i]}")
    print(f"Prediction: {predictions[i]} ({'correct' if predictions[i] == 1 else 'incorrect'})")

# Print the most likely answer
best_option_idx = torch.argmax(probabilities[:, 1])  # Get index of option with highest probability of being correct
print(f"\nMost likely answer: {options[best_option_idx]}")
