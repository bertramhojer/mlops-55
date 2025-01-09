from project.mmlu_loader import load_mmlu_dataset

# Load specific subjects
dataset = load_mmlu_dataset(subjects=["anatomy", "philosophy"], split="test", subset_size=100, seed=42)

# Use the dataset
for example in dataset:
    print(f"Question: {example['question']}")
    print(f"Choices: {example['choices']}")
    print(f"Answer: {example['answer']}")
