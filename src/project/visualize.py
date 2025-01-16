from project.configs import DatasetConfig, OptimizerConfig, TrainConfig
from project.data import get_processed_datasets
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

# load data
train_dataset = get_processed_datasets(
        split="auxiliary_train",
        mode="multiclass",
        subset_size=1000,
        )
git
# look at data label distribution
labels = [item["labels"] for item in train_dataset]
label_counts = np.unique(labels, return_counts=True)
plt.bar(label_counts[0], label_counts[1])
plt.savefig("train_label_distribution.png")

test_dataset = get_processed_datasets(
        split="test",
        mode="multiclass",
        subset_size=1000,
        )

# look at data label distribution
labels = [item["labels"] for item in test_dataset]
label_counts = np.unique(labels, return_counts=True)
plt.bar(label_counts[0], label_counts[1])
plt.savefig("test_label_distribution.png")

# print an example Q&A
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
train_example = tokenizer.decode(train_dataset[0]["input_ids"]).replace("[PAD]", "")
answer = train_dataset[0]["labels"]
print(f"Question: {train_example}, Answer: {answer}")

