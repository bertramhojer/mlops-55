import torch


def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]).long(),
        "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in batch]).long(),
        "labels": torch.stack([torch.tensor(item["labels"]) for item in batch]).long(),
    }