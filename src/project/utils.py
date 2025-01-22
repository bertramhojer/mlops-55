import torch

def aggregate_over_options(logits: torch.Tensor, labels: torch.Tensor, num_options: int = 4):
    """Aggregate logits over options."""
    preds = logits.softmax(dim=1)[:, 1]
    preds = preds.unfold(0, num_options, num_options)
    labels = labels.unfold(0, num_options, num_options)
    preds = preds.argmax(dim=1)
    labels = labels.argmax(dim=1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    return preds, labels