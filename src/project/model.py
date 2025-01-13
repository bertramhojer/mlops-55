import typing

import lightning as l
import torch
from transformers import AutoModelForSequenceClassification


class ModernBERTQA(l.LightningModule):
    """Modern BERT QA model for sequence classification."""

    def __init__(
        self,
        model_name: str,
        num_choices: int,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_params: dict[str, typing.Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_choices)
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self._validate_optimizer()

    def _validate_optimizer(self):
        """Validate optimizer parameters fits with optimizer class."""
        try:
            self.optimizer_cls(self.parameters(), **self.optimizer_params)
        except TypeError as e:
            msg = f"Optimizer parameters are not compatible with optimizer class: {e}"
            raise ValueError(msg) from e

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step of the model."""
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", output.loss)
        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step of the model."""
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("val_loss", output.loss)
        return output.loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step of the model."""
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("test_loss", output.loss)
        return output.loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for the model."""
        return self.optimizer_cls(self.parameters(), **self.optimizer_params)
