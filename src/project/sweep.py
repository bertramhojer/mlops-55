import typer

import wandb
from project.settings import settings
from project.train import run


def main() -> None:
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val/loss"},
        "parameters": {
            "train.epochs": {"values": [3]},
            "train.n_train_samples": {"values": [5000]},
            "train.n_val_samples": {"values": [1000]},
            "train.batch_size": {"values": [4, 16, 32]},
            "optimizer.optimizer_name": {"values": ["Adam", "SGD"]},
            "optimizer.optimizer_params.lr": {"min": 0.0001, "max": 0.1},
            "optimizer.optimizer_params.weight_decay": {"min": 0.001, "max": 0.1},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=settings.WANDB_PROJECT)

    wandb.agent(sweep_id, function=run, count=10)

if __name__ == "__main__":
    typer.run(main)
