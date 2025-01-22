# Import the W&B Python Library and log into W&B
import wandb

from project.train import run

# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [1, 3, 7]},
        "optimizer_name": {"values": ["Adam", "SGD"]},
        "optimizer_params": {
            "parameters": {
                "lr": {"max": 0.1, "min": 0.0001},
                "weight_decay": {"max": 0.1, "min": 0.001},
                "momentum": {"max": 0.9, "min": 0.1, "conditions": {"optimizer_name": "SGD"}},
            }
        },
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration)

wandb.agent(sweep_id, function=run, count=10)
