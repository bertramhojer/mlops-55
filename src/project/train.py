import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from project.model import ModernBERTQA


from project.data import get_processed_datasets

# Will enable run on certain servers, do no delete
import torch._dynamo  #noqa: F401
torch._dynamo.config.suppress_errors = True  #noqa: F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def train_model(
    model_name: str = typer.Argument("answerdotai/ModernBERT-base", help="Model name to train"),
    subjects: list[str] | None = typer.Option(None, help="Subjects to include in dataset"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    epochs: int = typer.Option(100, help="Number of epochs to train"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate for training"),
    output_dir: str = typer.Option("models", help="Output directory for saving model"),
    mode: str = typer.Option("multiclass", help="Mode for training"),
    train_subset_size: int | None = typer.Option(None, help="Subset size for training"),
    val_subset_size: int | None = typer.Option(None, help="Subset size for validation"),
):
    """
    Train model, saves model to output_dir.

    Args:
            model_name: Model name to train
            subjects: Subjects to include in dataset
            batch_size: Batch size for training
            epochs: Number of epochs to train
            learning_rate: Learning rate for training
            output_dir: Output directory for saving model
            mode: Mode for training
            train_subset_size: Subset size for training
            val_subset_size: Subset size for validation

    Returns:
            None
    
            TODO: fix binary classification
    """
    # Load processed datasets
    print("Loading datasets...")
    train_dataset = get_processed_datasets(
        split="auxiliary_train", # TODO: confirm with Bertram this is right?
        subjects=subjects,
        mode=mode,
        subset_size=train_subset_size
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = get_processed_datasets(
        split="validation",
        subjects=subjects,
        mode=mode,
        subset_size=val_subset_size
    )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_choices = train_dataset.__getoptions__()

    # Initialize model
    model = ModernBERTQA(
        model_name,
        num_choices=num_choices,
        optimizer_cls=torch.optim.AdamW,
        optimizer_params={"lr": learning_rate}
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir, monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=100, verbose=True, mode="min")

    # Train and save model
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if DEVICE.type == "cuda" else None,
        max_epochs=epochs, devices=list(range(torch.cuda.device_count())),
        default_root_dir=output_dir
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    


if __name__ == "__main__":
    app()


