from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import torch
from typing_extensions import Annotated

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset import load_autoencoder_data
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path,
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


# convAutoEncoder config type


@app.command()
def train_conv_auto_encoder(
    data_dir: str = typer.Option("", "--data-dir", "-d"),
    dataset: str = "",  # todo: change it
    dropout_p: float = typer.Option(0.2, "--dropout-p", "-dp"),
    latent_dim: Annotated[int, typer.Option(min=1)] = 10,
    shape=typer.Option((200, 200), "--shape", "-s"),
    seed: int = typer.Option(42, "--seed", "-s"),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    max_epochs: int = typer.Option(100, "--max-epochs", "-me"),
    patience: int = typer.Option(10, "--patience", "-p"),
    min_delta: float = typer.Option(0.0001, "--min-delta", "-md"),
    batch_size: int = typer.Option(200, "--batch-size", "-bs"),
    val_split: float = typer.Option(0.1, "--val-split", "-vs"),
    test_split: float = typer.Option(0.1, "--test-split", "-ts"),
):
    # import the model form the models directory
    from src.modeling.models import ConvAutoEncoder
    from src.modeling.models import LitConvAutoEncoder

    # architecture = {
    #     "shape": shape,
    #     "latent_dim": int(latent_dim),
    #     "dropout_p": float(dropout_p),
    # }

    torch.manual_seed(seed)
    model = ConvAutoEncoder(latent_dim=int(latent_dim), shape=(200, 200))
    version = model.__version__
    logger.info(f"ConvAutoEncoder version: {version}")
    logger.info(f"ConvAutoEncoder shape: {model.shape}")

    train_dataloader, val_dataloader, _ = load_autoencoder_data(
        data_dir,
        val_split=val_split,
        test_split=test_split,
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
    )
    logger.info(f"Loaded data from {data_dir}")

    autoencoder = LitConvAutoEncoder(model)

    # Train the model
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                min_delta=min_delta,
                patience=patience,
                verbose=False,
            )
        ],
        default_root_dir=f"{MODELS_DIR}/dataset_{dataset}/ConvAutoEncoder/{version}/training_runs",
    )

    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    app()
