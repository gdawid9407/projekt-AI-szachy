import argparse
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import MODELS_DIR, get_device
from src.dataset import ChessMemmapDataset
from src.network import ChessValueNetwork


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Executes one full training epoch over the training dataset.

    Args:
        model: Neural network undergoing optimization.
        dataloader: Training DataLoader producing batches of (B, 14, 8, 8) and (B, 1).
        criterion: Loss function (MSELoss).
        optimizer: Optimization algorithm (AdamW).
        device: Accelerator (mps / cuda / cpu).
        max_grad_norm: Value threshold for gradient clipping.

    Returns:
        float: Mean training loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    for batch_x, batch_y in progress:
        # Move tensors to target device: batch_x (B, 14, 8, 8), batch_y (B, 1)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass: produces predictions in range [-1.0, 1.0]
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backpropagation: compute gradients dL/dθ
        loss.backward()

        # Gradient clipping prevents exploding gradients through deep convolutional stacks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Parameter update: θ <- θ - η * ∇θ
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, num_batches)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluates the model on the validation split without tracking gradients.

    Returns:
        float: Mean validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / max(1, num_batches)


def run_training(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
    checkpoint_name: str = "chess_value_net.pt",
) -> Path:
    """
    Sets up datasets, model, and runs the supervised training loop.
    """
    device = get_device()
    print(f"Executing training on compute device: {device}")

    # Load memory-mapped binary dataset
    full_dataset = ChessMemmapDataset()
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Dataset partitioned: {train_size:,} train samples | {val_size:,} validation samples.")

    # DataLoaders: shuffle train set to break game-level spatial correlations
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
    )

    # Initialize model, loss function, and optimizer
    model = ChessValueNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    checkpoint_path = MODELS_DIR / checkpoint_name

    print("\n--- Starting Supervised Training ---")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        is_best = val_loss < best_val_loss
        status_suffix = ""
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            status_suffix = f" [Checkpoint saved -> {checkpoint_path.name}]"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f}"
            f"{status_suffix}"
        )

    print(f"\nTraining complete. Optimal model weights saved at: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Chess Value Network.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="chess_value_net.pt",
        help="Target checkpoint filename in models/",
    )

    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_name=args.checkpoint,
    )
