"""
Reusable training engine: EarlyStopping, per-epoch train/eval loops, and the
two-phase run_phase driver shared across notebook experiments.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class EarlyStopping:
    """Monitor val_accuracy, restore best weights on stop (mirrors keras.callbacks.EarlyStopping)."""

    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> None:
        """Update best score/state from the latest monitored metric."""
        is_better = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if is_better:
            self.best_score = score
            self.counter = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best recorded state_dict back into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch (with a tqdm progress bar), return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def run_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    early_stop_patience: int,
    history: Dict[str, List[float]],
    phase_label: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights_tensor: torch.Tensor,
    device: torch.device,
    lr_reduce_factor: float,
    lr_reduce_patience: int,
    min_lr: float,
) -> Optional[float]:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=min_lr,
    )
    early_stopping = EarlyStopping(patience=early_stop_patience, mode="max")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        early_stopping.step(val_acc, model)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{phase_label}] epoch {epoch}/{num_epochs} "
            f"loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.2e}"
        )

        if early_stopping.should_stop:
            print(
                f"Early stopping triggered at epoch {epoch} (patience={early_stop_patience})."
            )
            break

    early_stopping.restore_best(model)
    return early_stopping.best_score
