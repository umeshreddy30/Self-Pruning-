"""
utils.py
--------
Data loading, logging helpers, evaluation utilities, and plotting functions.
All functions are standalone and reusable across experiments.
"""

import logging
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "self_pruning", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout with a clean format."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        # ASCII-only to avoid Windows console encoding issues.
        fmt="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return (train_loader, test_loader).

    Training set uses standard augmentations (random crop + horizontal flip).
    Test set uses only normalisation – no augmentation.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks a running mean of a scalar metric (loss, accuracy, …)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum   = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum   += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate *model* on *loader* and return top-1 accuracy in [0, 1]."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return correct / total


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

def log_sparsity(
    sparsity_dict: dict[str, float],
    logger: logging.Logger,
    prefix: str = "",
) -> None:
    """Pretty-print a sparsity dict returned by model.compute_sparsity()."""
    parts = [f"{k}={v * 100:.1f}%" for k, v in sparsity_dict.items()]
    logger.info(f"{prefix}Sparsity → {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gate_histogram(
    gate_values: torch.Tensor,
    lam: float,
    save_path: str,
    threshold: float = 0.01,
    bins: int = 80,
) -> None:
    """Plot a histogram of all gate values and save to *save_path*.

    A vertical dashed line marks the pruning threshold.
    """
    vals = gate_values.numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5,
               label=f"threshold = {threshold}")
    ax.set_xlabel("Gate value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Distribution of Gate Values  |  λ = {lam}\n"
        f"Fraction < {threshold}: {(vals < threshold).mean() * 100:.1f}%",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list],
    lam: float,
    save_path: str,
) -> None:
    """Plot loss, accuracy, and sparsity curves for one lambda experiment.

    Parameters
    ----------
    history  : dict with keys 'train_loss', 'test_acc', 'sparsity'
    lam      : lambda value for the title
    save_path: where to save the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], color="#4C72B0")
    axes[0].set_title("Total Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, [a * 100 for a in history["test_acc"]], color="#55A868")
    axes[1].set_title("Test Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")

    axes[2].plot(epochs, [s * 100 for s in history["sparsity"]], color="#C44E52")
    axes[2].set_title("Overall Sparsity (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Sparsity (%)")

    fig.suptitle(f"Training curves  |  λ = {lam}", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_comparison(
    results: list[dict],
    save_path: str,
) -> None:
    """Bar chart comparing accuracy and sparsity across lambda values.

    Parameters
    ----------
    results  : list of dicts, each with keys 'lam', 'test_acc', 'sparsity'
    save_path: where to save the figure
    """
    lams      = [r["lam"]      for r in results]
    accs      = [r["test_acc"] * 100 for r in results]
    sparsities = [r["sparsity"] * 100 for r in results]

    x      = np.arange(len(lams))
    width  = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, accs,      width, label="Test Accuracy (%)", color="#4C72B0", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, sparsities, width, label="Sparsity (%)",      color="#C44E52", alpha=0.85)

    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#4C72B0")
    ax2.set_ylabel("Sparsity (%)",      fontsize=12, color="#C44E52")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lams])
    ax1.set_title("Accuracy vs. Sparsity for Different λ Values", fontsize=13)

    lines  = [bars1, bars2]
    labels = ["Test Accuracy (%)", "Sparsity (%)"]
    ax1.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save a training checkpoint dict to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    if logger:
        logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a checkpoint from *path* into *model* (and optionally *optimizer*)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint