"""
train.py
--------
Training loop and multi-lambda experiment runner for the Self-Pruning Network.

Usage:
    python train.py                         # run default experiment
    python train.py --lambdas 0 1e-4 1e-3  # sweep specific lambda values
    python train.py --epochs 30 --lr 1e-3  # custom hyperparameters
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import SelfPruningNet
from utils import (
    AverageMeter,
    compute_accuracy,
    get_cifar10_loaders,
    get_logger,
    log_sparsity,
    load_checkpoint,
    plot_gate_histogram,
    plot_lambda_comparison,
    plot_training_curves,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Config dataclass – single source of truth for hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_dir    : str       = "./data"
    batch_size  : int       = 256
    num_workers : int       = 2

    # Model
    hidden_dims : list[int] = field(default_factory=lambda: [1024, 512, 256])
    dropout_p   : float     = 0.3
    gate_init   : float     = 2.0          # sigmoid(2.0) ≈ 0.88

    # Optimisation
    epochs      : int       = 40
    lr          : float     = 3e-3
    weight_decay: float     = 1e-4         # L2 on weights (not gates)

    # Sparsity
    lam         : float     = 1e-4         # λ for sparsity penalty
    gate_threshold: float   = 0.01         # gates < this are "pruned"

    # Misc
    seed        : int       = 42
    output_dir  : str       = "./outputs"
    log_interval: int       = 50           # log every N batches
    save_best   : bool      = True


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    optimizer : optim.Optimizer,
    device    : torch.device,
    lam       : float,
    epoch     : int,
    config    : TrainConfig,
    logger,
) -> tuple[float, float, float]:
    """Run a single training epoch.

    Returns
    -------
    avg_total_loss  : float – total (cls + sparsity) loss averaged over batches
    avg_cls_loss    : float – classification loss only
    avg_sparse_loss : float – sparsity penalty only
    """
    model.train()

    total_meter  = AverageMeter("total_loss")
    cls_meter    = AverageMeter("cls_loss")
    sparse_meter = AverageMeter("sparse_loss")
    t0 = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        logits = model(images)

        # ── Losses ──────────────────────────────────────────────────────
        cls_loss    = criterion(logits, labels)
        sparse_loss = model.total_sparsity_loss()   # sum of L1(gates)

        # Normalise sparsity loss by total gate count so λ is scale-invariant
        n_gates     = sum(
            p.numel() for p in model.parameters() if p.requires_grad
            # only count gate_score params
        )
        # More precise: count only gate parameters
        n_gates = sum(
            layer.gate_scores.numel() for layer in model.prunable_layers()
        )
        norm_sparse = sparse_loss / n_gates        # mean gate value

        total_loss  = cls_loss + lam * norm_sparse

        total_loss.backward()

        # Gradient clipping prevents exploding gradients early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_meter.update(total_loss.item(), batch_size)
        cls_meter.update(cls_loss.item(), batch_size)
        sparse_meter.update(norm_sparse.item(), batch_size)

        if (batch_idx + 1) % config.log_interval == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Epoch {epoch:3d} | step {batch_idx + 1:4d}/{len(loader)} "
                f"| total={total_meter.avg:.4f} "
                f"cls={cls_meter.avg:.4f} "
                f"sparse={sparse_meter.avg:.4f} "
                f"({elapsed:.1f}s)"
            )

    return total_meter.avg, cls_meter.avg, sparse_meter.avg


# ---------------------------------------------------------------------------
# Full training run for a single lambda
# ---------------------------------------------------------------------------

def run_experiment(
    config   : TrainConfig,
    lam      : float,
    device   : torch.device,
    logger,
    train_loader,
    test_loader,
) -> dict:
    """Train the model for *config.epochs* epochs with sparsity weight *lam*.

    Returns a result dict:
        {
            'lam'       : float,
            'test_acc'  : float,   # best test accuracy in [0, 1]
            'sparsity'  : float,   # overall gate sparsity at end of training
            'history'   : dict,    # per-epoch logs
        }
    """
    logger.info("=" * 70)
    logger.info(f"Starting experiment  lam = {lam}")
    logger.info("=" * 70)

    torch.manual_seed(config.seed)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SelfPruningNet(
        input_dim   = 3 * 32 * 32,          # CIFAR-10
        hidden_dims = config.hidden_dims,
        num_classes = 10,
        dropout_p   = config.dropout_p,
        gate_init   = config.gate_init,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Use different parameter groups so we can apply weight_decay only to
    # weights and biases – NOT to gate_scores (they have their own L1 reg).
    weight_params = []
    gate_params   = []

    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            weight_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": weight_params, "weight_decay": config.weight_decay},
            {"params": gate_params,   "weight_decay": 0.0},    # no L2 on gates
        ],
        lr=config.lr,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── History ───────────────────────────────────────────────────────────
    history = {
        "train_loss"    : [],
        "cls_loss"      : [],
        "sparse_loss"   : [],
        "test_acc"      : [],
        "sparsity"      : [],
    }

    best_acc     = 0.0
    lam_tag      = str(lam).replace(".", "_")
    ckpt_path    = os.path.join(config.output_dir, f"best_model_lam{lam_tag}.pt")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        t_loss, c_loss, s_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, lam, epoch, config, logger
        )
        scheduler.step()

        # Evaluate accuracy
        test_acc = compute_accuracy(model, test_loader, device)

        # Compute sparsity
        sparsity_dict = model.compute_sparsity(threshold=config.gate_threshold)
        overall_sparsity = sparsity_dict["overall"]

        history["train_loss"].append(t_loss)
        history["cls_loss"].append(c_loss)
        history["sparse_loss"].append(s_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(overall_sparsity)

        logger.info(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"loss={t_loss:.4f} | "
            f"acc={test_acc * 100:.2f}% | "
            f"sparsity={overall_sparsity * 100:.1f}% | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        log_sparsity(sparsity_dict, logger, prefix="  ")

        # Save best checkpoint
        if config.save_best and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                {
                    "epoch"          : epoch,
                    "lam"            : lam,
                    "model_state"    : model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc"       : best_acc,
                },
                ckpt_path,
                logger,
            )

    # ── Post-training plots ───────────────────────────────────────────────
    gate_vals = model.gate_values_all()

    plot_gate_histogram(
        gate_vals,
        lam=lam,
        save_path=os.path.join(config.output_dir, f"gates_lam{lam_tag}.png"),
        threshold=config.gate_threshold,
    )

    plot_training_curves(
        history,
        lam=lam,
        save_path=os.path.join(config.output_dir, f"curves_lam{lam_tag}.png"),
    )

    logger.info(
        f"\n{'─' * 60}\n"
        f"Experiment DONE | λ={lam} | best_acc={best_acc * 100:.2f}% | "
        f"final_sparsity={history['sparsity'][-1] * 100:.1f}%\n"
        f"{'─' * 60}"
    )

    return {
        "lam"      : lam,
        "test_acc" : best_acc,
        "sparsity" : history["sparsity"][-1],
        "history"  : history,
    }


# ---------------------------------------------------------------------------
# Multi-lambda sweep
# ---------------------------------------------------------------------------

def run_lambda_sweep(
    lambdas : list[float],
    config  : TrainConfig,
    device  : torch.device,
    logger,
) -> list[dict]:
    """Run one experiment per lambda value and produce a comparison plot."""
    train_loader, test_loader = get_cifar10_loaders(
        data_dir    = config.data_dir,
        batch_size  = config.batch_size,
        num_workers = config.num_workers,
        pin_memory  = device.type == "cuda",
    )

    all_results = []
    for lam in lambdas:
        cfg     = TrainConfig(**{**config.__dict__, "lam": lam})   # copy with new lam
        result  = run_experiment(cfg, lam, device, logger, train_loader, test_loader)
        all_results.append(result)

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info(f"{'lam':>12}  {'Best Acc':>10}  {'Sparsity':>10}")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(f"{r['lam']:>12.1e}  {r['test_acc'] * 100:>9.2f}%  {r['sparsity'] * 100:>9.1f}%")
    logger.info("=" * 60)

    # Comparison plot
    plot_lambda_comparison(
        all_results,
        save_path=os.path.join(config.output_dir, "lambda_comparison.png"),
    )
    logger.info(f"Comparison plot saved to {config.output_dir}/lambda_comparison.png")

    # Persist results for report generation / downstream consumption
    results_path = os.path.join(config.output_dir, "lambda_sweep_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "batch_size": config.batch_size,
                    "hidden_dims": config.hidden_dims,
                    "dropout_p": config.dropout_p,
                    "gate_init": config.gate_init,
                    "gate_threshold": config.gate_threshold,
                    "weight_decay": config.weight_decay,
                    "seed": config.seed,
                    "data_dir": config.data_dir,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Sweep JSON saved to {results_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network – CIFAR-10")
    parser.add_argument("--lambdas",     nargs="+", type=float,
                        default=[0.0, 1e-5, 1e-4, 1e-3],
                        help="Sparsity weights to sweep over")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--lr",          type=float, default=3e-3)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--hidden_dims", nargs="+",  type=int, default=[1024, 512, 256])
    parser.add_argument("--dropout_p",   type=float, default=0.3)
    parser.add_argument("--gate_init",   type=float, default=2.0)
    parser.add_argument("--threshold",   type=float, default=0.01,
                        help="Gates below this are counted as 'pruned'")
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--output_dir",  type=str,   default="./outputs")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=2)
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    logger = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    config = TrainConfig(
        data_dir      = args.data_dir,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        hidden_dims   = args.hidden_dims,
        dropout_p     = args.dropout_p,
        gate_init     = args.gate_init,
        epochs        = args.epochs,
        lr            = args.lr,
        gate_threshold= args.threshold,
        seed          = args.seed,
        output_dir    = args.output_dir,
    )

    run_lambda_sweep(
        lambdas = args.lambdas,
        config  = config,
        device  = device,
        logger  = logger,
    )


if __name__ == "__main__":
    main()