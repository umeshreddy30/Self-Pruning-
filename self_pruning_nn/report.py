"""
report.py
---------
Generate a Markdown report from the saved sweep JSON produced by train.py.

Usage:
  python report.py --results ./outputs/lambda_sweep_results.json --out REPORT.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _fmt_float(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}"


def _fmt_percent(x: float, ndigits: int = 2) -> str:
    return f"{x * 100:.{ndigits}f}%"


def _fmt_lam(lam: float) -> str:
    return f"{lam:.1e}"


def render_report(payload: dict) -> str:
    cfg = payload.get("config", {})
    results = payload.get("results", [])

    lines: list[str] = []
    lines.append("# Self-Pruning Neural Network (CIFAR-10) — Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## What is the L1 gate penalty doing?")
    lines.append("")
    lines.append(
        "Each `PrunableLinear` layer learns `gate_scores`, then applies `gates = sigmoid(gate_scores)` "
        "so every weight has a soft mask (g ∈ (0,1)). During training we add an L1 penalty on "
        "the gate values: `λ × mean(|gates|)`."
    )
    lines.append("")
    lines.append(
        "Intuition: L1 encourages many gate values to shrink toward 0, which suppresses the corresponding "
        "weights (effectively pruning them). Larger λ typically increases sparsity but can reduce accuracy."
    )
    lines.append("")
    lines.append("## Experiment setup")
    lines.append("")
    lines.append("- **Dataset**: CIFAR-10 (torchvision) with random crop + horizontal flip augmentation")
    lines.append(f"- **Epochs**: {cfg.get('epochs', 'n/a')}")
    lines.append(f"- **Batch size**: {cfg.get('batch_size', 'n/a')}")
    lines.append(f"- **LR**: {cfg.get('lr', 'n/a')}")
    lines.append(f"- **Hidden dims**: {cfg.get('hidden_dims', 'n/a')}")
    lines.append(f"- **Dropout**: {cfg.get('dropout_p', 'n/a')}")
    lines.append(f"- **Gate init (pre-sigmoid)**: {cfg.get('gate_init', 'n/a')}")
    lines.append(f"- **Prune threshold**: {cfg.get('gate_threshold', 'n/a')}")
    lines.append("")
    lines.append("## Results (lambda sweep)")
    lines.append("")
    lines.append("| λ | Best test accuracy | Final sparsity |")
    lines.append("|---:|---:|---:|")

    # Sort by lambda value for stable tables
    for r in sorted(results, key=lambda d: float(d.get("lam", 0.0))):
        lam = float(r.get("lam", 0.0))
        acc = float(r.get("test_acc", 0.0))
        sp = float(r.get("sparsity", 0.0))
        lines.append(f"| `{_fmt_lam(lam)}` | **{_fmt_percent(acc)}** | **{_fmt_percent(sp)}** |")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/lambda_comparison.png`: accuracy vs sparsity across λ")
    lines.append("- `outputs/gates_lam*.png`: gate histograms (with threshold line)")
    lines.append("- `outputs/curves_lam*.png`: loss/accuracy/sparsity curves per λ")
    lines.append("- `outputs/lambda_sweep_results.json`: machine-readable sweep outputs")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True, help="Path to lambda_sweep_results.json")
    p.add_argument("--out", type=str, default="REPORT.md", help="Output markdown filename")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    md = render_report(payload)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()

