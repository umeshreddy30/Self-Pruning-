# Self-Pruning Neural Network (CIFAR-10) — Report

_Generated: 2026-04-20 16:37:21_

## What is the L1 gate penalty doing?

Each `PrunableLinear` layer learns `gate_scores`, then applies `gates = sigmoid(gate_scores)` so every weight has a soft mask (g ∈ (0,1)). During training we add an L1 penalty on the gate values: `λ × mean(|gates|)`.

Intuition: L1 encourages many gate values to shrink toward 0, which suppresses the corresponding weights (effectively pruning them). Larger λ typically increases sparsity but can reduce accuracy.

## Experiment setup

- **Dataset**: CIFAR-10 (torchvision) with random crop + horizontal flip augmentation
- **Epochs**: 20
- **Batch size**: 256
- **LR**: 0.003
- **Hidden dims**: [1024, 512, 256]
- **Dropout**: 0.3
- **Gate init (pre-sigmoid)**: 2.0
- **Prune threshold**: 0.01

## Results (lambda sweep)

| λ | Best test accuracy | Final sparsity |
|---:|---:|---:|
| `0.0e+00` | **57.13%** | **0.00%** |
| `1.0e-05` | **56.98%** | **0.00%** |
| `1.0e-04` | **57.14%** | **0.00%** |
| `1.0e-03` | **57.25%** | **0.00%** |

## Artifacts

- `outputs/lambda_comparison.png`: accuracy vs sparsity across λ
- `outputs/gates_lam*.png`: gate histograms (with threshold line)
- `outputs/curves_lam*.png`: loss/accuracy/sparsity curves per λ
- `outputs/lambda_sweep_results.json`: machine-readable sweep outputs
