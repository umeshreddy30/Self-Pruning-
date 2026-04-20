# Self-Pruning Neural Networks (Project Notes)

## Soft gates

In this project, every weight in `PrunableLinear` has a **learnable soft gate**:

- `gates = sigmoid(gate_scores)` so \(g \in (0,1)\)
- `pruned_weight = weight * gates`

If a gate is near 0, the corresponding weight contributes almost nothing; if it is near 1, the weight behaves normally.

## Sparsity objective

Training uses a combined loss:

- `CrossEntropyLoss + λ × mean(|gates|)`

The L1 term encourages smaller gate values, pushing many gates toward 0. Increasing λ typically increases sparsity.

## Sparsity measurement

We estimate sparsity as the **fraction of gates below a threshold**, e.g. threshold \(= 0.01\):

- `sparsity = mean(gates < threshold)`

This gives per-layer sparsity and overall sparsity across the network.

