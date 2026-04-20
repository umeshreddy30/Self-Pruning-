"""
models.py
---------
Defines the PrunableLinear layer and the full Self-Pruning Neural Network.

Each PrunableLinear layer owns a set of learnable gate_scores.
During the forward pass:
    gates         = sigmoid(gate_scores)          ∈ (0, 1)
    pruned_weights = weight * gates
    output         = pruned_weights @ x + bias

The sigmoid gates are soft masks that the optimizer can push towards 0
(pruned) or 1 (kept). A separate L1 sparsity penalty applied during
training encourages many gates to collapse to ~0.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """A fully-connected layer with learnable, per-weight soft gates.

    Parameters
    ----------
    in_features  : int   – number of input features
    out_features : int   – number of output neurons
    bias         : bool  – whether to include a bias term (default True)
    gate_init    : float – initial value for gate_scores before sigmoid.
                          sigmoid(2.0) ≈ 0.88, so weights start mostly active.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # ── Trainable parameters ──────────────────────────────────────────
        # Standard weight matrix, initialised with Kaiming uniform (same as
        # nn.Linear) so training starts in a healthy gradient regime.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # gate_scores has the SAME shape as weight.
        # Initialised to gate_init so sigmoid(gate_init) ≈ 0.88 (most weights
        # start active). The optimiser will push scores toward large negatives
        # (≈ pruned) when the sparsity penalty is large enough.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), gate_init)
        )

    # ------------------------------------------------------------------
    def gates(self) -> torch.Tensor:
        """Return the current soft gate values in (0, 1)."""
        return torch.sigmoid(self.gate_scores)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = self.gates()                     # (out, in)
        pruned_weight = self.weight * gates              # element-wise mask
        return F.linear(x, pruned_weight, self.bias)

    # ------------------------------------------------------------------
    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values – used as the regularisation term."""
        return self.gates().abs().sum()

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class SelfPruningNet(nn.Module):
    """A feed-forward classifier with multiple PrunableLinear layers.

    Architecture (for CIFAR-10, 32×32×3 → 10 classes):
        Flatten → [PrunableLinear → BN → ReLU → Dropout] × n_hidden → PrunableLinear

    Parameters
    ----------
    input_dim    : int        – flattened input size (3072 for CIFAR-10)
    hidden_dims  : list[int]  – sizes of hidden layers, e.g. [1024, 512, 256]
    num_classes  : int        – number of output classes (10 for CIFAR-10)
    dropout_p    : float      – dropout probability after each hidden layer
    gate_init    : float      – forwarded to every PrunableLinear
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        gate_init: float = 2.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        self.flatten = nn.Flatten()

        # Build hidden blocks dynamically
        dims       = [input_dim] + hidden_dims
        blocks: list[nn.Module] = []

        for i in range(len(dims) - 1):
            blocks.append(PrunableLinear(dims[i], dims[i + 1], gate_init=gate_init))
            blocks.append(nn.BatchNorm1d(dims[i + 1]))
            blocks.append(nn.ReLU(inplace=True))
            if dropout_p > 0:
                blocks.append(nn.Dropout(p=dropout_p))

        self.hidden = nn.Sequential(*blocks)

        # Output layer – also prunable so we can measure its sparsity
        self.head = PrunableLinear(dims[-1], num_classes, gate_init=gate_init)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.hidden(x)
        return self.head(x)

    # ------------------------------------------------------------------
    def prunable_layers(self) -> list[PrunableLinear]:
        """Return every PrunableLinear contained in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    # ------------------------------------------------------------------
    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across ALL prunable layers."""
        losses = [layer.sparsity_loss() for layer in self.prunable_layers()]
        return torch.stack(losses).sum()

    # ------------------------------------------------------------------
    def compute_sparsity(self, threshold: float = 0.01) -> dict[str, float]:
        """Compute the fraction of gates below *threshold* for each layer
        and for the whole network.

        Returns a dict with keys 'layerN' and 'overall'.
        """
        total_weights = 0
        total_pruned  = 0
        stats: dict[str, float] = {}

        for idx, layer in enumerate(self.prunable_layers()):
            with torch.no_grad():
                g       = layer.gates()
                pruned  = (g < threshold).sum().item()
                total   = g.numel()
                sparsity = pruned / total

            stats[f"layer{idx}"] = sparsity
            total_weights        += total
            total_pruned         += pruned

        stats["overall"] = total_pruned / total_weights if total_weights > 0 else 0.0
        return stats

    # ------------------------------------------------------------------
    def gate_values_all(self) -> torch.Tensor:
        """Concatenate all gate values into a single flat tensor (for plotting)."""
        with torch.no_grad():
            parts = [layer.gates().flatten() for layer in self.prunable_layers()]
        return torch.cat(parts).cpu()