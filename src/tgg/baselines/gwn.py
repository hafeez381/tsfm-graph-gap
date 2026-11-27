"""Graph WaveNet reproduction: the STGNN reference (Decision 12, Decision 1, Decision 4).

Input (B, L, N, 2) holds the z-scored lookback and the time of day; output (B, N, H) is the
de-normalised forecast in mph. The scaler is fit on the full training split, never the fraction block.
"""
import numpy as np
import torch

H, L = 12, 12

DEFAULTS = {"hidden_size": 32, "ff_size": 256, "n_layers": 8, "temporal_kernel_size": 2,
            "spatial_kernel_size": 2, "emb_size": 10, "dilation": 2, "dilation_mod": 2,
            "norm": "batch", "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "batch_windows": 64,
            "max_epochs": 100, "patience_epochs": 20}


class Scaler:
    """z-score fit on the full training split (Decision 4: scalers are part of the pipeline, not the fraction)."""

    def __init__(self, mean: float, std: float):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def fit(cls, X_filled: np.ndarray, train_end: int) -> "Scaler":
        x = X_filled[:train_end]
        return cls(x.mean(), x.std())

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def assert_trained(model: torch.nn.Module) -> None:
    """Raise unless every parameter requires grad (Graph WaveNet is trained, never frozen)."""
    if not all(p.requires_grad for p in model.parameters()):
        raise AssertionError("a Graph WaveNet parameter is frozen")


def build(N: int, A: np.ndarray, cfg: dict | None = None):
    """GraphWaveNetModel plus its edge_index/edge_weight from the road graph A (Decision 2)."""
    from tsl.nn.models import GraphWaveNetModel
    from tsl.ops.connectivity import adj_to_edge_index

    c = {**DEFAULTS, **(cfg or {})}
    model = GraphWaveNetModel(
        input_size=2, output_size=1, horizon=H, hidden_size=c["hidden_size"], ff_size=c["ff_size"],
        n_layers=c["n_layers"], temporal_kernel_size=c["temporal_kernel_size"],
        spatial_kernel_size=c["spatial_kernel_size"], learned_adjacency=True, n_nodes=N,
        emb_size=c["emb_size"], dilation=c["dilation"], dilation_mod=c["dilation_mod"],
        norm=c["norm"], dropout=c["dropout"])
    edge_index, edge_weight = adj_to_edge_index(torch.from_numpy(A))
    assert_trained(model)
    return model, edge_index, edge_weight


def lookback_ids(t: np.ndarray, Lw: int = L) -> np.ndarray:
    """(n, Lw) step ids, oldest first, lb[:, -1] == t (Decision 3's window convention)."""
    return t[:, None] - np.arange(Lw - 1, -1, -1)[None, :]


def make_input(X_filled: np.ndarray, tod: np.ndarray, t: np.ndarray, scaler: Scaler) -> np.ndarray:
    """(n, L, N, 2) float32: z-scored speed and time of day, oldest step first."""
    lb = lookback_ids(t)
    x = scaler.transform(X_filled[lb]).astype(np.float32)              # (n, L, N)
    tod_b = np.broadcast_to(tod[lb][:, :, None], x.shape).astype(np.float32)
    return np.stack([x, tod_b], axis=-1)


def predict(model: torch.nn.Module, edge_index, edge_weight, x: torch.Tensor, scaler: Scaler) -> torch.Tensor:
    """(B, N, H) de-normalised forecast in mph."""
    out = model(x, edge_index, edge_weight)                            # (B, H, N, 1)
    out = out.squeeze(-1).transpose(1, 2)                              # (B, N, H)
    return out * scaler.std + scaler.mean
