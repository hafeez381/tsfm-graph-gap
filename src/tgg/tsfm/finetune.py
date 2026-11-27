"""Full fine-tuning of Chronos-Bolt Small: adapts the whole encoder, no spatial information (Decision 13)."""
import numpy as np
import torch

from tgg.tsfm import bolt

MEDIAN_IDX = bolt.MEDIAN_IDX
MIN_LOOKBACK_STD = 0.5                                               # Decision 13: drop flat windows


def load(device: str = "cuda"):
    """Chronos-Bolt Small with every parameter trainable (the opposite of bolt.load's frozen copy)."""
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(bolt.MODEL_ID, device_map=device, torch_dtype=torch.float32)
    pipeline.model.train()
    for p in pipeline.model.parameters():
        p.requires_grad = True
    assert_unfrozen(pipeline)
    return pipeline


def assert_unfrozen(pipeline) -> None:
    if not all(p.requires_grad for p in pipeline.model.parameters()):
        raise AssertionError("chronos-bolt-small has a frozen parameter during full fine-tuning")


def drop_flat_windows(x_raw: np.ndarray, m_raw: np.ndarray) -> np.ndarray:
    """(n,) bool, True where the observed lookback's std is at least MIN_LOOKBACK_STD mph.

    Bolt's instance norm divides by this std, so a flat window's normalised target explodes.
    A window with no observed step at all is also dropped (std is undefined)."""
    n = x_raw.shape[0]
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        obs = x_raw[i][m_raw[i]]
        keep[i] = obs.size > 0 and obs.std() >= MIN_LOOKBACK_STD
    return keep


def contexts_and_targets(X: np.ndarray, M: np.ndarray, t: np.ndarray, L: int = 12, H: int = 12
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sensor independent series: context (n*N, L) NaN at missing steps, target (n*N, H) mph,
    target_mask (n*N, H) bool. n is len(t), flattened over all N sensors of each window."""
    N = X.shape[1]
    lb = t[:, None] - np.arange(L - 1, -1, -1)[None, :]               # (n, L)
    x_raw, m_raw = X[lb].transpose(0, 2, 1), M[lb].transpose(0, 2, 1).astype(bool)   # (n, N, L)
    context = np.where(m_raw, x_raw, np.nan).astype(np.float32).reshape(-1, L)      # (n*N, L)
    fut = t[:, None] + np.arange(1, H + 1)[None, :]
    y = X[fut].transpose(0, 2, 1).astype(np.float32).reshape(-1, H)                 # (n*N, H)
    my = M[fut].transpose(0, 2, 1).astype(bool).reshape(-1, H)
    return context, y, my


def loss_and_preds(pipeline, context: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor):
    """One forward pass of the unfrozen model: pinball loss (eq. 8) and the de-normalised median forecast."""
    out = pipeline.model(context=context, target=target, target_mask=target_mask)
    y0 = out.quantile_preds[:, MEDIAN_IDX, :]
    return out.loss, y0
