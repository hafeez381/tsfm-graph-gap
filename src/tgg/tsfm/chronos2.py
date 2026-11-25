"""Frozen Chronos-2: one group per window, so attention spans exactly that window's sensors (Decision 10)."""
import numpy as np
import torch

MODEL_ID = "amazon/chronos-2"
H = 12


def assert_frozen(pipeline) -> None:
    """Raise unless every parameter of the Chronos-2 model has requires_grad=False."""
    if any(p.requires_grad for p in pipeline.model.parameters()):
        raise AssertionError("chronos-2 has a trainable parameter")


def load(device: str = "cuda"):
    """Frozen BaseChronosPipeline for amazon/chronos-2."""
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.float32)
    pipeline.model.eval()
    for p in pipeline.model.parameters():
        p.requires_grad = False
    assert_frozen(pipeline)
    return pipeline


def run(pipeline, windows: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """windows (n, N, L) with NaN at masked steps (Decision 1) -> yhat (n, N, 12), the 0.5 quantile
    (Decision 9). Each window is one multivariate predict() item, so group attention spans exactly
    its N sensors and no others (Decision 10); `batch_size` windows (not sensors) per predict() call."""
    median_idx = list(pipeline.quantiles).index(0.5)
    n, N = windows.shape[0], windows.shape[1]
    out = np.empty((n, N, H), dtype=np.float32)
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(windows[i:i + batch_size].astype(np.float32))
        preds = pipeline.predict(batch, prediction_length=H)   # list of (N, n_quantiles, H) tensors
        for k, p in enumerate(preds):
            out[i + k] = p[:, median_idx, :].to(torch.float32).cpu().numpy()
    return out
