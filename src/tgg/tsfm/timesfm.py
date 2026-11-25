"""Frozen TimesFM 2.5 200M: zero-shot forecast and the embedding it is cached from (Decision 11).

Two separately compiled instances are used: max_context=12 (rounded up to 32) for the bolt-L12-sized
cache, and max_context=288 for the zero-shot-only comparison at L=288 (no cache, guide C8).
"""
import numpy as np
import torch

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PATCH = 32                                                            # input_patch_len
D_RAW = 1280                                                          # model_dims, before PCA
D = 512                                                               # PCA-reduced width stored in the cache
MAX_HORIZON = 128
PCA_FIT_N = 200_000
PCA_SEED = 651


def assert_frozen(model) -> None:
    """Raise unless every parameter of the wrapped TimesFM module has requires_grad=False."""
    if any(p.requires_grad for p in model.model.parameters()):
        raise AssertionError("timesfm-2.5-200m has a trainable parameter")


def _wrap_last_layer(model) -> None:
    """Forward hook on the last transformer layer (`model.model.stacked_xf[-1]`); its output is
    (out, cache) (guide C8). Stashes the last input-patch token of `out` per internal batch chunk,
    since force_flip_invariance=False and horizon <= 128 mean exactly one forward pass per chunk."""
    captured: list = []
    model._tgg_captured = captured

    def hook(_module, _inputs, output):
        out, _cache = output
        captured.append(out[:, -1, :].detach().cpu())

    model.model.stacked_xf[-1].register_forward_hook(hook)


def load(max_context: int):
    """Frozen TimesFM 2.5 200M, compiled for max_context rounded up to a multiple of 32 (Decision 11)."""
    import timesfm

    max_context = -(-max_context // PATCH) * PATCH                   # ceil to a multiple of PATCH
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID, torch_compile=False)
    for p in model.model.parameters():
        p.requires_grad = False
    assert_frozen(model)
    model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=MAX_HORIZON,
                                          per_core_batch_size=1024, force_flip_invariance=False))
    _wrap_last_layer(model)
    return model


def run(model, lookbacks: list[np.ndarray], horizon: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """lookbacks: 1D arrays with NaN at missing steps (Decision 1); forecast() strips leading NaN and
    interpolates the rest itself, and must not be called on an all-NaN row (guide C8; the caller
    writes obs=False for those and skips them here).

    -> (y0 (n, horizon) point forecast, h_raw (n, 1280) last input-patch embedding, before PCA).
    forecast() pads its input list to a multiple of the compiled batch size with dummy rows at the
    end; the hook buffer is trimmed to len(lookbacks) so the cache never misaligns by window.
    """
    model._tgg_captured.clear()
    y0, _quantiles = model.forecast(horizon, list(lookbacks))
    h_raw = torch.cat(model._tgg_captured, dim=0)[: len(lookbacks)].numpy()
    return y0, h_raw


def fit_pca(h_raw_sample: np.ndarray, n_components: int = D, seed: int = PCA_SEED):
    """PCA fitted on a sample of raw (1280-wide) embeddings, drawn from training windows only
    (guide C8; the caller checks that with `tgg.baselines.naive.assert_train_only_fit` on the window
    ids it sampled). Raise unless the components explain at least 95% of the variance (Decision 11).
    n_components is clamped to the sample's rank, for small (e.g. --mode smoke) samples."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, *h_raw_sample.shape)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(h_raw_sample)
    explained = float(pca.explained_variance_ratio_.sum())
    if explained < 0.95:
        raise AssertionError(f"PCA explains only {explained:.3f} of the variance, need >= 0.95")
    return pca


def apply_pca(pca, h_raw: np.ndarray) -> np.ndarray:
    return pca.transform(h_raw).astype(np.float32)
