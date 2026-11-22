"""Import every heavy dependency and run one forward pass of each; print versions. Run on the Kaggle T4.

Each check is independent, so one failure does not hide the others. Exits non-zero if any check fails.
"""
import importlib.metadata as md
import sys
import traceback

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
results = {}


def check(name):
    def wrap(fn):
        try:
            results[name] = f"ok: {fn()}"
        except Exception:
            results[name] = "FAIL\n" + traceback.format_exc(limit=3)
        return fn
    return wrap


@check("tsl GraphWaveNetModel(edge_index)")
def _():
    from tsl.nn.models import GraphWaveNetModel
    n, b = 5, 2
    m = GraphWaveNetModel(input_size=1, output_size=1, horizon=12, n_nodes=n).to(dev)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=dev)
    y = m(torch.randn(b, 12, n, 1, device=dev), ei)
    return tuple(y.shape)


@check("chronos-bolt-small")
def _():
    from chronos import BaseChronosPipeline
    p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=dev, torch_dtype=torch.float32)
    q, mean = p.predict_quantiles(torch.randn(3, 288).cumsum(-1), prediction_length=12, quantile_levels=[0.5])
    return tuple(mean.shape)


@check("timesfm-2.5-200m")
def _():
    import numpy as np
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=12, per_core_batch_size=8))
    point, _ = m.forecast(horizon=12, inputs=[np.random.randn(288).cumsum()] * 3)
    return point.shape


@check("transformers PatchTSTForPrediction")
def _():
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    cfg = PatchTSTConfig(num_input_channels=1, context_length=96, patch_length=16, patch_stride=16,
                         prediction_length=12)
    out = PatchTSTForPrediction(cfg).to(dev)(past_values=torch.randn(4, 96, 1, device=dev))
    return tuple(out.prediction_outputs.shape)


print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
for pkg in ("torch", "torch-spatiotemporal", "chronos-forecasting", "timesfm", "transformers", "pandas", "numpy"):
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, "not installed")
for k, v in results.items():
    print(f"[{k}] {v}")
sys.exit(int(any(v.startswith("FAIL") for v in results.values())))
