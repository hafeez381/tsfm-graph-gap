# Do Time-Series Foundation Models Need a Graph?

A frozen time-series foundation model (Chronos-Bolt Small, TimesFM 2.5) forecasts each road sensor of METR-LA and PeMS-Bay on its own. A small graph attention adapter, trained on the model's cached forecasts and embeddings, adds a correction from the road graph. The code compares this adapter with zero-shot use, Chronos-2, full fine-tuning, PatchTST and Graph WaveNet. Results are added as each stage lands.

## Setup

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv -e ".[stgnn,tsfm,patchtst,dev]"
.venv/bin/python -m pytest -q
```

Versions: torch-spatiotemporal 0.9.5, chronos-forecasting 2.3.2, timesfm 3.0.2, pandas < 3. Every run uses seed 651. Data and training scripts take `--mode smoke` (synthetic data, runs on a laptop) or `--mode full`.

## Compute

Training and caching run on Kaggle T4 GPUs as private kernels; checkpoints and logs are copied to private Hugging Face repositories. The rules are in [docs/compute.md](docs/compute.md).

```bash
kaggle/build_code.sh               # package the committed code as the private dataset <user>/tgg-code
kaggle/push_kernel.sh skeleton     # run kaggle/kernels/skeleton/run.py on a T4
kaggle kernels status <user>/tgg-skeleton
kaggle kernels output <user>/tgg-skeleton -p results/kaggle/skeleton
```

## Reproducing the results

### Walking skeleton

Synthetic data through windows, a fake cache, a zero-initialised stub adapter, predictions and scoring. At step 0 the adapter's test MAE equals the last-value MAE exactly.

```bash
PYTHONPATH=. .venv/bin/python -m scripts.skeleton
```

## Attribution

- Libraries: PyTorch, Torch Spatiotemporal (tsl), chronos-forecasting, timesfm, Hugging Face transformers and huggingface_hub.
- Models and data: Chronos-Bolt Small and Chronos-2 (Amazon), TimesFM 2.5 200M (Google), METR-LA and PeMS-Bay as distributed by tsl.

No code was copied from public implementations. Code was written with LLM assistance.

## References

- Ansari et al., *Chronos: Learning the Language of Time Series*, TMLR 2024.
- Das et al., *A decoder-only foundation model for time-series forecasting*, ICML 2024.
- Wu et al., *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI 2019.
- Veličković et al., *Graph Attention Networks*, ICLR 2018.
- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023.
