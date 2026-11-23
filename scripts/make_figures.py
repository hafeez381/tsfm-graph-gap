"""results/*/day_sums.npz -> reports/<mid|final>/figures/*.pdf.

Implemented: MAE by horizon (1-12) per model, one panel per population.
shortcut: learning curves (C23) and robustness curves (C19) are added when their run folders exist.
Usage: python -m scripts.make_figures [--report mid] [--results results] [--ds metr_la]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tgg.eval import stats  # noqa: E402


def mae_by_horizon(results: str, ds: str, out: str) -> str:
    paths = sorted(p for p in glob.glob(os.path.join(results, "*", "day_sums.npz")) if os.path.basename(os.path.dirname(p)).endswith(ds))
    data = {os.path.basename(os.path.dirname(p)): stats.load(p) for p in paths}
    pops = [p for p in ("all", "onset", "steady") if all(p in d["populations"] for d in data.values())]
    fig, axes = plt.subplots(1, len(pops), figsize=(3.2 * len(pops), 2.6), sharey=True, squeeze=False)
    minutes = 5 * np.arange(1, 13)
    for ax, pop, tag in zip(axes[0], pops, "abc"):
        for run, d in data.items():
            k = d["populations"].index(pop)
            ax.plot(minutes, d["abs"][k].sum(0) / d["cnt"][k].sum(0), marker="o", ms=2.5, label=run)
        ax.set_title(f"({tag}) {pop}", fontsize=9)
        ax.set_xlabel("horizon (min)")
    axes[0][0].set_ylabel("MAE (mph)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=min(4, len(data)), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = os.path.join(out, f"mae_by_horizon_{ds}.pdf")
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="mid", choices=["mid", "final"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--ds", default="metr_la")
    a = ap.parse_args()
    out = os.path.join("reports", a.report, "figures")
    os.makedirs(out, exist_ok=True)
    print(mae_by_horizon(a.results, a.ds, out))
