"""results/*/metrics.json -> reports/<mid|final>/tables/*.tex. Deterministic: same results give byte-identical files.

Usage: python -m scripts.make_tables [--report mid] [--results results] [--ds metr_la]
"""
import argparse
import glob
import json
import os

POP_NAMES = {"all": r"$P_{\text{all}}$", "onset": r"$P_{\text{onset}}$", "steady": r"$P_{\text{steady}}$",
             "graph_hi": r"$P_{\text{graph-hi}}$", "graph_lo": r"$P_{\text{graph-lo}}$"}


def runs(results: str, ds: str) -> list[tuple[str, dict]]:
    """(run id, test metrics) for every run of dataset `ds`, sorted by run id."""
    out = []
    for path in sorted(glob.glob(os.path.join(results, "*", "metrics.json"))):
        run = os.path.basename(os.path.dirname(path))
        if run.endswith(ds):
            out.append((run, json.load(open(path))))
    return out


def main_table(rs: list[tuple[str, dict]]) -> str:
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & \multicolumn{3}{c}{MAE (mph)} & \multicolumn{3}{c}{RMSE (mph)} \\",
            r"Model & 15 min & 30 min & 60 min & 15 min & 30 min & 60 min \\", r"\midrule"]
    for run, m in rs:
        p = m["populations"]["all"]
        vals = [p[f"mae_h{h}"] for h in (3, 6, 12)] + [p[f"rmse_h{h}"] for h in (3, 6, 12)]
        rows.append(run.replace("_", r"\_") + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    return "\n".join(rows + [r"\bottomrule", r"\end{tabular}", ""])


def population_table(rs: list[tuple[str, dict]]) -> str:
    pops = [p for p in POP_NAMES if all(p in m["populations"] and "mae_h12" in m["populations"][p] for _, m in rs)]
    rows = [r"\begin{tabular}{l" + "r" * len(pops) + "}", r"\toprule",
            "60-min MAE (mph) & " + " & ".join(POP_NAMES[p] for p in pops) + r" \\", r"\midrule"]
    for run, m in rs:
        rows.append(run.replace("_", r"\_") + " & " + " & ".join(f"{m['populations'][p]['mae_h12']:.2f}" for p in pops) + r" \\")
    rows.append(r"\midrule")
    rows.append("targets & " + " & ".join(f"{rs[0][1]['populations'][p]['n_targets']:,}".replace(",", "{,}") for p in pops) + r" \\")
    return "\n".join(rows + [r"\bottomrule", r"\end{tabular}", ""])


def run(report: str, results: str, ds: str) -> list[str]:
    rs = runs(results, ds)
    if not rs:
        raise SystemExit(f"no results for {ds} under {results}")
    out = os.path.join("reports", report, "tables")
    os.makedirs(out, exist_ok=True)
    written = []
    for name, text in ((f"main_{ds}.tex", main_table(rs)), (f"populations_{ds}.tex", population_table(rs))):
        with open(os.path.join(out, name), "w") as f:
            f.write(text)
        written.append(os.path.join(out, name))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="mid", choices=["mid", "final"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--ds", default="metr_la")
    print("\n".join(run(**vars(ap.parse_args()))))
