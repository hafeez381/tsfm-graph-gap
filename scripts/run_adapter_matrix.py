"""Run several (variant, fraction) adapter training runs on one cache in one process (guide C16).

Usage: python -m scripts.run_adapter_matrix --cache bolt-L12 --fractions 1.0,0.25,0.05 \
           --variants gat,mlp,full --mode full
"""
import argparse
import json

from scripts.train_adapter import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, dest="cache_name")
    ap.add_argument("--variants", default="gat,mlp,full")
    ap.add_argument("--fractions", default="1.0")
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache-root", default="cache", dest="cache_root")
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=651)
    a = ap.parse_args()

    out = []
    for variant in a.variants.split(","):
        for fraction in (float(x) for x in a.fractions.split(",")):
            print(f"=== {a.cache_name} {variant} f{fraction} ===", flush=True)
            res = run(cache_name=a.cache_name, variant=variant, fraction=fraction, mode=a.mode, ds=a.ds,
                      data=a.data, cache_root=a.cache_root, results=a.results, seed=a.seed)
            print(json.dumps(res, default=str), flush=True)
            out.append(res["run_id"])
    print("done:", out, flush=True)
