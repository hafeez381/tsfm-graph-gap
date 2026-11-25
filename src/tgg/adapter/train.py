"""Step-based adapter training loop: any variant, any cache, any fraction (Decisions 1, 8)."""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from tgg.adapter.data import CacheDataset
from tgg.adapter.model import Adapter, combine_mask, masks
from tgg.seed import seed_everything

DEFAULTS = {"lr": 1e-3, "wd": 0.0, "batch_windows": 32, "eval_every": 200,
            "max_steps": 20000, "patience_evals": 10, "seed": 651,
            "d": 512, "m": 64, "heads": 4, "H": 12, "val_ids": None}


def assert_masked(m: torch.Tensor) -> None:
    """Raise if a loss or metric would be computed over an empty mask (never a silently-wrong 0/0)."""
    if not m.any():
        raise AssertionError("no observed targets in this batch")


def assert_no_encoder_loaded() -> None:
    """The training script reads only the cache; no frozen encoder is ever imported (guide C12)."""
    if any(name in sys.modules for name in ("chronos", "timesfm")):
        raise AssertionError("an encoder module is loaded; the adapter trains on the cache only")


def masked_mae_loss(yhat: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    assert_masked(m)
    return (yhat - y).abs()[m].mean()


def _collate(batch: list[dict]) -> dict:
    def to(key, dtype=torch.float32):
        return torch.as_tensor(np.stack([b[key] for b in batch]), dtype=dtype)

    return {"h": to("h"), "y0": to("y0"), "s": to("s"), "obs": to("obs", torch.bool),
            "y": to("y"), "m": to("m", torch.bool), "t": to("t", torch.int64)}


def validate(model: Adapter, variant_mask: torch.Tensor, loader: DataLoader) -> float:
    """Masked MAE over the whole loader (never a subsample, so fractions stay comparable)."""
    model.eval()
    total_abs, total_n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            mask = combine_mask(variant_mask, batch["obs"])
            yhat, _ = model(batch["h"], batch["y0"], batch["s"], mask)
            m = batch["m"]
            total_abs += (yhat - batch["y"]).abs()[m].sum().item()
            total_n += int(m.sum().item())
    model.train()
    if total_n == 0:
        raise AssertionError("no observed targets in the validation split")
    return total_abs / total_n


def train(cfg: dict) -> dict:
    """cfg (required): cache_dir, variant, X, M, train_ids, nbr; optional: DEFAULTS' keys.

    Returns {"best_state", "best_step", "best_val_mae", "history"}. Validation runs on the whole
    validation split every eval_every steps; training stops at max_steps or patience_evals evals
    without improvement, whichever comes first (Decision 8).
    """
    assert_no_encoder_loaded()
    c = {**DEFAULTS, **cfg}
    seed_everything(c["seed"])

    train_ds = CacheDataset(c["cache_dir"], "train", c["X"], c["M"], c["train_ids"])
    val_ds = CacheDataset(c["cache_dir"], "val", c["X"], c["M"], c["val_ids"])
    train_loader = DataLoader(train_ds, batch_size=c["batch_windows"], shuffle=True,
                               collate_fn=_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=c["batch_windows"], shuffle=False, collate_fn=_collate)

    variant_mask = masks(c["nbr"])[c["variant"]]
    model = Adapter(d=c["d"], m=c["m"], heads=c["heads"], H=c["H"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["wd"])

    init_val_mae = validate(model, variant_mask, val_loader)
    history = [{"step": 0, "train_loss": None, "val_mae": init_val_mae}]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_step, best_val, since_best = 0, init_val_mae, 0

    step, data_iter, loss = 0, iter(train_loader), None
    while step < c["max_steps"] and since_best < c["patience_evals"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        mask = combine_mask(variant_mask, batch["obs"])
        yhat, _ = model(batch["h"], batch["y0"], batch["s"], mask)
        loss = masked_mae_loss(yhat, batch["y"], batch["m"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

        if step % c["eval_every"] == 0:
            val_mae = validate(model, variant_mask, val_loader)
            history.append({"step": step, "train_loss": loss.item(), "val_mae": val_mae})
            if val_mae < best_val:
                best_val, best_step, since_best = val_mae, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1

    return {"best_state": best_state, "best_step": best_step, "best_val_mae": best_val, "history": history}
