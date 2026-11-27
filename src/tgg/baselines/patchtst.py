"""PatchTST: a trained Transformer with no graph and no pretraining (Decision 14).

Channel independence: each sensor is its own univariate series of shape (B*N, L, 1). Scaling is
`"std"` (RevIN), fit and reversed per window inside the model, so inputs and outputs are both in mph.
"""
import torch

H, L = 12, 12

DEFAULTS = {"patch_length": 4, "patch_stride": 2, "d_model": 64, "num_hidden_layers": 3,
            "num_attention_heads": 4, "ffn_dim": 128, "attention_dropout": 0.2, "ff_dropout": 0.2,
            "head_dropout": 0.2, "scaling": "std"}


def build(cfg: dict | None = None) -> torch.nn.Module:
    from transformers import PatchTSTConfig, PatchTSTForPrediction

    c = {**DEFAULTS, **(cfg or {})}
    config = PatchTSTConfig(
        num_input_channels=1, context_length=L, prediction_length=H, patch_length=c["patch_length"],
        patch_stride=c["patch_stride"], d_model=c["d_model"], num_hidden_layers=c["num_hidden_layers"],
        num_attention_heads=c["num_attention_heads"], ffn_dim=c["ffn_dim"],
        attention_dropout=c["attention_dropout"], ff_dropout=c["ff_dropout"],
        head_dropout=c["head_dropout"], scaling=c["scaling"], loss="mse")
    model = PatchTSTForPrediction(config)
    assert model.model.patchifier.num_patches == 5, model.model.patchifier.num_patches
    return model


def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x: (B, N, L) mph, forward-filled. Returns (B, N, H) mph."""
    B, N, Lw = x.shape
    out = model(past_values=x.reshape(B * N, Lw, 1)).prediction_outputs  # (B*N, H, 1)
    return out.squeeze(-1).reshape(B, N, -1)
