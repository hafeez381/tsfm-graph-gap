"""Frozen Chronos-Bolt Small: zero-shot forecast and the embedding it is cached from (Decisions 5, 9)."""
import torch

MODEL_ID = "amazon/chronos-bolt-small"
D = 512                                                               # encoder hidden width
MEDIAN_IDX = 4                                                        # index of the 0.5 quantile among Bolt's 9


def assert_frozen(pipeline) -> None:
    """Raise unless every parameter of the Bolt model has requires_grad=False."""
    if any(p.requires_grad for p in pipeline.model.parameters()):
        raise AssertionError("chronos-bolt-small has a trainable parameter")


def _wrap_encode(pipeline) -> None:
    """Monkey-patch model.encode to stash (hidden, loc_scale, input_embeds, attention_mask) for the
    last call, so predict()'s own encoder pass is captured instead of a second one being run. A plain
    forward hook cannot see this, because encode takes the mask as a keyword argument (Decision 5)."""
    original = pipeline.model.encode
    captured: dict = {}
    pipeline._tgg_captured = captured

    def capturing_encode(context, mask=None):
        hidden, loc_scale, input_embeds, attention_mask = original(context, mask=mask)
        captured["hidden"], captured["loc_scale"] = hidden, loc_scale
        captured["input_embeds"], captured["attention_mask"] = input_embeds, attention_mask
        return hidden, loc_scale, input_embeds, attention_mask

    pipeline.model.encode = capturing_encode


def load(device: str = "cuda"):
    """Frozen BaseChronosPipeline for amazon/chronos-bolt-small, eval mode, encoder wrapped (Decision 5)."""
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.float32)
    pipeline.model.eval()
    for p in pipeline.model.parameters():
        p.requires_grad = False
    assert_frozen(pipeline)
    _wrap_encode(pipeline)
    return pipeline


def _last_patch_token(hidden: torch.Tensor, pipeline) -> torch.Tensor:
    """The last patch token's hidden state, excluding [REG] if the model appends one (guide C7, Ablation 2)."""
    cfg = getattr(pipeline.model, "chronos_config", None)
    use_reg = getattr(cfg, "use_reg_token", False) if cfg is not None else False
    idx = hidden.shape[1] - (2 if use_reg and hidden.shape[1] > 1 else 1)
    return hidden[:, idx, :]


def run(pipeline, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """contexts (B, L) with NaN where missing (Decision 1) -> (y0 (B,12), h (B,512), loc (B,), sd (B,), h_last (B,512)).

    y0 is the 0.5 quantile of pipeline.predict (Decision 9). h is the mean of the captured hidden
    states over tokens whose attention mask is 1 (observed patches plus [REG]), pooled from the same
    predict() call so the encoder runs once per window (Decision 5). h_last is the last patch token
    alone (guide C7), cheap to compute alongside h; callers decide whether to store it.
    """
    pipeline._tgg_captured.clear()
    quantiles = pipeline.predict(contexts, prediction_length=12)      # (B, 9, 12), already in mph, cpu
    y0 = quantiles[:, MEDIAN_IDX, :].to(torch.float32)
    hidden = pipeline._tgg_captured["hidden"].to(torch.float32)       # (B, tokens, 512)
    mask = pipeline._tgg_captured["attention_mask"].to(torch.float32)  # (B, tokens)
    h = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    h_last = _last_patch_token(hidden, pipeline)
    loc, scale = pipeline._tgg_captured["loc_scale"]
    loc, sd = loc.squeeze(-1).to(torch.float32), scale.squeeze(-1).to(torch.float32)
    return y0.cpu(), h.cpu(), loc.cpu(), sd.cpu(), h_last.cpu()
