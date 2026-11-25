import pytest
import torch

from tgg.tsfm import bolt


class _FakeModel(torch.nn.Module):
    """Mimics ChronosBoltModelForForecasting.encode's signature and return shape, 2 tokens, one masked."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))

    def encode(self, context, mask=None):
        B, tokens = context.shape[0], 2
        hidden = torch.ones(B, tokens, bolt.D) * torch.tensor([1.0, 2.0]).view(1, tokens, 1)
        attention_mask = torch.tensor([[1.0, 0.0]] * B)           # second token masked out
        loc_scale = (torch.zeros(B, 1), torch.ones(B, 1))
        return hidden, loc_scale, torch.zeros(B, tokens, bolt.D), attention_mask


class _FakePipeline:
    def __init__(self):
        self.model = _FakeModel()

    def predict(self, contexts, prediction_length=12):
        self.model.encode(contexts)                                # the same single encoder pass
        return torch.zeros(contexts.shape[0], 9, prediction_length)


def test_wrap_encode_pools_only_unmasked_tokens():
    pipeline = _FakePipeline()
    bolt._wrap_encode(pipeline)
    y0, h, loc, sd = bolt.run(pipeline, torch.zeros(3, 12))
    assert y0.shape == (3, 12) and torch.allclose(y0, torch.zeros(3, 12))
    assert h.shape == (3, bolt.D) and torch.allclose(h, torch.ones(3, bolt.D))   # token 2 (value 2) is masked out
    assert torch.allclose(loc, torch.zeros(3)) and torch.allclose(sd, torch.ones(3))


def test_assert_frozen():
    pipeline = _FakePipeline()
    with pytest.raises(AssertionError):
        bolt.assert_frozen(pipeline)
    for p in pipeline.model.parameters():
        p.requires_grad = False
    bolt.assert_frozen(pipeline)
