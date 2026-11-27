import torch

from tgg.baselines import patchtst


def test_build_reports_5_patches_for_a_12_step_context():
    model = patchtst.build()
    assert model.model.patchifier.num_patches == 5


def test_build_has_no_dropout_field_typo():
    model = patchtst.build()
    assert model.config.attention_dropout == 0.2
    assert model.config.ff_dropout == 0.2
    assert model.config.head_dropout == 0.2


def test_predict_output_shape_is_B_N_H():
    model = patchtst.build()
    x = torch.rand(3, 5, patchtst.L) * 10 + 55
    yhat = patchtst.predict(model, x)
    assert yhat.shape == (3, 5, patchtst.H)


def test_predict_reverses_instance_norm_to_mph_scale():
    """A near-constant 60 mph window should forecast near 60 mph, not a z-scored residual."""
    model = patchtst.build()
    model.eval()
    x = torch.full((2, 4, patchtst.L), 60.0) + torch.randn(2, 4, patchtst.L) * 0.05
    with torch.no_grad():
        yhat = patchtst.predict(model, x)
    assert (yhat.mean() - 60.0).abs() < 5.0
