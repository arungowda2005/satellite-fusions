import pytest, torch
from src.models import build_model

def _batch(B=8):
    return torch.randn(B, 12, 13), torch.ones(B, 12, 2, dtype=torch.bool)

@pytest.mark.parametrize("name", ["s2only", "concat", "ma"])
def test_output_shape(name):
    x, m = _batch()
    assert build_model(name)(x, m).shape == (8, 10)

def test_ma_all_missing_is_finite():
    x, m = torch.zeros(4, 12, 13), torch.zeros(4, 12, 2, dtype=torch.bool)
    assert torch.isfinite(build_model("ma")(x, m)).all()

def test_ma_output_depends_on_mask():
    model = build_model("ma").eval()
    x, m = _batch(4)
    m2 = m.clone(); m2[:, :, 1] = False
    assert not torch.allclose(model(x, m), model(x, m2))

def test_baselines_ignore_mask():
    for name in ["s2only", "concat"]:
        model = build_model(name).eval()
        x, m = _batch(4)
        assert torch.allclose(model(x, m), model(x, torch.zeros_like(m)))