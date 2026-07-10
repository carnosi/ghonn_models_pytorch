import torch

from ghonn_models_pytorch import GHONU, HONU


def test_honu_vectorizes_outputs_and_supports_chunking() -> None:
    model = HONU(
        in_features=2,
        order=2,
        out_features=3,
        bias=True,
        weight_init_mode="ones",
        monomial_chunk_size=2,
    )
    output = model(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))

    assert output.shape == (2, 3)
    assert torch.equal(output[0], torch.full((3,), 25.0))


def test_zero_order_gate_is_a_honu() -> None:
    model = GHONU(
        in_features=2,
        predictor_order=1,
        gate_order=0,
        weight_init_mode="ones",
    )
    x = torch.tensor([[2.0, 3.0]])

    assert model.gate is None
    assert torch.equal(model(x), model.predictor(x))


def test_gate_can_be_frozen_with_native_pytorch_api() -> None:
    model = GHONU(in_features=2, predictor_order=1, gate_order=1)
    model.gate.requires_grad_(False)

    assert all(not parameter.requires_grad for parameter in model.gate.parameters())
