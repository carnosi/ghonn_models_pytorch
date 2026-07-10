import torch

from ghonn_models_pytorch import (
    ConvGhonn,
    GhonuBank,
    GHONN,
    GHONU,
    HonuBank,
    HONN,
    HONU,
    RevIN,
    SAGHONN,
)


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


def test_saghonn_supports_all_heads_and_auxiliary_output() -> None:
    x = torch.randn(2, 4, 2)
    for head_type in ("flat", "pool", "query"):
        model = SAGHONN(
            input_length=4,
            in_features=2,
            out_features=3,
            d_model=4,
            n_heads=2,
            head_type=head_type,
            predicted_feature_indices=(1,),
            aux_all_features=True,
        )
        output, auxiliary = model(x)
        assert output.shape == (2, 3, 1)
        assert auxiliary.shape == (2, 3, 2)


def test_saghonn_supports_temporal_windows_and_no_revin() -> None:
    model = SAGHONN(
        input_length=4,
        in_features=2,
        out_features=2,
        d_model=4,
        n_heads=2,
        temporal_lookback=2,
        use_rev_in=False,
    )
    assert model(torch.randn(2, 4, 2)).shape == (2, 2, 1)


def test_convghonn_and_revin_are_reusable_public_modules() -> None:
    x = torch.randn(2, 4, 2)
    conv = ConvGhonn(2, 3, lookback=2, padding_mode="zeros")
    assert conv(x).shape == (2, 4, 3)

    revin = RevIN(2)
    assert torch.allclose(revin(revin(x, "norm"), "denorm"), x, atol=1e-5)


def test_convghonn_preserves_temporal_feature_order() -> None:
    model = ConvGhonn(
        2,
        1,
        lookback=1,
        padding_mode="zeros",
        predictor_orders=1,
        gate_orders=0,
        bias=False,
        weight_init_mode="ones",
    )
    with torch.no_grad():
        model.bank.groups[0].predictor.weight[:, 0] = torch.tensor([1.0, 10.0, 100.0, 1000.0])

    output = model(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))

    assert torch.equal(output.flatten(), torch.tensor([2100.0, 4321.0]))


def test_ghonu_bank_and_ghonn_share_vectorized_outputs() -> None:
    x = torch.randn(2, 3)
    bank = GhonuBank(3, 2, weight_init_mode="ones")
    assert bank(x).shape == (2, 2)

    model = GHONN(
        in_features=3,
        out_features=2,
        layer_size=2,
        predictor_orders=[1],
        gate_orders=[1],
        output_type="linear",
    )
    assert model(x).shape == (2, 2)
    assert model.ghonu.indices == [[0, 1]]
    assert len(model.predictors) == len(model.ghonu.groups)
    assert len(model.gates) == len(model.ghonu.groups)

    honu_bank = HonuBank(3, 2, orders=(1, 2), activations=("identity", "tanh"))
    honn = HONN(3, 2, 2, [1, 2], activations=("identity", "tanh"))
    assert honu_bank(x).shape == honn.honu(x).shape == (2, 2)
