import torch

from ghonn_models_pytorch import (
    GHONN,
    GHONU,
    HONN,
    HONU,
    SAGHONN,
    ConvGhonn,
    ConvGhonn1D,
    GhonuBank,
    HonuBank,
    RevIN,
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


def test_honu_chunking_matches_the_full_monomial_projection() -> None:
    full = HONU(4, 3, 2, monomial_chunk_size=None)
    chunked = HONU(4, 3, 2, monomial_chunk_size=3)
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(5, 4)

    assert torch.allclose(chunked(x), full(x))


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


def test_saghonn_matches_the_trial_0896_profile() -> None:
    model = SAGHONN(
        input_length=64,
        in_features=2,
        out_features=3,
        use_rev_in=False,
    )
    assert model(torch.randn(2, 64, 2)).shape == (2, 3, 1)
    assert model.trend_expert.lookback == model.residual_expert.lookback == 3
    assert model.decomposition.trend_kernel_size == 96
    assert model.decomposition.seasonal_kernel_size == 3
    assert model.seasonal_expert.conv.kernel_size == 9
    assert model.seasonal_expert.conv.n_kernels == 16
    assert model.seasonal_expert.conv.predictor_orders[:2] == (2, 1)
    assert model.seasonal_expert.conv.gate_orders[:3] == (2, 1, 3)
    assert model.dropout.p == 0.06045054705802519
    assert model.decode_dropout.p == 0.11928223192119175
    assert len(model.attention_layers) == 1
    assert model.attention_layers[0].attention.num_heads == 8
    assert model.head[0].out_features == 576


def test_convghonn_and_revin_are_reusable_public_modules() -> None:
    x = torch.randn(2, 4, 2)
    conv = ConvGhonn(2, 3, lookback=2, padding_mode="zeros")
    assert conv(x).shape == (2, 4, 3)
    conv_1d = ConvGhonn1D(4, 2, kernel_size=3, n_kernels=2)
    assert conv_1d(x.transpose(1, 2)).shape == (2, 4, 4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert conv_1d(x.transpose(1, 2)).shape == (2, 4, 4)

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

    assert torch.equal(output.flatten(), torch.tensor([2010.0, 4231.0]))


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


def test_ghonu_bank_supports_autocast() -> None:
    bank = GhonuBank(2, 3, predictor_orders=(1, 2), gate_orders=(1, 2, 3))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = bank(torch.randn(2, 2))

    assert output.shape == (2, 3)
