import pytest
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

    # 1 + a + b + a² + ab + b² for each independent output.
    torch.testing.assert_close(output, torch.tensor([[25.0] * 3, [71.0] * 3]))


def test_honu_chunking_matches_the_full_monomial_projection() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        full = HONU(4, 3, 2, monomial_chunk_size=None)
        chunked = HONU(4, 3, 2, monomial_chunk_size=3)
    chunked.load_state_dict(full.state_dict())
    x = torch.linspace(-0.8, 0.8, 20).reshape(5, 4).requires_grad_()
    chunked_x = x.detach().clone().requires_grad_()

    expected = full(x)
    actual = chunked(chunked_x)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(chunked_x.grad, x.grad)
    for full_parameter, chunked_parameter in zip(
        full.parameters(), chunked.parameters(), strict=True
    ):
        torch.testing.assert_close(chunked_parameter.grad, full_parameter.grad)


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
    model = GHONU(
        in_features=2,
        predictor_order=1,
        gate_order=1,
        gate_activation="identity",
        weight_init_mode="ones",
    )
    model.gate.requires_grad_(False)
    x = torch.tensor([[2.0, 3.0]], requires_grad=True)

    output = model(x)
    torch.testing.assert_close(output, torch.tensor([[36.0]]))
    output.sum().backward()
    assert all(parameter.grad is None for parameter in model.gate.parameters())
    assert all(parameter.grad is not None for parameter in model.predictor.parameters())
    # Frozen gate weights still contribute to the derivative with respect to x.
    torch.testing.assert_close(x.grad, torch.tensor([[12.0, 12.0]]))


@pytest.mark.parametrize("use_rev_in", [False, True])
def test_saghonn_zero_head_repeats_last_selected_features(use_rev_in: bool) -> None:
    model = SAGHONN(
        input_length=4,
        in_features=3,
        out_features=3,
        d_model=4,
        n_heads=2,
        ffn_dim=8,
        decoder_hidden_dim=8,
        temporal_lookback=1,
        conv_kernel_size=3,
        conv_n_kernels=2,
        trend_kernel_size=3,
        predicted_feature_indices=(2, 0),
        use_rev_in=use_rev_in,
    )
    model.eval()
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    expected = x[:, -1, [2, 0]].unsqueeze(1).expand(-1, 3, -1)
    torch.testing.assert_close(model(x), expected)


def test_convolutions_support_public_input_layouts_and_autocast() -> None:
    x = torch.randn(2, 4, 2)
    conv = ConvGhonn(2, 3, lookback=2, padding_mode="zeros")
    assert conv(x).shape == (2, 4, 3)
    conv_1d = ConvGhonn1D(4, 2, kernel_size=3, n_kernels=2)
    assert conv_1d(x.transpose(1, 2)).shape == (2, 4, 4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert conv_1d(x.transpose(1, 2)).shape == (2, 4, 4)


def test_revin_normalization_round_trip() -> None:
    x = torch.tensor([[[1.0, 20.0], [3.0, 40.0], [5.0, 60.0]]])
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


def test_banks_and_networks_preserve_mixed_order_output_positions() -> None:
    x = torch.tensor([[2.0, 3.0]])
    bank = GhonuBank(
        2, 3, predictor_orders=(1, 2), gate_activations="identity", weight_init_mode="ones"
    )
    ghonn = GHONN(
        in_features=2,
        out_features=3,
        layer_size=3,
        predictor_orders=[1, 2],
        gate_orders=[1],
        gate_activations="identity",
        output_type="raw",
        weight_init_mode="ones",
    )
    # Linear basis sums to 6, quadratic basis to 25; every gate sums to 6.
    expected = torch.tensor([[36.0, 150.0, 36.0]])
    torch.testing.assert_close(bank(x), expected)
    torch.testing.assert_close(ghonn(x), expected)

    honu_bank = HonuBank(2, 3, orders=(1, 2), weight_init_mode="ones")
    honn = HONN(2, 3, 3, [1, 2], output_type="raw", weight_init_mode="ones")
    expected = torch.tensor([[6.0, 25.0, 6.0]])
    torch.testing.assert_close(honu_bank(x), expected)
    torch.testing.assert_close(honn(x), expected)


def test_ghonu_bank_supports_autocast() -> None:
    bank = GhonuBank(2, 3, predictor_orders=(1, 2), gate_orders=(1, 2, 3))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = bank(torch.randn(2, 2))

    assert output.shape == (2, 3)
