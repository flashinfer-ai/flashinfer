import pytest
import torch

import flashinfer


def _make_inputs(
    outer_shape: tuple[int, ...],
    hidden_size: int,
    post_ndim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    hc = 4
    x = torch.randn((*outer_shape, hidden_size), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(
        (*outer_shape, hc, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    if post_ndim == 1:
        post_layer_mix = torch.randn(
            (*outer_shape, hc), dtype=torch.float32, device="cuda"
        )
    else:
        post_layer_mix = torch.randn(
            (*outer_shape, hc, 1), dtype=torch.float32, device="cuda"
        )
    comb_res_mix = torch.randn(
        (*outer_shape, hc, hc), dtype=torch.float32, device="cuda"
    )
    return (
        x.contiguous(),
        residual.contiguous(),
        post_layer_mix.contiguous(),
        comb_res_mix.contiguous(),
    )


def _mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    post = post_layer_mix
    if post.ndim == residual.ndim - 1:
        post = post.unsqueeze(-1)
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh", comb_res_mix.float(), residual.float()
    )
    return (x.float().unsqueeze(-2) * post.float() + mixed_residual).bfloat16()


def _assert_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    err = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert err < 0.005, f"output rel-norm diff {err:.4g} exceeds 0.005"
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("outer_shape", "hidden_size", "post_ndim"),
    [
        ((1,), 64, 1),
        ((2, 3), 128, 2),
        ((2,), 4096, 2),
        ((4,), 7168, 1),
    ],
)
def test_mhc_post_matches_reference(
    outer_shape: tuple[int, ...],
    hidden_size: int,
    post_ndim: int,
) -> None:
    x, residual, post_layer_mix, comb_res_mix = _make_inputs(
        outer_shape, hidden_size, post_ndim
    )

    got = flashinfer.mhc_post(x, residual, post_layer_mix, comb_res_mix)
    ref = _mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)

    assert got.shape == residual.shape
    assert got.dtype == residual.dtype
    _assert_close(got, ref)
