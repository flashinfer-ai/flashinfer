import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


RMS_EPS = 1e-6
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT_VALUE = 1.0
SINKHORN_REPEAT = 20


def _require_sm80_for_bf16() -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 8:
        pytest.skip("mHC BF16 tests require an SM80+ GPU.")


def _sinkhorn_normalize_ref(
    x: torch.Tensor,
    repeat: int = SINKHORN_REPEAT,
    eps: float = MHC_SINKHORN_EPS,
) -> torch.Tensor:
    x = x.softmax(dim=-1) + eps
    x = x / (x.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x / (x.sum(dim=-2, keepdim=True) + eps)
    return x


def _mhc_pre_big_fuse_ref(
    dot_mix: torch.Tensor,
    sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dot_mix.ndim == residual.ndim:
        dot_mix = dot_mix.sum(dim=0)
        sqrsum = sqrsum.sum(dim=0)

    hc = residual.shape[-2]
    rstd = torch.rsqrt(sqrsum.float().unsqueeze(-1) / float(k) + RMS_EPS)
    mixes = dot_mix.float() * rstd

    pre_logits = mixes[..., :hc] * mhc_scale[0] + mhc_base[:hc]
    post_logits = mixes[..., hc : 2 * hc] * mhc_scale[1] + mhc_base[hc : 2 * hc]
    comb_logits = mixes[..., 2 * hc :] * mhc_scale[2] + mhc_base[2 * hc :]

    pre_mix = torch.sigmoid(pre_logits).unsqueeze(-1) + MHC_PRE_EPS
    post_mix = (torch.sigmoid(post_logits) * MHC_POST_MULT_VALUE).unsqueeze(-1)
    comb_mix = comb_logits.view(*residual.shape[:-2], hc, hc)
    comb_mix = _sinkhorn_normalize_ref(comb_mix)
    layer_input = (pre_mix * residual.float()).sum(dim=-2).bfloat16()
    return post_mix, comb_mix, layer_input


def _mhc_pre_big_fuse_with_prenorm_ref(
    dot_mix: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dot_mix.ndim == residual.ndim:
        assert dot_mix.shape[0] == 1
        dot_mix = dot_mix.squeeze(0)
    sqrsum = residual.flatten(-2).float().square().sum(dim=-1)
    return _mhc_pre_big_fuse_ref(
        dot_mix,
        sqrsum,
        residual,
        mhc_scale,
        mhc_base,
        residual.shape[-2] * residual.shape[-1],
    )


def _assert_close_tuple(
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for got_tensor, ref_tensor in zip(got, ref, strict=True):
        err = (
            got_tensor.float() - ref_tensor.float()
        ).norm() / ref_tensor.float().norm().clamp_min(1e-6)
        assert err < 0.006, f"output rel-norm diff {err:.4g} exceeds 0.006"
    torch.testing.assert_close(got[0], ref[0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got[1], ref[1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got[2], ref[2], atol=1e-2, rtol=1e-2)


def _make_common_inputs(
    outer_shape: tuple[int, ...],
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    residual = (
        torch.randn((*outer_shape, 4, hidden_size), dtype=torch.float32, device="cuda")
        * 0.01
    ).bfloat16()
    mhc_scale = torch.randn((3,), dtype=torch.float32, device="cuda") * 0.1
    mhc_base = torch.randn((24,), dtype=torch.float32, device="cuda") * 0.1
    return residual.contiguous(), mhc_scale.contiguous(), mhc_base.contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_splits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    ("outer_shape", "hidden_size"),
    [
        ((2,), 64),
        ((1,), 4096),
        ((2, 3), 128),
    ],
)
def test_mhc_pre_big_fuse_matches_reference(
    num_splits: int,
    outer_shape: tuple[int, ...],
    hidden_size: int,
) -> None:
    _require_sm80_for_bf16()
    residual, mhc_scale, mhc_base = _make_common_inputs(outer_shape, hidden_size)
    k = 4 * hidden_size
    if num_splits == 1:
        dot_mix = (
            torch.randn((*outer_shape, 24), dtype=torch.float32, device="cuda") * 0.01
        )
        sqrsum = torch.rand(outer_shape, dtype=torch.float32, device="cuda") * float(k)
    else:
        dot_mix = (
            torch.randn(
                (num_splits, *outer_shape, 24), dtype=torch.float32, device="cuda"
            )
            * 0.01
        )
        sqrsum = torch.rand(
            (num_splits, *outer_shape), dtype=torch.float32, device="cuda"
        ) * (float(k) / num_splits)

    got = flashinfer.mhc.mhc_pre_big_fuse(
        dot_mix.contiguous(),
        sqrsum.contiguous(),
        residual,
        mhc_scale,
        mhc_base,
        k,
        rms_eps=RMS_EPS,
        mhc_pre_eps=MHC_PRE_EPS,
        mhc_sinkhorn_eps=MHC_SINKHORN_EPS,
        mhc_post_mult_value=MHC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
        num_splits=num_splits,
    )
    ref = _mhc_pre_big_fuse_ref(dot_mix, sqrsum, residual, mhc_scale, mhc_base, k)

    assert got[0].shape == (*outer_shape, 4, 1)
    assert got[1].shape == (*outer_shape, 4, 4)
    assert got[2].shape == (*outer_shape, hidden_size)
    _assert_close_tuple(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("outer_shape", "hidden_size", "leading_split_dim"),
    [
        ((2,), 64, False),
        ((1,), 4096, True),
        ((4,), 7168, False),
    ],
)
def test_mhc_pre_big_fuse_with_prenorm_matches_reference(
    outer_shape: tuple[int, ...],
    hidden_size: int,
    leading_split_dim: bool,
) -> None:
    _require_sm80_for_bf16()
    residual, mhc_scale, mhc_base = _make_common_inputs(outer_shape, hidden_size)
    dot_mix = torch.randn((*outer_shape, 24), dtype=torch.float32, device="cuda") * 0.01
    if leading_split_dim:
        dot_mix = dot_mix.unsqueeze(0)

    got = flashinfer.mhc.mhc_pre_big_fuse_with_prenorm(
        dot_mix.contiguous(),
        residual,
        mhc_scale,
        mhc_base,
        rms_eps=RMS_EPS,
        mhc_pre_eps=MHC_PRE_EPS,
        mhc_sinkhorn_eps=MHC_SINKHORN_EPS,
        mhc_post_mult_value=MHC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
    )
    ref = _mhc_pre_big_fuse_with_prenorm_ref(dot_mix, residual, mhc_scale, mhc_base)

    assert got[0].shape == (*outer_shape, 4, 1)
    assert got[1].shape == (*outer_shape, 4, 4)
    assert got[2].shape == (*outer_shape, hidden_size)
    _assert_close_tuple(got, ref)
