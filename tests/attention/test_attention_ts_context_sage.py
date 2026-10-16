"""Sage attention on the PrimTS context kernel: numerics, plan contract and prepared scales."""

from __future__ import annotations


import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper,
    SageAttentionConfig,
    SageAttentionParams,
)
from flashinfer.utils import get_compute_capability
from tests.attention.test_attention_ts_sage import (
    _dequantize,
    _fold,
    _ieee_float32_references,  # noqa: F401
    _random_scales,
    _reference_sage_scales,
)

_FP8 = torch.float8_e4m3fn
_HEAD_DIM = 128
# The context kernel folds one 128-token K tile per online-softmax step and
# keeps its exponent anchor while a tile raises the row maximum by at most
# this many binades (its lazy correction threshold).
_KV_TILE = 128
_ANCHOR_HEADROOM_LOG2 = 8.0
_REQUIRES_CONTEXT_SAGE_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS context Sage attention needs an SM100-class GPU",
)


def _plan(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    sage: SageAttentionConfig | None,
    **overrides,
) -> BatchPrefillTSWrapper:
    wrapper = BatchPrefillTSWrapper()
    options = dict(
        device="cuda:0",
        batch_size=batch_size,
        max_seq_len_q=seq_len_q,
        max_kv_len=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_HEAD_DIM,
        q_dtype=_FP8,
        k_dtype=_FP8,
        out_dtype=torch.bfloat16,
        sage=sage,
    )
    options.update(overrides)
    wrapper.plan(**options)
    return wrapper


def _random_params(
    config: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> SageAttentionParams:
    v_rows = (batch_size, num_kv_heads, _HEAD_DIM)
    return SageAttentionParams(
        q_scale=_random_scales(
            num_qo_heads, batch_size, seq_len_q, config.q_block_size, 0.1, 0.4
        ),
        k_scale=_random_scales(
            num_kv_heads, batch_size, seq_len_kv, config.k_block_size, 0.5, 2.0
        ),
        v_scale=torch.empty(v_rows, device="cuda").uniform_(0.25, 1.0),
        v_mean=torch.randn(v_rows, device="cuda") if config.v_mean else None,
    )


@torch.no_grad()
def _reference(q, k, v, params: SageAttentionParams, *, config, sm_scale: float):
    """FP32 attention on the dequantized inputs with the kernel's tile model.

    ``O = sfV * softmax(sfQ sfK (Q8 K8^T) sm_scale) V8 + v_mean`` per batch
    entry and KV head, folded one 128-token K tile at a time.
    """
    batch_size, seq_len_kv, num_kv_heads, _ = k.shape
    num_qo_heads = q.shape[2]
    group = num_qo_heads // num_kv_heads
    q_real = _dequantize(q, params.q_scale, config.q_block_size)
    k_real = _dequantize(k, params.k_scale, config.k_block_size)
    output = torch.zeros(q.shape, device="cuda")
    for batch_idx in range(batch_size):
        for kv_head in range(num_kv_heads):
            heads = slice(kv_head * group, (kv_head + 1) * group)
            logits = (
                torch.einsum(
                    "tgd,sd->tgs",
                    q_real[batch_idx, :, heads],
                    k_real[batch_idx, :, kv_head],
                )
                * sm_scale
            )
            state = None
            for begin in range(0, seq_len_kv, _KV_TILE):
                tile = slice(begin, begin + _KV_TILE)
                # The kernel's P scale is one at this headroom.
                state = _fold(
                    state,
                    logits[..., tile],
                    v[batch_idx, tile, kv_head].float(),
                    headroom=_ANCHOR_HEADROOM_LOG2,
                    p_scale=1.0,
                )
            _, total, acc = state
            result = acc / total.unsqueeze(-1) * params.v_scale[batch_idx, kv_head]
            if params.v_mean is not None:
                result = result + params.v_mean[batch_idx, kv_head]
            output[batch_idx, :, heads] = result
    return output


@_REQUIRES_CONTEXT_SAGE_GPU
@pytest.mark.parametrize(
    "batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, "
    "q_block_size, k_block_size, v_mean",
    [
        (1, 1024, 1024, 8, 8, 1, 16, False),
        (2, 300, 1004, 8, 4, 2, 16, True),
        (1, 256, 4096, 4, 4, 128, 64, True),
    ],
)
@torch.no_grad()
def test_context_sage_matches_dequantized_reference(
    batch_size,
    seq_len_q,
    seq_len_kv,
    num_qo_heads,
    num_kv_heads,
    q_block_size,
    k_block_size,
    v_mean,
) -> None:
    """Sage context attention matches FP32 attention on the dequantized inputs."""
    torch.manual_seed(0)
    config = SageAttentionConfig(
        q_block_size=q_block_size, k_block_size=k_block_size, v_mean=v_mean
    )
    shape = dict(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
    )
    sm_scale = _HEAD_DIM**-0.5
    wrapper = _plan(sage=config, sm_scale=sm_scale, **shape)
    params = _random_params(config, **shape)
    q = torch.randn((batch_size, seq_len_q, num_qo_heads, _HEAD_DIM), device="cuda").to(
        _FP8
    )
    k = torch.randn(
        (batch_size, seq_len_kv, num_kv_heads, _HEAD_DIM), device="cuda"
    ).to(_FP8)
    v = torch.randn_like(k.float()).to(_FP8)
    actual = wrapper.run(q, k, v, sage=params)
    torch.cuda.synchronize()
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    expected = _reference(q, k, v, params, config=config, sm_scale=sm_scale)
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=2e-3)


@_REQUIRES_CONTEXT_SAGE_GPU
@pytest.mark.parametrize(
    "batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, q_block_size, v_mean",
    [
        (1, 300, 1004, 4, 4, 1, False),
        (2, 257, 640, 8, 4, 2, True),
    ],
)
def test_context_sage_prepared_scales_match_reference(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, q_block_size, v_mean
) -> None:
    torch.manual_seed(0)
    config = SageAttentionConfig(
        q_block_size=q_block_size, k_block_size=16, v_mean=v_mean
    )
    shape = dict(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
    )
    wrapper = _plan(sage=config, **shape)
    params = _random_params(config, **shape)
    q = torch.randn((batch_size, seq_len_q, num_qo_heads, _HEAD_DIM), device="cuda").to(
        _FP8
    )
    k = torch.randn(
        (batch_size, seq_len_kv, num_kv_heads, _HEAD_DIM), device="cuda"
    ).to(_FP8)
    v = torch.randn_like(k.float()).to(_FP8)
    out = wrapper.run(q, k, v, sage=params)
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    state = wrapper._plan_state.sage
    assert state is not None
    reference = _reference_sage_scales(
        state.layout, params, batch_size=batch_size, num_kv_heads=num_kv_heads
    )
    torch.testing.assert_close(state.scales, reference, rtol=0, atol=0)


@_REQUIRES_CONTEXT_SAGE_GPU
def test_context_sage_plan_rejects_unsupported_recipes() -> None:
    shape = dict(
        batch_size=1, seq_len_q=256, seq_len_kv=256, num_qo_heads=2, num_kv_heads=2
    )
    config = SageAttentionConfig()
    with pytest.raises(NotImplementedError, match="float16 or bfloat16 output"):
        _plan(sage=config, out_dtype=_FP8, **shape)
    with pytest.raises(NotImplementedError, match="float8_e4m3fn Q, K and V"):
        _plan(sage=config, q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, **shape)
    with pytest.raises(NotImplementedError, match="dense, fixed-length"):
        _plan(sage=config, mask_type="causal", **shape)
    with pytest.raises(NotImplementedError, match="one K scale per 16 tokens"):
        _plan(sage=SageAttentionConfig(k_block_size=4), **shape)
    with pytest.raises(TypeError, match="SageAttentionConfig"):
        _plan(sage=SageAttentionParams, **shape)


@_REQUIRES_CONTEXT_SAGE_GPU
def test_context_sage_run_requires_matching_params() -> None:
    shape = dict(
        batch_size=1, seq_len_q=256, seq_len_kv=256, num_qo_heads=2, num_kv_heads=2
    )
    config = SageAttentionConfig()
    q = torch.randn((1, 256, 2, _HEAD_DIM), device="cuda").to(_FP8)
    k = torch.randn_like(q.float()).to(_FP8)
    v = torch.randn_like(q.float()).to(_FP8)
    sage_plan = _plan(sage=config, **shape)
    with pytest.raises(ValueError, match="required by a Sage plan"):
        sage_plan.run(q, k, v)
    params = _random_params(config, **shape)
    wrong = SageAttentionParams(
        q_scale=params.q_scale,
        k_scale=params.k_scale,
        v_scale=params.v_scale[:, :1],
    )
    with pytest.raises(ValueError, match="v_scale must have shape"):
        sage_plan.run(q, k, v, sage=wrong)
    plain_plan = _plan(sage=None, **shape)
    with pytest.raises(ValueError, match="rejected by a plan without Sage"):
        plain_plan.run(q, k, v, sage=params)
