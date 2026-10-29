"""Sage attention on the PrimTS context kernel: numerics, plan contract and prepared scales."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper,
    SageAttentionConfig,
    SageAttentionParams,
)
from flashinfer.utils import get_compute_capability
from tests.attention.sage_test_utils import (
    REQUIRES_INT8_QK_GPU,
    dequantize,
    fold,
    ieee_float32_references,  # noqa: F401
    random_qk,
    random_qk_scales,
    reference_sage_scales,
)

_FP8 = torch.float8_e4m3fn
_INT8 = torch.int8
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


@_REQUIRES_CONTEXT_SAGE_GPU
def test_context_sage_cold_compile(tmp_path: Path) -> None:
    """Compile SMEM Sage scales without prior paged imports or cached kernels."""
    target = (
        f"{Path(__file__).resolve()}::test_context_sage_matches_dequantized_reference"
        "[1-512-1000-4-4-1-1-False-float8_e4m3fn]"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--maxfail=1", target],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(
            os.environ,
            PYTEST_ADDOPTS="",
            FLASHINFER_WORKSPACE_BASE=str(tmp_path),
            CUTE_DSL_CACHE_DIR=str(tmp_path / "cute_dsl"),
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, result.stdout + result.stderr


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


def _qk_dtype_options(qk_dtype: torch.dtype) -> dict:
    """Plan keywords of a Q/K dtype; INT8 Q/K keep E4M3 V."""
    if qk_dtype == _INT8:
        return dict(q_dtype=_INT8, k_dtype=_INT8, v_dtype=_FP8)
    return dict(q_dtype=qk_dtype, k_dtype=qk_dtype)


def _dtype_id(value):
    """Id the dtype column of a parametrized case; other columns keep pytest's."""
    return str(value).removeprefix("torch.") if isinstance(value, torch.dtype) else None


def _random_params(
    config: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    qk_dtype: torch.dtype = _FP8,
) -> SageAttentionParams:
    """Random positive scales for ``qk_dtype`` Q/K and E4M3 V."""
    q_scale, k_scale = random_qk_scales(
        num_qo_heads,
        num_kv_heads,
        batch_size,
        seq_len_q,
        seq_len_kv,
        config.q_block_size,
        config.k_block_size,
        qk_dtype,
    )
    v_rows = (batch_size, num_kv_heads, _HEAD_DIM)
    return SageAttentionParams(
        q_scale=q_scale,
        k_scale=k_scale,
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
    q_real = dequantize(q, params.q_scale, config.q_block_size)
    k_real = dequantize(k, params.k_scale, config.k_block_size)
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
                state = fold(
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


# The three K block sizes cover every K transport and score scaling the
# context kernel selects.
_INT8_K_BLOCK_SIZES = (16, 4, 1)


@_REQUIRES_CONTEXT_SAGE_GPU
@pytest.mark.parametrize(
    "batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, "
    "q_block_size, k_block_size, v_mean, qk_dtype",
    [
        (1, 1024, 1024, 8, 8, 1, 16, False, _FP8),
        (2, 300, 1004, 8, 4, 2, 16, True, _FP8),
        (1, 256, 4096, 4, 4, 128, 64, True, _FP8),
        (2, 300, 1004, 8, 4, 2, 4, True, _FP8),
        (1, 512, 1000, 4, 4, 1, 1, False, _FP8),
        # 256-token Q scale blocks span two Q tiles.
        (2, 600, 1004, 8, 4, 256, 16, True, _FP8),
        *(
            pytest.param(*case, _INT8, marks=REQUIRES_INT8_QK_GPU)
            for case in (
                (1, 1024, 1024, 8, 8, 1, 16, False),
                (2, 300, 1004, 8, 4, 2, 4, True),
                (1, 512, 1000, 4, 4, 1, 1, False),
            )
        ),
    ],
    ids=_dtype_id,
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
    qk_dtype,
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
    wrapper = _plan(
        sage=config, sm_scale=sm_scale, **_qk_dtype_options(qk_dtype), **shape
    )
    params = _random_params(config, qk_dtype=qk_dtype, **shape)
    q = random_qk((batch_size, seq_len_q, num_qo_heads, _HEAD_DIM), qk_dtype)
    k = random_qk((batch_size, seq_len_kv, num_kv_heads, _HEAD_DIM), qk_dtype)
    v = torch.randn(k.shape, device="cuda").to(_FP8)
    actual = wrapper.run(q, k, v, sage=params)
    torch.cuda.synchronize()
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    expected = _reference(q, k, v, params, config=config, sm_scale=sm_scale)
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=2e-3)


@_REQUIRES_CONTEXT_SAGE_GPU
@REQUIRES_INT8_QK_GPU
@pytest.mark.parametrize("k_block_size", _INT8_K_BLOCK_SIZES)
@torch.no_grad()
def test_context_int8_extreme_scores_stay_in_the_bias_binade(k_block_size) -> None:
    """INT8 scores of +-2**21 keep their exact biased float form in every softmax mode."""
    torch.manual_seed(0)
    config = SageAttentionConfig(q_block_size=1, k_block_size=k_block_size)
    shape = dict(
        batch_size=1, seq_len_q=256, seq_len_kv=1000, num_qo_heads=2, num_kv_heads=2
    )
    sm_scale = _HEAD_DIM**-0.5
    wrapper = _plan(sage=config, sm_scale=sm_scale, **_qk_dtype_options(_INT8), **shape)
    params = _random_params(config, qk_dtype=_INT8, **shape)
    q = random_qk((1, 256, 2, _HEAD_DIM), _INT8)
    k = random_qk((1, 1000, 2, _HEAD_DIM), _INT8)
    # Query token 5 against key tokens 300 and 700 gives the extreme dot
    # products +2**21 and -128 * 127 * 128; their scales keep the logits
    # near +-2 so both stay visible in the softmax.
    q[:, 5] = -128
    k[:, 300] = -128
    k[:, 700] = 127
    params.q_scale[:, 5] = 2.0**-14
    params.k_scale[:, 300 // k_block_size] = 2.0**-6
    params.k_scale[:, 700 // k_block_size] = 2.0**-6
    v = torch.randn(k.shape, device="cuda").to(_FP8)
    actual = wrapper.run(q, k, v, sage=params)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    expected = _reference(q, k, v, params, config=config, sm_scale=sm_scale)
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=2e-3)


@_REQUIRES_CONTEXT_SAGE_GPU
def test_context_int8_qk_plans_only_on_sm100(monkeypatch: pytest.MonkeyPatch) -> None:
    """INT8 Q/K need the INT8 tcgen05 MMA, which only SM100 has."""
    shape = dict(
        batch_size=1, seq_len_q=256, seq_len_kv=256, num_qo_heads=2, num_kv_heads=2
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 3))
    with pytest.raises(NotImplementedError, match="SM100a"):
        _plan(sage=SageAttentionConfig(), **_qk_dtype_options(_INT8), **shape)


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
    # A Sage plan without an output dtype writes BF16.
    wrapper = _plan(sage=config, out_dtype=None, **shape)
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
    reference = reference_sage_scales(
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
    with pytest.raises(NotImplementedError, match="float8_e4m3fn or int8 Q/K"):
        _plan(sage=config, q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, **shape)
    with pytest.raises(NotImplementedError, match="require a Sage recipe"):
        _plan(sage=None, **_qk_dtype_options(_INT8), **shape)
    with pytest.raises(NotImplementedError, match="float8_e4m3fn V"):
        _plan(sage=config, q_dtype=_INT8, k_dtype=_INT8, v_dtype=_INT8, **shape)
    with pytest.raises(NotImplementedError, match="dense, fixed-length"):
        _plan(sage=config, mask_type="causal", **shape)
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
