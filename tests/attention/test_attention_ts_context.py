# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public-contract and correctness coverage for PrimTS context attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from typing import Optional, Sequence

import pytest
import torch

import flashinfer.attention.prims_ts.context as _context_api
from flashinfer.attention.prims_ts import (
    BatchPrefillTSWrapper,
    batch_prefill_with_kv_cache,
)


_REQUIRES_CONTEXT_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS context attention requires SM100 or SM103",
)

_HEAD_DIM = 128
_FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class _ContextCase:
    """One fixed or packed-ragged context-attention problem."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    qo_indptr: Optional[torch.Tensor]
    kv_indptr: Optional[torch.Tensor]
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    mask_type: str
    window_left: int
    sm_scale: float
    output_scale: float
    output_dtype: torch.dtype

    @property
    def packed(self) -> bool:
        return self.qo_indptr is not None


def _cumulative(lengths: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return tuple(offsets)


def _make_context_case(
    *,
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    qkv_dtype: torch.dtype,
    packed: bool,
    mask_type: str,
    window_left: int = -1,
    output_dtype: Optional[torch.dtype] = None,
    output_scale: float = 0.75,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> _ContextCase:
    """Create deterministic compact BSHD or packed THD input storage."""

    q_lengths = tuple(int(length) for length in q_lengths)
    k_lengths = tuple(int(length) for length in k_lengths)
    if not q_lengths or len(q_lengths) != len(k_lengths):
        raise ValueError("Q and KV lengths must describe the same non-empty batch")
    if min(q_lengths) <= 0 or min(k_lengths) <= 0:
        raise ValueError("sequence lengths must be positive")
    if not packed and (len(set(q_lengths)) != 1 or len(set(k_lengths)) != 1):
        raise ValueError("fixed storage requires uniform Q and KV lengths")

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    input_scale = 0.125 if qkv_dtype == _FP8 else 0.2
    if packed:
        q_shape = (sum(q_lengths), num_qo_heads, _HEAD_DIM)
        kv_shape = (sum(k_lengths), num_kv_heads, _HEAD_DIM)
    else:
        q_shape = (len(q_lengths), q_lengths[0], num_qo_heads, _HEAD_DIM)
        kv_shape = (len(k_lengths), k_lengths[0], num_kv_heads, _HEAD_DIM)

    q = (
        input_scale
        * torch.randn(q_shape, generator=generator, device=device, dtype=torch.float32)
    ).to(qkv_dtype)
    k = (
        input_scale
        * torch.randn(kv_shape, generator=generator, device=device, dtype=torch.float32)
    ).to(qkv_dtype)
    v = (
        input_scale
        * torch.randn(kv_shape, generator=generator, device=device, dtype=torch.float32)
    ).to(qkv_dtype)
    if packed:
        qo_indptr = torch.tensor(
            _cumulative(q_lengths), dtype=torch.int32, device=device
        )
        kv_indptr = torch.tensor(
            _cumulative(k_lengths), dtype=torch.int32, device=device
        )
    else:
        qo_indptr = None
        kv_indptr = None

    return _ContextCase(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=1.0 / math.sqrt(_HEAD_DIM),
        output_scale=output_scale,
        output_dtype=qkv_dtype if output_dtype is None else output_dtype,
    )


def _request_slice(
    tensor: torch.Tensor,
    lengths: tuple[int, ...],
    batch_idx: int,
    *,
    packed: bool,
) -> torch.Tensor:
    if not packed:
        return tensor[batch_idx]
    begin = sum(lengths[:batch_idx])
    return tensor[begin : begin + lengths[batch_idx]]


def _visible_kv_bounds(
    *,
    q_length: int,
    k_length: int,
    query_idx: int,
    mask_type: str,
    window_left: int,
) -> tuple[int, int]:
    """Return FlashInfer's bottom-right visible interval ``[begin, end)``."""

    if mask_type == "dense":
        end = k_length
    elif mask_type == "causal":
        end = k_length - q_length + query_idx + 1
    else:
        raise ValueError("mask_type must be 'dense' or 'causal'")
    if end <= 0:
        raise ValueError("bottom-right causal attention requires Q length <= KV")
    begin = 0 if window_left < 0 else max(0, end - window_left - 1)
    return begin, end


@torch.no_grad()
def _context_reference(case: _ContextCase) -> torch.Tensor:
    """Independent FP32 MHA/GQA oracle for fixed and packed context inputs."""

    outputs = []
    for batch_idx, (q_length, k_length) in enumerate(
        zip(case.q_lengths, case.k_lengths, strict=True)
    ):
        q = _request_slice(case.q, case.q_lengths, batch_idx, packed=case.packed)
        k = _request_slice(case.k, case.k_lengths, batch_idx, packed=case.packed)
        v = _request_slice(case.v, case.k_lengths, batch_idx, packed=case.packed)
        q = q.float()
        k = k.float()
        v = v.float()
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError("Q head count must be divisible by KV head count")
        head_ratio = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(head_ratio, dim=1)
        v = v.repeat_interleave(head_ratio, dim=1)

        request_output = []
        for query_idx in range(q_length):
            begin, end = _visible_kv_bounds(
                q_length=q_length,
                k_length=k_length,
                query_idx=query_idx,
                mask_type=case.mask_type,
                window_left=case.window_left,
            )
            scores = torch.einsum("hd,khd->hk", q[query_idx], k[begin:end])
            probabilities = torch.softmax(scores * case.sm_scale, dim=-1)
            request_output.append(
                torch.einsum("hk,khd->hd", probabilities, v[begin:end])
                * case.output_scale
            )
        outputs.append(torch.stack(request_output))
    return torch.cat(outputs) if case.packed else torch.stack(outputs)


def _assert_context_correct(actual: torch.Tensor, case: _ContextCase) -> None:
    expected = _context_reference(case)
    assert actual.shape == case.q.shape
    assert actual.dtype == case.output_dtype
    assert torch.isfinite(actual.float()).all()
    # Select by the least precise input/output type. FP8 includes the kernel's
    # E4M3 probability quantization as well as optional E4M3 output rounding.
    if case.q.dtype == _FP8 or case.output_dtype == _FP8:
        rtol, atol, max_relative_l2 = 5e-2, 1.3e-1, 1e-1
    elif case.q.dtype == torch.bfloat16 or case.output_dtype == torch.bfloat16:
        rtol, atol, max_relative_l2 = 2e-2, 1e-2, 2e-2
    else:
        rtol, atol, max_relative_l2 = 1e-2, 2e-3, 1e-2
    torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-6)
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected) / denominator
    assert float(relative_l2) <= max_relative_l2


def _plan_wrapper(wrapper: BatchPrefillTSWrapper, case: _ContextCase) -> None:
    wrapper.plan(
        case.q,
        case.k,
        case.v,
        qo_indptr=case.qo_indptr,
        kv_indptr=case.kv_indptr,
        mask_type=case.mask_type,
        window_left=case.window_left,
        sm_scale=case.sm_scale,
        output_scale=case.output_scale,
        out_dtype=case.output_dtype,
    )


def _run_one_shot(case: _ContextCase, *, out: Optional[torch.Tensor] = None):
    return batch_prefill_with_kv_cache(
        case.q,
        case.k,
        case.v,
        qo_indptr=case.qo_indptr,
        kv_indptr=case.kv_indptr,
        mask_type=case.mask_type,
        window_left=case.window_left,
        sm_scale=case.sm_scale,
        output_scale=case.output_scale,
        out_dtype=case.output_dtype,
        out=out,
    )


# ---------------------------------------------------------------------------
# CPU-only oracle and public API contract
# ---------------------------------------------------------------------------


def test_attention_ts_context_fixed_oracle_is_bottom_right_causal():
    case = _make_context_case(
        q_lengths=(2,),
        k_lengths=(4,),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float32,
        packed=False,
        mask_type="causal",
        output_scale=1.0,
        device="cpu",
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, :, 0, 0] = torch.arange(1, 5, dtype=torch.float32)

    causal = _context_reference(case)
    dense = _context_reference(replace(case, mask_type="dense"))
    windowed = _context_reference(replace(case, window_left=1))
    torch.testing.assert_close(causal[0, :, 0, 0], torch.tensor((2.0, 2.5)))
    torch.testing.assert_close(dense[0, :, 0, 0], torch.tensor((2.5, 2.5)))
    # Sq=2, Sk=4 shifts causal Q by two positions. A one-token left window
    # therefore sees K[1:3] then K[2:4]. The old unshifted bound produced
    # (2.0, 2.5), identical to the unwindowed causal result above.
    torch.testing.assert_close(windowed[0, :, 0, 0], torch.tensor((2.5, 3.5)))
    assert not torch.equal(windowed, causal)
    assert not torch.equal(causal[:, 0], dense[:, 0])
    assert _visible_kv_bounds(
        q_length=2,
        k_length=4,
        query_idx=0,
        mask_type="causal",
        window_left=1,
    ) == (1, 3)


def test_attention_ts_context_packed_oracle_applies_left_window_per_row():
    case = _make_context_case(
        q_lengths=(3, 4),
        k_lengths=(3, 4),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float32,
        packed=True,
        mask_type="causal",
        window_left=1,
        output_scale=1.0,
        device="cpu",
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[:3, 0, 0] = torch.tensor((1.0, 2.0, 3.0))
    case.v[3:, 0, 0] = torch.tensor((11.0, 12.0, 13.0, 14.0))

    output = _context_reference(case)
    expected = torch.tensor((1.0, 1.5, 2.5, 11.0, 11.5, 12.5, 13.5))
    torch.testing.assert_close(output[:, 0, 0], expected)
    torch.testing.assert_close(output[:, 1, 0], expected)
    assert torch.count_nonzero(output[..., 1:]) == 0


def test_attention_ts_context_public_surface_is_semantic():
    surfaces = (
        BatchPrefillTSWrapper.plan,
        BatchPrefillTSWrapper.run,
        batch_prefill_with_kv_cache,
    )
    forbidden = (
        "autotuner",
        "clc",
        "config",
        "persistent",
        "profile",
        "schedule",
        "single_kv",
        "stage",
        "tile",
        "warp",
    )
    violations = [
        parameter
        for surface in surfaces
        for parameter in inspect.signature(surface).parameters
        if any(part in parameter for part in forbidden)
    ]
    assert violations == []

    plan_parameters = inspect.signature(BatchPrefillTSWrapper.plan).parameters
    run_parameters = inspect.signature(BatchPrefillTSWrapper.run).parameters
    one_shot_parameters = inspect.signature(batch_prefill_with_kv_cache).parameters
    assert tuple(plan_parameters) == (
        "self",
        "q",
        "k",
        "v",
        "qo_indptr",
        "kv_indptr",
        "mask_type",
        "window_left",
        "sm_scale",
        "output_scale",
        "out_dtype",
    )
    assert tuple(run_parameters) == ("self", "q", "k", "v", "out")
    assert tuple(one_shot_parameters) == (
        "q",
        "k",
        "v",
        "qo_indptr",
        "kv_indptr",
        "mask_type",
        "window_left",
        "sm_scale",
        "output_scale",
        "out_dtype",
        "out",
    )
    for parameters in (plan_parameters, one_shot_parameters):
        assert parameters["qo_indptr"].default is None
        assert parameters["kv_indptr"].default is None
        assert parameters["mask_type"].default == "dense"
        assert parameters["window_left"].default == -1
        assert parameters["sm_scale"].default is None
        assert parameters["output_scale"].default == 1.0
        assert parameters["out_dtype"].default is None
    assert one_shot_parameters["out"].default is None


def test_attention_ts_context_run_requires_plan():
    wrapper = BatchPrefillTSWrapper()
    placeholder = torch.empty(0)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(placeholder, placeholder, placeholder)


def test_attention_ts_context_geometry_and_overlap_helpers_are_exact():
    assert _context_api._derive_q_offset_geometry((900, 1), (900, 1000), "causal") == (
        True,
        999,
    )
    assert _context_api._derive_q_offset_geometry((900, 1), (900, 1000), "dense") == (
        False,
        0,
    )

    storage = torch.empty(32, dtype=torch.uint8)
    lhs = storage[2:18]
    partial = storage[10:26]
    disjoint = storage[18:]
    distinct = torch.empty_like(lhs)
    assert _context_api._tensors_overlap(lhs, lhs)
    assert _context_api._tensors_overlap(lhs, partial)
    assert _context_api._tensors_overlap(partial, lhs)
    assert not _context_api._tensors_overlap(lhs, disjoint)
    assert not _context_api._tensors_overlap(lhs, distinct)

    byte_storage = torch.empty(64, dtype=torch.uint8)
    metadata = byte_storage[:12].view(torch.int32)
    aliased_out = byte_storage.view(torch.float16)
    with pytest.raises(ValueError, match="out must not overlap qo_indptr storage"):
        _context_api._validate_out_does_not_overlap_inputs(
            aliased_out,
            ("qo_indptr", metadata),
        )


# ---------------------------------------------------------------------------
# CPU-only private task-graph and cd442fd8 policy coverage
# ---------------------------------------------------------------------------


def _build_static_task_graph(
    *,
    is_persistent: bool,
    is_clc_dynamic: bool,
    is_causal: bool,
    head_paired: bool,
    window_size_left: int,
    num_kv_tiles: int,
    q_offset: int = 0,
    causal_single_kv_tile: bool = False,
    has_varlen: bool = False,
):
    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import (
        FmhaTs,
        build_fmha_task_manager,
    )

    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=cutlass.Float16,
        out_dtype=cutlass.Float16,
        d=_HEAD_DIM,
        is_persistent=is_persistent,
        is_causal=is_causal,
        is_clc_dynamic=is_clc_dynamic,
        head_paired=head_paired,
        window_size_left=window_size_left,
        h_r=8 if head_paired else 1,
        causal_single_kv_tile=causal_single_kv_tile,
    )
    fmha.cfg.has_varlen = has_varlen
    fmha.cfg.has_q_offset = q_offset != 0
    task_manager, *_ = build_fmha_task_manager(
        cfg=fmha.cfg,
        tile_sched_params=None,
        tma_q_desc=None,
        tma_k_desc=None,
        tma_v_desc=None,
        tma_o_desc=None,
        cum_seqlen_q=None,
        cum_seqlen_k=None,
        num_kv_tiles=num_kv_tiles,
        q_offset=q_offset,
        is_persistent=is_persistent,
        is_clc_dynamic=is_clc_dynamic,
        exhaustive_deadlock_race_check=False,
    )
    return fmha, task_manager


@pytest.mark.parametrize(
    (
        "is_persistent",
        "is_clc_dynamic",
        "is_causal",
        "head_paired",
        "window_size_left",
        "q_offset",
    ),
    (
        pytest.param(False, False, False, False, 0, 0, id="nonpersistent_dense"),
        pytest.param(True, False, True, False, 0, 128, id="persistent_causal"),
        pytest.param(True, True, False, False, 0, 0, id="clc_dense"),
        pytest.param(True, True, True, True, 64, 0, id="clc_window_gqa"),
    ),
)
def test_attention_ts_context_private_schedule_modes_build_cpu_task_graph(
    is_persistent: bool,
    is_clc_dynamic: bool,
    is_causal: bool,
    head_paired: bool,
    window_size_left: int,
    q_offset: int,
):
    _, task_manager = _build_static_task_graph(
        is_persistent=is_persistent,
        is_clc_dynamic=is_clc_dynamic,
        is_causal=is_causal,
        head_paired=head_paired,
        window_size_left=window_size_left,
        num_kv_tiles=2,
        q_offset=q_offset,
    )
    tasks = {task.name: task for task in task_manager.tasks}
    expected = {
        "Softmax0Task",
        "Softmax1Task",
        "CorrectionTask",
        "MmaTask",
        "LoadTask",
        "EpilogueTask",
        "SchedulerTask" if is_clc_dynamic else "PaddingTask",
    }
    assert set(tasks) == expected
    assert tasks["Softmax0Task"].warp_idx == 0
    assert tasks["Softmax1Task"].warp_idx == 4
    assert tasks["CorrectionTask"].warp_idx == 8
    assert tasks["MmaTask"].warp_idx == 12
    assert tasks["LoadTask"].warp_idx == 13
    assert tasks["EpilogueTask"].warp_idx == 14


def test_attention_ts_context_bottom_right_window_origins_include_q_offset():
    from flashinfer.attention.prims_ts.kernels.fmha_context.helpers import (
        bottom_right_window_left_bound,
        bottom_right_window_tile_start,
    )

    # Fixed Sq=128, Sk=384 has q_offset=256. For window_left=64, Q row zero
    # begins at K=192, so the first intersecting 128-token K/V tile is tile 1.
    assert bottom_right_window_left_bound(0, 256, 64) == 192
    assert (
        bottom_right_window_tile_start(
            seq_coord=0,
            q_tile_m=128,
            kv_tile_n=128,
            q_offset=256,
            window_size_left=64,
        )
        == 1
    )

    _, task_manager = _build_static_task_graph(
        is_persistent=True,
        is_clc_dynamic=True,
        is_causal=True,
        head_paired=True,
        window_size_left=64,
        num_kv_tiles=3,
        q_offset=256,
    )
    tasks = {task.name: task for task in task_manager.tasks}
    # N covers K tiles [1, 3); N-1 loop tasks execute one iteration before
    # their tail. The old unshifted tile origin produced N=3 and N-1=2.
    for name in ("LoadTask", "EpilogueTask"):
        assert tasks[name].get_domain((0, 0, 0)) == 2
    for name in (
        "MmaTask",
        "CorrectionTask",
        "Softmax0Task",
        "Softmax1Task",
    ):
        assert tasks[name].get_domain((0, 0, 0)) == 1
    gmem_qkv = next(
        resource
        for resource in tasks["LoadTask"].src_resources
        if resource.name == "gmem_qkv"
    )
    assert gmem_qkv.q_offset_default == 256


def test_attention_ts_context_head_paired_window_tail_masks_both_bounds():
    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_resources import (
        TmemSPResource,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_context.helpers import (
        bottom_right_window_left_bound,
    )

    # The tail can be the only score tile. For Sq=33, Sk=65 and window_left=31,
    # Q row zero may see K[1:33], not K[0:33]. A right-only causal tail mask
    # incorrectly retains K=0.
    index_q = 0
    q_offset = 32
    window_size_left = 31
    left = bottom_right_window_left_bound(index_q, q_offset, window_size_left)
    right = index_q + q_offset
    assert (left, right) == (1, 32)
    assert list(range(max(0, left), min(128, right + 1)))[0] == 1

    # Keep a structural guard at the consumer boundary: this path is separate
    # from left_masked_row_max(), so its two-sided tail behavior is otherwise
    # invisible to the static task-domain tests.
    source = inspect.getsource(TmemSPResource.right_masked_row_max)
    assert "bottom_right_window_left_bound" in source
    assert "mask = mask & left_mask" in source


@pytest.mark.parametrize(
    ("window_left", "has_varlen", "has_q_offset", "kv_tile_n", "expected"),
    (
        pytest.param(126, False, False, 128, True, id="fixed_narrow"),
        pytest.param(127, False, False, 128, False, id="fixed_boundary"),
        pytest.param(128, False, False, 128, False, id="fixed_perf_fast_path"),
        pytest.param(128, True, False, 128, True, id="packed_conservative"),
        pytest.param(128, False, True, 128, True, id="offset_conservative"),
        pytest.param(128, False, False, 64, True, id="asymmetric_conservative"),
    ),
)
def test_attention_ts_context_window_tail_left_mask_specialization(
    window_left: int,
    has_varlen: bool,
    has_q_offset: bool,
    kv_tile_n: int,
    expected: bool,
):
    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_resources import (
        FmhaConfig,
        TmemSPResource,
    )

    cfg = FmhaConfig(
        window_size_left=window_left,
        has_varlen=has_varlen,
        has_q_offset=has_q_offset,
        kv_tile_n=kv_tile_n,
    )
    resource = object.__new__(TmemSPResource)
    resource.cfg = cfg
    assert resource.needs_window_tail_left_mask is expected


def test_attention_ts_context_packed_window_domain_covers_misaligned_offset():
    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import (
        FmhaTs,
        _select_fmha_domain_policy,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_context.helpers import (
        bottom_right_window_max_tiles,
        bottom_right_window_tile_start,
    )

    q_tile_m = 128
    kv_tile_n = 128
    window_size_left = 64
    max_tiles = bottom_right_window_max_tiles(
        q_tile_m=q_tile_m,
        kv_tile_n=kv_tile_n,
        window_size_left=window_size_left,
    )
    assert max_tiles == 3

    def actual_span(q_offset: int) -> int:
        first = bottom_right_window_tile_start(
            seq_coord=0,
            q_tile_m=q_tile_m,
            kv_tile_n=kv_tile_n,
            q_offset=q_offset,
            window_size_left=window_size_left,
        )
        last = (q_offset + q_tile_m - 1) // kv_tile_n
        return last - first + 1

    # Offset 64 aligns the left edge and needs two tiles. Offset 191 starts at
    # K=127, so the same 192-token interval intersects tiles 0, 1, and 2.
    # Offset zero is clipped at sequence start and leaves two right-OOB tiles
    # in the common domain, which the packed loop must right-mask.
    assert actual_span(64) == 2
    assert actual_span(191) == 3
    assert actual_span(0) == 1

    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=cutlass.Float16,
        out_dtype=cutlass.Float16,
        d=_HEAD_DIM,
        is_persistent=True,
        is_causal=True,
        is_clc_dynamic=True,
        head_paired=True,
        window_size_left=window_size_left,
        h_r=8,
    )
    fmha.cfg.has_varlen = True
    policy = _select_fmha_domain_policy(
        fmha.cfg,
        num_kv_tiles=8,
        q_offset=999,
    )
    assert policy.domain_n_kwargs["packed_window"] is True
    assert policy.domain_n_kwargs["offset"] == 0
    assert policy.domain_n_minus_1_kwargs["packed_window"] is True
    assert policy.domain_n_minus_1_kwargs["offset"] == 1


def test_attention_ts_context_packed_domain_uses_host_max_q_offset():
    q_lengths = (900, 1)
    k_lengths = (900, 1000)
    shape_offset = max(k_lengths) - max(q_lengths)
    q_offsets = tuple(
        k_length - q_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    )
    max_q_offset = max(q_offsets)
    assert shape_offset == 100
    assert q_offsets == (0, 999)
    assert max_q_offset == 999
    assert _context_api._derive_q_offset_geometry(q_lengths, k_lengths, "causal") == (
        True,
        999,
    )

    # FmhaTs must receive the per-request maximum. Use the Python-static fixed
    # graph here so get_domain() needs no MLIR context; the causal domain math
    # is identical, while the bounded GPU matrix exercises real packed input.
    # The old max-shape offset covered only three tiles, while request 1 needs
    # all eight.
    _, task_manager = _build_static_task_graph(
        is_persistent=True,
        is_clc_dynamic=True,
        is_causal=True,
        head_paired=False,
        window_size_left=0,
        num_kv_tiles=8,
        q_offset=max_q_offset,
        has_varlen=False,
    )
    tasks = {task.name: task for task in task_manager.tasks}
    assert tasks["LoadTask"].get_domain((0, 0, 0)) == 8
    assert tasks["MmaTask"].get_domain((0, 0, 0)) == 7
    assert tasks["Softmax1Task"].get_domain((0, 0, 0)) == 7
    tmem_resources = {
        resource.name: resource
        for task in task_manager.tasks
        for resource in task.src_resources
        if resource.name in ("tmem_sp0", "tmem_sp1")
    }
    assert tmem_resources["tmem_sp0"].uses_query_paired_q_offset_loop_mask
    assert tmem_resources["tmem_sp1"].uses_query_paired_q_offset_loop_mask


@pytest.mark.parametrize(
    ("input_name", "output_name", "expected_qkv_iters", "expected_o_iters"),
    (
        pytest.param("fp16", "fp8", 2, 1, id="fp16_to_fp8"),
        pytest.param("fp8", "bf16", 1, 2, id="fp8_to_bf16"),
        pytest.param("bf16", "bf16", 2, 2, id="bf16_to_bf16"),
    ),
)
def test_attention_ts_context_head_paired_output_tma_uses_output_width(
    input_name: str,
    output_name: str,
    expected_qkv_iters: int,
    expected_o_iters: int,
):
    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import FmhaTs

    dtype = {
        "fp16": cutlass.Float16,
        "bf16": cutlass.BFloat16,
        "fp8": cutlass.Float8E4M3FN,
    }
    cfg = FmhaTs(
        d=_HEAD_DIM,
        is_persistent=True,
        is_causal=True,
        is_clc_dynamic=True,
        head_paired=True,
        window_size_left=31,
        h_r=4,
        in_dtype=dtype[input_name],
        out_dtype=dtype[output_name],
    ).cfg
    assert cfg.tma_copy_qkv_iters == expected_qkv_iters
    assert cfg.tma_copy_o_iters == expected_o_iters
    assert cfg.tma_copy_o_granu_inner == _HEAD_DIM // expected_o_iters
    assert cfg.tma_copy_o_granu_inner * cfg.o_dtype.width // 8 == 128


@pytest.mark.parametrize(
    ("is_causal", "single_kv_tile", "dtype_name", "expected"),
    (
        pytest.param(False, False, "fp16", (184, 88, 56), id="dense_fp16"),
        pytest.param(False, False, "bf16", (184, 88, 56), id="dense_bf16"),
        pytest.param(False, False, "fp8", (184, 88, 56), id="dense_fp8"),
        pytest.param(True, True, "fp16", (192, 96, 32), id="single_fp16"),
        pytest.param(True, True, "bf16", (192, 96, 32), id="single_bf16"),
        pytest.param(True, True, "fp8", (184, 88, 56), id="single_fp8"),
        pytest.param(True, False, "fp16", (192, 96, 32), id="causal_fp16"),
        pytest.param(True, False, "bf16", (192, 96, 32), id="causal_bf16"),
        pytest.param(True, False, "fp8", (184, 88, 56), id="causal_fp8"),
    ),
)
def test_attention_ts_context_register_budget_matches_cd442fd8(
    is_causal: bool,
    single_kv_tile: bool,
    dtype_name: str,
    expected: tuple[int, int, int],
):
    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import FmhaTs

    dtype = {
        "fp16": cutlass.Float16,
        "bf16": cutlass.BFloat16,
        "fp8": cutlass.Float8E4M3FN,
    }[dtype_name]
    cfg = FmhaTs(
        d=_HEAD_DIM,
        is_causal=is_causal,
        causal_single_kv_tile=single_kv_tile,
        in_dtype=dtype,
        out_dtype=dtype,
    ).cfg
    actual = (cfg.num_regs_softmax, cfg.num_regs_correction, cfg.num_regs_other)
    assert actual == expected
    assert 8 * actual[0] + 4 * actual[1] + 4 * actual[2] == 2048


def test_attention_ts_context_single_kv_tile_uses_static_zero_loop_domains():
    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import (
        FmhaTs,
        _select_fmha_domain_policy,
    )

    fmha = FmhaTs(d=_HEAD_DIM, is_causal=True, causal_single_kv_tile=True)
    policy = _select_fmha_domain_policy(
        fmha.cfg,
        num_kv_tiles=1,
        q_offset=0,
    )
    assert not fmha.cfg.skip_causal_invalid_peer0
    assert policy.domain_n_kwargs == {"domain": 1}
    assert policy.domain_n_minus_1_kwargs == {"domain": 0}
    assert policy.softmax0_domain_kwargs == {"domain": 0}
    assert policy.softmax1_domain_kwargs == {"domain": 0}

    _, task_manager = _build_static_task_graph(
        is_persistent=True,
        is_clc_dynamic=True,
        is_causal=True,
        head_paired=False,
        window_size_left=0,
        num_kv_tiles=1,
        causal_single_kv_tile=True,
    )
    domains = {task.name: task.domain for task in task_manager.tasks}
    assert domains["LoadTask"] == 1
    assert domains["MmaTask"] == 0
    assert domains["Softmax0Task"] == 0
    assert domains["Softmax1Task"] == 0
    assert domains["CorrectionTask"] == 0


def test_attention_ts_context_single_kv_tile_rejects_invalid_private_use():
    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import FmhaTs

    with pytest.raises(ValueError, match="requires query-paired causal"):
        FmhaTs(d=_HEAD_DIM, causal_single_kv_tile=True)
    with pytest.raises(ValueError, match="requires query-paired causal"):
        FmhaTs(
            d=_HEAD_DIM,
            is_causal=True,
            head_paired=True,
            h_r=8,
            causal_single_kv_tile=True,
        )
    with pytest.raises(ValueError, match="exactly one compile-time K/V tile"):
        _build_static_task_graph(
            is_persistent=True,
            is_clc_dynamic=True,
            is_causal=True,
            head_paired=False,
            window_size_left=0,
            num_kv_tiles=2,
            causal_single_kv_tile=True,
        )


def test_attention_ts_context_d256_rejects_oversized_tmem_layout():
    import cutlass

    from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import FmhaTs

    with pytest.raises(ValueError, match="requires 768 TMEM columns"):
        FmhaTs(
            d=256,
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            in_dtype=cutlass.Float8E4M3FN,
            out_dtype=cutlass.Float8E4M3FN,
        )


# ---------------------------------------------------------------------------
# Public validation and bounded SM100/SM103 correctness matrix
# ---------------------------------------------------------------------------


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_public_rejections():
    base = _make_context_case(
        q_lengths=(8,),
        k_lengths=(8,),
        num_qo_heads=4,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="dense",
        device="cuda",
        seed=2026071401,
    )

    with pytest.raises(NotImplementedError, match="head_dim=128"):
        BatchPrefillTSWrapper().plan(
            torch.empty((1, 8, 4, 64), dtype=base.q.dtype, device="cuda"),
            torch.empty((1, 8, 2, 64), dtype=base.k.dtype, device="cuda"),
            torch.empty((1, 8, 2, 64), dtype=base.v.dtype, device="cuda"),
        )
    with pytest.raises(NotImplementedError, match="torch.float16"):
        BatchPrefillTSWrapper().plan(base.q.float(), base.k.float(), base.v.float())
    with pytest.raises(NotImplementedError, match="same dtype"):
        BatchPrefillTSWrapper().plan(base.q, base.k.half(), base.v)
    with pytest.raises(ValueError, match="same shape"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v[:, :-1])
    with pytest.raises(ValueError, match="divisible"):
        BatchPrefillTSWrapper().plan(
            torch.empty((1, 8, 6, 128), dtype=base.q.dtype, device="cuda"),
            torch.empty((1, 8, 4, 128), dtype=base.k.dtype, device="cuda"),
            torch.empty((1, 8, 4, 128), dtype=base.v.dtype, device="cuda"),
        )
    with pytest.raises(ValueError, match="exactly 'dense' or 'causal'"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v, mask_type="custom")
    with pytest.raises(ValueError, match="window_left=0"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v, window_left=0)
    with pytest.raises(ValueError, match="requires mask_type='causal'"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v, window_left=31)
    with pytest.raises(NotImplementedError, match="grouped-query attention"):
        mha_q = torch.empty((1, 8, 2, 128), dtype=base.q.dtype, device="cuda")
        mha_kv = torch.empty((1, 8, 2, 128), dtype=base.k.dtype, device="cuda")
        BatchPrefillTSWrapper().plan(
            mha_q,
            mha_kv,
            mha_kv,
            mask_type="causal",
            window_left=31,
        )
    with pytest.raises(ValueError, match="Sq <= Sk"):
        BatchPrefillTSWrapper().plan(
            torch.empty((1, 9, 4, 128), dtype=base.q.dtype, device="cuda"),
            base.k,
            base.v,
            mask_type="causal",
        )
    with pytest.raises(ValueError, match="must be provided together"):
        BatchPrefillTSWrapper().plan(
            base.q,
            base.k,
            base.v,
            qo_indptr=torch.tensor((0, 8), dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(ValueError, match="fixed Q/K/V"):
        BatchPrefillTSWrapper().plan(base.q[0], base.k[0], base.v[0])
    with pytest.raises(ValueError, match="compact"):
        BatchPrefillTSWrapper().plan(base.q.transpose(1, 2), base.k, base.v)
    with pytest.raises(NotImplementedError, match="torch.float16"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v, out_dtype=torch.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v, sm_scale=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        BatchPrefillTSWrapper().plan(base.q, base.k, base.v, output_scale=float("inf"))

    packed = _make_context_case(
        q_lengths=(4, 4),
        k_lengths=(4, 4),
        num_qo_heads=4,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=True,
        mask_type="dense",
        device="cuda",
        seed=2026071402,
    )
    bad_indptr = torch.tensor((0, 0, 8), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="strictly increasing"):
        BatchPrefillTSWrapper().plan(
            packed.q,
            packed.k,
            packed.v,
            qo_indptr=bad_indptr,
            kv_indptr=packed.kv_indptr,
        )
    with pytest.raises(ValueError, match="CUDA tensor"):
        BatchPrefillTSWrapper().plan(
            packed.q,
            packed.k,
            packed.v,
            qo_indptr=packed.qo_indptr.cpu(),
            kv_indptr=packed.kv_indptr,
        )


_CONTEXT_SMOKE_CASES = (
    pytest.param(
        torch.float16,
        False,
        (33,),
        (65,),
        4,
        4,
        "dense",
        -1,
        torch.float16,
        False,
        id="fixed-fp16-dense-mha-one-shot",
    ),
    pytest.param(
        torch.bfloat16,
        False,
        (65,),
        (65,),
        8,
        2,
        "causal",
        -1,
        torch.bfloat16,
        True,
        id="fixed-bf16-causal-single-tile-wrapper",
    ),
    pytest.param(
        torch.bfloat16,
        True,
        (257, 1),
        (257, 257),
        8,
        2,
        "causal",
        -1,
        torch.bfloat16,
        False,
        id="packed-bf16-asymmetric-max-offset-one-shot",
    ),
    pytest.param(
        torch.float16,
        True,
        (33, 257),
        (65, 257),
        4,
        4,
        "dense",
        -1,
        torch.float16,
        False,
        id="packed-fp16-dense-mixed-k-lengths-one-shot",
    ),
    pytest.param(
        _FP8,
        False,
        (33,),
        (65,),
        8,
        1,
        "causal",
        31,
        _FP8,
        True,
        id="fixed-fp8-left-window-wrapper",
    ),
    pytest.param(
        torch.float16,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        _FP8,
        True,
        id="fixed-window-fp16-to-fp8-wrapper",
    ),
    pytest.param(
        _FP8,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        torch.bfloat16,
        True,
        id="fixed-window-fp8-to-bf16-wrapper",
    ),
    pytest.param(
        torch.bfloat16,
        True,
        (128, 128, 128),
        (128, 192, 319),
        8,
        2,
        "causal",
        64,
        torch.bfloat16,
        False,
        id="packed-bf16-window-clipped-aligned-misaligned-one-shot",
    ),
)


@pytest.mark.parametrize(
    (
        "qkv_dtype",
        "packed",
        "q_lengths",
        "k_lengths",
        "num_qo_heads",
        "num_kv_heads",
        "mask_type",
        "window_left",
        "output_dtype",
        "use_wrapper",
    ),
    _CONTEXT_SMOKE_CASES,
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_bounded_public_correctness_matrix(
    qkv_dtype: torch.dtype,
    packed: bool,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    num_qo_heads: int,
    num_kv_heads: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
    use_wrapper: bool,
):
    case = _make_context_case(
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        qkv_dtype=qkv_dtype,
        packed=packed,
        mask_type=mask_type,
        window_left=window_left,
        output_dtype=output_dtype,
        device="cuda",
        seed=2026071410 + int(packed) + num_qo_heads,
    )
    if use_wrapper:
        wrapper = BatchPrefillTSWrapper()
        _plan_wrapper(wrapper, case)
        actual = wrapper.run(case.q, case.k, case.v)
        policy = dict(wrapper._policy)
        assert policy["scheduler"] == "clc_dynamic_persistent"
        assert policy["pairing"] == ("head" if window_left > 0 else "query")
        assert policy["causal_single_kv_tile"] is (
            mask_type == "causal"
            and not packed
            and window_left < 0
            and q_lengths == k_lengths
            and max(k_lengths) <= 128
        )
    else:
        actual = _run_one_shot(case)
    _assert_context_correct(actual, case)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_fixed_dense_k_tail_excludes_tma_padding():
    # With zero Q/K, every real key has score zero. A missing right-edge mask
    # therefore dilutes the output by 65/128 because TMA zero-fills the rest
    # of the final K/V tile and softmax would count those lanes as real keys.
    case = _make_context_case(
        q_lengths=(33,),
        k_lengths=(65,),
        num_qo_heads=4,
        num_kv_heads=4,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="dense",
        output_dtype=torch.float16,
        output_scale=0.75,
        device="cuda",
        seed=2026071422,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.fill_(1.0)

    actual = _run_one_shot(case)
    _assert_context_correct(actual, case)
    torch.testing.assert_close(
        actual.float(),
        torch.full_like(actual.float(), case.output_scale),
        rtol=0.0,
        atol=1e-3,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_packed_dense_k_bounds_exclude_peer_requests():
    # The first request owns only 65 K/V rows while the shared domain has
    # three 128-row tiles. Give the following request a distinct V marker so
    # both final-tile crossing and wholly-OOB tiles are observable.
    case = _make_context_case(
        q_lengths=(33, 257),
        k_lengths=(65, 257),
        num_qo_heads=4,
        num_kv_heads=4,
        qkv_dtype=torch.float16,
        packed=True,
        mask_type="dense",
        output_dtype=torch.float16,
        output_scale=0.75,
        device="cuda",
        seed=2026071423,
    )
    case.q.zero_()
    case.k.zero_()
    case.v[:65].fill_(1.0)
    case.v[65:].fill_(2.0)

    actual = _run_one_shot(case)
    _assert_context_correct(actual, case)
    torch.testing.assert_close(
        actual[:33].float(),
        torch.full_like(actual[:33].float(), 0.75),
        rtol=0.0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual[33:].float(),
        torch.full_like(actual[33:].float(), 1.5),
        rtol=0.0,
        atol=1e-3,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_fixed_window_tail_excludes_left_marker():
    # This shape has exactly one K/V tile in the head-paired window domain.
    # Every query's visible interval starts after K=0, so a marker at K=0
    # detects a tail path that applies only the causal right bound.
    case = _make_context_case(
        q_lengths=(33,),
        k_lengths=(65,),
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        window_left=31,
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        device="cuda",
        seed=2026071419,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, 0, :, 0] = 64.0

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    actual = wrapper.run(case.q, case.k, case.v)
    _assert_context_correct(actual, case)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_fixed_window_loop_excludes_right_marker():
    # The non-tile-aligned bottom-right offset makes K tile 0 a LOOP tile even
    # though its upper lanes are to the right of early query rows. A tail-only
    # right mask therefore leaks this marker into Q row zero.
    case = _make_context_case(
        q_lengths=(64,),
        k_lengths=(129,),
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="causal",
        window_left=63,
        output_dtype=torch.float16,
        output_scale=1.0,
        device="cuda",
        seed=2026071424,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, 66:, :, 0] = 64.0

    actual = _run_one_shot(case)
    _assert_context_correct(actual, case)
    assert torch.count_nonzero(actual[0, 0]) == 0


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_reuse_cache_stream_and_cuda_graph():
    _context_api._get_compiled_context.cache_clear()
    first = _make_context_case(
        q_lengths=(65,),
        k_lengths=(65,),
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071420,
    )
    second = _make_context_case(
        q_lengths=first.q_lengths,
        k_lengths=first.k_lengths,
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071421,
    )

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, first)
    with pytest.raises(ValueError, match="out must not overlap q storage"):
        wrapper.run(first.q, first.k, first.v, out=first.q)
    after_first_plan = _context_api._get_compiled_context.cache_info()
    peer = BatchPrefillTSWrapper()
    _plan_wrapper(peer, second)
    after_second_plan = _context_api._get_compiled_context.cache_info()
    assert after_second_plan.misses == after_first_plan.misses
    assert after_second_plan.hits == after_first_plan.hits + 1
    assert peer._compiled is wrapper._compiled

    shared_out = torch.full_like(first.q, float("nan"), dtype=first.output_dtype)
    returned = wrapper.run(first.q, first.k, first.v, out=shared_out)
    assert returned is shared_out
    _assert_context_correct(shared_out, first)
    first_result = shared_out.clone()

    worker_stream = torch.cuda.Stream()
    complete = torch.cuda.Event()
    with torch.cuda.stream(worker_stream):
        shared_out.fill_(float("nan"))
        returned = wrapper.run(second.q, second.k, second.v, out=shared_out)
        assert returned is shared_out
        complete.record()
    torch.cuda.current_stream().wait_event(complete)
    _assert_context_correct(shared_out, second)
    assert not torch.equal(shared_out, first_result)

    graph_out = torch.full_like(second.q, float("nan"), dtype=second.output_dtype)
    wrapper.run(second.q, second.k, second.v, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(second.q, second.k, second.v, out=graph_out)
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_context_correct(graph_out, second)
