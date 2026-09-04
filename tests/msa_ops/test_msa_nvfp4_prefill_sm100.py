"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Behavioural coverage for the NVFP4 paged-KV MSA prefill route on compute
capability 10.0/10.3.

The file has two halves, and they have deliberately different requirements.

SECTION A -- the dispatch guard, no GPU required. ``check_surface`` is ordered
"semantics, then layout, then device and architecture LAST" exactly so that its
whole semantic and layout surface is exercisable on CPU tensors, and these tests
do that: a fully well-formed CPU call must be turned away by the device
predicate and by nothing before it, and each malformed call must be turned away
by the one predicate it breaks. A guard is the only thing standing between a
neighbouring geometry and a kernel that bakes its geometry into its addressing,
so it is tested at the same granularity as the kernel.

SECTION B -- the kernel itself, skipped unless the device is compute capability
10.0 or 10.3. Correctness against the composable reference, block-table width
parity and independence, and the claim that this route allocates
nothing.
"""

import functools
import math
from types import ModuleType
from typing import Any

import pytest
import torch


# Model geometry and the packed-page map, restated from the storage contract in
# csrc/msa_prefill_nvfp4_specialized.cu rather than imported: the module cannot
# be imported at module scope here, and a test that derives its fixtures from
# the constants it is checking cannot notice them changing. Section A asserts
# these against the module's own copies.
_HEAD_DIM = 128
_PAGE_SIZE = 128
_NUM_QO_HEADS = 64
_NUM_KV_HEADS = 4
_TOPK = 16
_SCALE_VEC = 16
_DATA_DIM = _HEAD_DIM // 2
_SCALE_DIM = _HEAD_DIM // _SCALE_VEC
_DATA_HEAD_STRIDE = _PAGE_SIZE * _DATA_DIM
_SCALE_HEAD_STRIDE = _PAGE_SIZE * _SCALE_DIM
_K_SCALE_BYTE_OFFSET = _NUM_KV_HEADS * _DATA_HEAD_STRIDE
_V_DATA_BYTE_OFFSET = _K_SCALE_BYTE_OFFSET + _NUM_KV_HEADS * _SCALE_HEAD_STRIDE
_V_SCALE_BYTE_OFFSET = _V_DATA_BYTE_OFFSET + _K_SCALE_BYTE_OFFSET
_PAGE_BYTES = _V_SCALE_BYTE_OFFSET + _NUM_KV_HEADS * _SCALE_HEAD_STRIDE

# e4m3 encodings of 0.5, 1.0, 1.5 and 2.0. The fixtures draw block scales from
# this set rather than from arbitrary bytes so that no draw is a NaN, a
# subnormal or large enough to make a dequantized key overflow bf16; Section A
# asserts that these four bytes really do decode to those four values.
_E4M3_SCALE_BYTES = (0x30, 0x38, 0x3C, 0x40)
_E4M3_SCALE_VALUES = (0.5, 1.0, 1.5, 2.0)

# Pages materialized at a time while filling a pool. Bounds the transient the
# fixture itself needs, which matters for the 2048-page long-context case.
_POOL_CHUNK_PAGES = 256


def _guard_module() -> ModuleType:
    """Import the dispatch guard, or skip if the package cannot be imported.

    Nothing in Section A needs a GPU. Importing the ``flashinfer`` package
    itself does currently need a CUDA-enabled PyTorch build, because unrelated
    modules query device 0 at import time; that is a property of the package and
    not of this route, so a CPU-only build reports "not exercised" rather than a
    failure that says nothing about the guard.
    """

    if torch.version.cuda is None:
        pytest.skip("importing flashinfer requires a CUDA-enabled PyTorch build")
    from flashinfer.msa_ops import _nvfp4_prefill_sm100

    return _nvfp4_prefill_sm100


def _require_supported_gpu() -> torch.device:
    # The availability check comes first, before the import: importing the
    # flashinfer package itself needs a CUDA-enabled PyTorch build, so on a
    # CPU-only host every test in Section B has to reach a skip rather than an
    # import error that says nothing about this route.
    if not torch.cuda.is_available():
        pytest.skip("requires an SM100 or SM103 CUDA device")
    from flashinfer.utils import get_compute_capability, version_at_least

    device = torch.device("cuda")
    capability = get_compute_capability(device)
    minimum_cuda = {(10, 0): "12.8", (10, 3): "12.9"}.get(capability)
    if minimum_cuda is None:
        pytest.skip("requires compute capability 10.0 or 10.3")
    cuda_version = torch.version.cuda
    if cuda_version is None:
        pytest.skip("requires a CUDA-enabled PyTorch build")
    if not version_at_least(cuda_version, minimum_cuda):
        pytest.skip(
            f"compute capability {capability[0]}.{capability[1]} requires "
            f"CUDA {minimum_cuda} or newer"
        )
    return device


# ---------------------------------------------------------------------------
# packed-page fixtures
# ---------------------------------------------------------------------------
def _page_views(
    pool: torch.Tensor, num_pages: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut the four K/V regions out of one flat page pool.

    The route consumes a planar ``[K data | K scale | V data | V scale]`` page
    in place, and proves that relationship from the byte offsets between the
    four base pointers, so the four inputs must be strided views of a single
    ``num_pages * 73728`` byte allocation and nothing else.
    """

    data_shape = (num_pages, _NUM_KV_HEADS, _PAGE_SIZE, _DATA_DIM)
    scale_shape = (num_pages, _NUM_KV_HEADS, _PAGE_SIZE, _SCALE_DIM)
    data_stride = (_PAGE_BYTES, _DATA_HEAD_STRIDE, _DATA_DIM, 1)
    scale_stride = (_PAGE_BYTES, _SCALE_HEAD_STRIDE, _SCALE_DIM, 1)
    return (
        torch.as_strided(pool, data_shape, data_stride, 0),
        torch.as_strided(pool, data_shape, data_stride, _V_DATA_BYTE_OFFSET),
        torch.as_strided(pool, scale_shape, scale_stride, _K_SCALE_BYTE_OFFSET),
        torch.as_strided(pool, scale_shape, scale_stride, _V_SCALE_BYTE_OFFSET),
    )


@functools.lru_cache(maxsize=1)
def _v_scale_physical_index() -> torch.Tensor:
    """Physical flat position of every logical ``(token, group)`` V block scale.

    The cache writer stores the scale of logical ``(t, s)`` at
    ``((t // 4) * 4 + s // 2, (s % 2) * 4 + t % 4)``. Transcribed one entry at a
    time from that sentence, on purpose: the module states the same map as
    vectorized index arithmetic, and a round-trip against a copy of that
    arithmetic would prove nothing. Two independent spellings that invert each
    other do.
    """

    index = torch.empty(_PAGE_SIZE * _SCALE_DIM, dtype=torch.int64)
    for token in range(_PAGE_SIZE):
        for group in range(_SCALE_DIM):
            physical_token = (token // 4) * 4 + group // 2
            physical_group = (group % 2) * 4 + token % 4
            index[token * _SCALE_DIM + group] = (
                physical_token * _SCALE_DIM + physical_group
            )
    return index


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``(..., 128)`` e2m1 codes into ``(..., 64)`` bytes.

    Even elements occupy the low nibble of each byte, which is the order
    ``reference`` unpacks and the order the kernel's dequant reads.
    """

    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _write_page_scales(
    destination: torch.Tensor, logical: torch.Tensor, *, swizzled: bool
) -> None:
    """Store logical ``(page, head, token, group)`` block scales into a page.

    K scales are stored linearly; V scales are ``(4, 4)``-swizzled inside
    ``(token, scale index)``. The swizzle is invisible to shape, dtype and
    stride, so it is the one part of the page map that only a value test can
    pin down.
    """

    if not swizzled:
        destination.copy_(logical)
        return
    index = _v_scale_physical_index().to(logical.device)
    flat = logical.reshape(*logical.shape[:2], _PAGE_SIZE * _SCALE_DIM)
    physical = torch.empty_like(flat)
    physical[..., index] = flat
    destination.copy_(physical.reshape(logical.shape))


def _random_packed_pool(
    num_pages: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, ...]:
    """Fill one page pool with random NVFP4 codes and block scales.

    Random e2m1 CODES and e4m3 scale BYTES are drawn directly instead of
    quantizing bf16 values. Every 4-bit code is a valid e2m1 magnitude and every
    drawn scale byte is a finite e4m3, so the pool needs no quantization round
    trip -- and a bug in a round trip written here would look exactly like a
    kernel bug. ``reference`` reads the same bytes and defines the ground truth.
    """

    generator = torch.Generator(device=device).manual_seed(seed)
    pool = torch.empty(num_pages * _PAGE_BYTES, dtype=torch.uint8, device=device)
    k, v, k_scale, v_scale = _page_views(pool, num_pages)
    scale_bytes = torch.tensor(_E4M3_SCALE_BYTES, dtype=torch.uint8, device=device)
    for start in range(0, num_pages, _POOL_CHUNK_PAGES):
        stop = min(num_pages, start + _POOL_CHUNK_PAGES)
        code_shape = (stop - start, _NUM_KV_HEADS, _PAGE_SIZE, _HEAD_DIM)
        scale_shape = (stop - start, _NUM_KV_HEADS, _PAGE_SIZE, _SCALE_DIM)
        for data, scales, swizzled in (
            (k, k_scale, False),
            (v, v_scale, True),
        ):
            codes = torch.randint(
                0, 16, code_shape, dtype=torch.uint8, device=device, generator=generator
            )
            data[start:stop].copy_(_pack_e2m1_codes(codes))
            picks = torch.randint(
                0,
                len(_E4M3_SCALE_BYTES),
                scale_shape,
                device=device,
                generator=generator,
            )
            _write_page_scales(
                scales[start:stop], scale_bytes[picks], swizzled=swizzled
            )
    return pool, k, v, k_scale, v_scale


def _cu_seqlens(lengths: list[int], device: torch.device | None = None) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return result


def _make_q2k_indices(
    *,
    q_lens: list[int],
    kv_lens: list[int],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Ascending, distinct, ``-1`` tail-padded block selections, always in range.

    This is the contract ``msa_topk_select`` documents and the one both the
    kernel and the reference consume without re-validating, so a fixture that
    broke it would be testing something neither implements. Queries are the
    right-aligned tail of their request's KV, so a query at KV position ``p``
    may only select blocks that begin at or before ``p``.
    """

    total_q = sum(q_lens)
    output = torch.full((_NUM_KV_HEADS, total_q, _TOPK), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query_start = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        offset = kv_len - q_len
        for local_q in range(q_len):
            visible = min(
                math.ceil(kv_len / _PAGE_SIZE),
                math.ceil((offset + local_q + 1) / _PAGE_SIZE),
            )
            for kv_head in range(_NUM_KV_HEADS):
                candidates = torch.randperm(visible, generator=generator)
                count = min(_TOPK, visible)
                output[kv_head, query_start + local_q, :count] = (
                    candidates[:count].sort().values.to(torch.int32)
                )
        query_start += q_len
    return output.to(device).contiguous()


def _build_problem(
    *,
    q_lens: list[int],
    kv_lens: list[int],
    device: torch.device,
    seed: int,
    max_blocks: int | None = None,
) -> dict[str, Any]:
    """One complete, servable NVFP4 paged prefill call.

    Pages are handed out in reverse physical order so that a route which
    ignored the block table and indexed the pool directly would read the wrong
    page rather than accidentally the right one.
    """

    page_counts = [math.ceil(kv_len / _PAGE_SIZE) for kv_len in kv_lens]
    num_pages = sum(page_counts)
    width = max(page_counts) if max_blocks is None else max_blocks
    assert width >= max(page_counts)
    pool, k, v, k_scale, v_scale = _random_packed_pool(num_pages, device, seed)

    page_table = torch.full((len(kv_lens), width), -1, dtype=torch.int32)
    physical = list(reversed(range(num_pages)))
    logical_page = 0
    for request, count in enumerate(page_counts):
        for block in range(count):
            page_table[request, block] = physical[logical_page]
            logical_page += 1

    total_q = sum(q_lens)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    q = (
        torch.randn(
            (total_q, _NUM_QO_HEADS, _HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / 3.0
    ).to(torch.bfloat16)
    return {
        "pool": pool,
        "q": q,
        "k": k,
        "v": v,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q2k_indices": _make_q2k_indices(
            q_lens=q_lens, kv_lens=kv_lens, seed=seed + 2, device=device
        ),
        "cu_seqlens_q": _cu_seqlens(q_lens, device),
        "page_table": page_table.to(device).contiguous(),
        "seqused_k": torch.tensor(kv_lens, dtype=torch.int32, device=device),
        "softmax_scale": float(_HEAD_DIM**-0.5),
        # Neither global scale is 1.0: a route that dropped one would otherwise
        # agree with the reference exactly.
        "k_global_scale": 0.75,
        "v_global_scale": 1.25,
    }


def _serve(problem: dict[str, Any]) -> torch.Tensor:
    """Run the problem through the public API, exactly as a consumer would."""

    from flashinfer.msa_ops import msa_sparse_attention

    return msa_sparse_attention(
        problem["q"],
        problem["k"],
        problem["v"],
        problem["q2k_indices"],
        problem["cu_seqlens_q"],
        causal=True,
        softmax_scale=problem["softmax_scale"],
        page_table=problem["page_table"],
        seqused_k=problem["seqused_k"],
        k_scale=problem["k_scale"],
        v_scale=problem["v_scale"],
        k_global_scale=problem["k_global_scale"],
        v_global_scale=problem["v_global_scale"],
    )


def _reference(problem: dict[str, Any]) -> torch.Tensor:
    from flashinfer.msa_ops import _nvfp4_prefill_sm100 as module

    return module.reference(
        q=problem["q"],
        k_data=problem["k"],
        v_data=problem["v"],
        k_scale=problem["k_scale"],
        v_scale=problem["v_scale"],
        q2k_indices=problem["q2k_indices"],
        cu_seqlens_q=problem["cu_seqlens_q"],
        page_table=problem["page_table"],
        seqused_k=problem["seqused_k"],
        softmax_scale=problem["softmax_scale"],
        k_global_scale=problem["k_global_scale"],
        v_global_scale=problem["v_global_scale"],
        out=torch.zeros_like(problem["q"]),
    )


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Cosine similarity and RMS error relative to the reference's own RMS."""

    left = actual.float().reshape(-1)
    right = expected.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0).item()
    relative_rms = (
        torch.sqrt(torch.mean((left - right) ** 2)) / torch.sqrt(torch.mean(right**2))
    ).item()
    return cosine, relative_rms


# ===========================================================================
# SECTION A -- dispatch guard. No GPU is used by anything below this line.
# ===========================================================================
def _cpu_surface(
    *,
    num_pages: int = 2,
    batch_size: int = 1,
    total_q: int = 8,
    max_blocks: int = 1,
    scale_dtype: torch.dtype = torch.uint8,
) -> dict[str, Any]:
    """A fully well-formed ``check_surface`` call, as CPU tensors.

    Everything the guard inspects before its device predicate is correct here,
    so a test can break exactly one thing and attribute the rejection to it.
    """

    pool = torch.zeros(num_pages * _PAGE_BYTES, dtype=torch.uint8)
    k, v, k_scale, v_scale = _page_views(pool, num_pages)
    if scale_dtype is not torch.uint8:
        k_scale = k_scale.view(scale_dtype)
        v_scale = v_scale.view(scale_dtype)
    q2k_indices = torch.full((_NUM_KV_HEADS, total_q, _TOPK), -1, dtype=torch.int32)
    q2k_indices[:, :, 0] = 0
    counts = [total_q // batch_size] * batch_size
    counts[-1] += total_q - sum(counts)
    return {
        "q": torch.zeros((total_q, _NUM_QO_HEADS, _HEAD_DIM), dtype=torch.bfloat16),
        "k": k,
        "v": v,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "q2k_indices": q2k_indices,
        "cu_seqlens_q": _cu_seqlens(counts),
        "page_table": torch.zeros((batch_size, max_blocks), dtype=torch.int32),
        "seqused_k": torch.full((batch_size,), _PAGE_SIZE, dtype=torch.int32),
        "cu_seqlens_k": None,
        "causal": True,
        "return_softmax_lse": False,
        "return_temperature_lse": False,
        "lse_temperature_scale": 1.0,
        "k_global_scale": 1.0,
        "v_global_scale": 1.0,
        "q_offset": None,
    }


def _check(surface: dict[str, Any], **overrides: Any) -> str | None:
    module = _guard_module()
    return module.check_surface(**{**surface, **overrides})


def _is_device_reason(reason: str | None) -> bool:
    return reason is not None and ("CUDA" in reason or "compute capability" in reason)


def test_a_well_formed_call_is_turned_away_only_by_the_device_check() -> None:
    """Every semantic and layout predicate passes on CPU tensors.

    This is the property the rest of Section A depends on: the guard is ordered
    so that a host with no GPU still reaches the last predicate. If a future
    edit hoists a device check above the semantics, this test fails and every
    other rejection test below silently stops proving what it names.
    """

    reason = _check(_cpu_surface())
    assert reason is not None, "a CPU tensor cannot be served by this route"
    assert _is_device_reason(reason), reason


@pytest.mark.parametrize("name", ["k", "v"])
def test_packed_kv_must_be_uint8(name: str) -> None:
    """A cache that is not packed bytes is not this route's cache.

    The kernel reads two e2m1 codes per byte; any other element type would be
    reinterpreted rather than converted.
    """

    surface = _cpu_surface()
    reason = _check(surface, **{name: surface[name].view(torch.int8)})
    assert reason is not None
    assert reason.startswith(name), reason
    assert "packed NVFP4" in reason, reason


@pytest.mark.parametrize("name", ["k_scale", "v_scale"])
def test_block_scales_are_required(name: str) -> None:
    """NVFP4 without block scales is not decodable, so it is refused.

    A missing scale tensor cannot be defaulted to 1.0: the scales carry the
    per-16-element exponent that the 4-bit codes do not.
    """

    reason = _check(_cpu_surface(), **{name: None})
    assert reason is not None
    assert "k_scale and v_scale" in reason, reason


@pytest.mark.parametrize("name", ["k_global_scale", "v_global_scale"])
def test_global_scales_are_required(name: str) -> None:
    """Both global dequant scales are required, not optional with a default.

    They are the per-tensor half of the NVFP4 encoding; assuming 1.0 for an
    absent one would silently rescale the whole attend.
    """

    reason = _check(_cpu_surface(), **{name: None})
    assert reason is not None
    assert "k_global_scale and v_global_scale" in reason, reason


def test_q_must_be_bfloat16() -> None:
    """The MMA operand type is fixed, so a bf16 Q is a hard requirement."""

    surface = _cpu_surface()
    reason = _check(surface, q=surface["q"].to(torch.float16))
    assert reason is not None
    assert "bfloat16" in reason, reason


def test_a_non_causal_call_is_refused_rather_than_answered_causally() -> None:
    """The kernel masks unconditionally, so ``causal=False`` cannot be served.

    This is the difference between a preference and a capability: a non-causal
    call that fell through to this kernel would receive a causal answer with no
    diagnostic at all.
    """

    reason = _check(_cpu_surface(), causal=False)
    assert reason is not None
    assert "causal-only" in reason, reason


def test_an_explicit_q_offset_is_refused() -> None:
    """The query offset is derived, so an explicit one would be a second source.

    The kernel places the queries as the right-aligned tail of their request's
    KV (``seqused_k - query_length``); an explicit offset it does not read would
    be silently ignored.
    """

    reason = _check(_cpu_surface(), q_offset=torch.zeros(1, dtype=torch.int32))
    assert reason is not None
    assert "q_offset" in reason, reason


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"return_softmax_lse": True}, id="softmax-lse"),
        pytest.param({"return_temperature_lse": True}, id="temperature-lse"),
        pytest.param({"lse_temperature_scale": 2.0}, id="lse-temperature-scale"),
    ],
)
def test_lse_requests_are_refused(override: dict[str, Any]) -> None:
    """The kernel produces no LSE, so any call that wants one is declined.

    Returning the output alone and letting the caller discover the missing LSE
    later would be worse than not serving the call.
    """

    reason = _check(_cpu_surface(), **override)
    assert reason is not None
    assert "LSE" in reason, reason


@pytest.mark.parametrize("name", ["page_table", "seqused_k"])
def test_the_paged_layout_is_required(name: str) -> None:
    """This route serves paged KV only; a flat cache has no block table."""

    reason = _check(_cpu_surface(), **{name: None})
    assert reason is not None
    assert "seqused_k" in reason, reason


def test_cu_seqlens_k_is_refused_because_seqused_k_is_the_length_authority() -> None:
    """Two independent sources of KV length are refused, not reconciled.

    ``seqused_k`` is the sole authority on this route and the kernel derives the
    causal offset from it. Accepting ``cu_seqlens_k`` as well would mean
    silently preferring one of two possibly disagreeing answers, and checking
    that they agree would need a device-to-host copy of both.
    """

    reason = _check(
        _cpu_surface(), cu_seqlens_k=torch.tensor([0, _PAGE_SIZE], dtype=torch.int32)
    )
    assert reason is not None
    assert "cu_seqlens_k" in reason, reason
    assert "seqused_k" in reason, reason


@pytest.mark.parametrize(
    ("num_qo_heads", "head_dim"),
    [
        pytest.param(32, _HEAD_DIM, id="32-query-heads"),
        pytest.param(_NUM_QO_HEADS, 64, id="head-dim-64"),
    ],
)
def test_the_model_geometry_is_fixed(num_qo_heads: int, head_dim: int) -> None:
    """A neighbouring geometry must never reach a kernel built from this one.

    64 query heads and head_dim 128 are compile-time constants of the shared
    memory map and the 128-row MMA tile, not tuning parameters.
    """

    surface = _cpu_surface()
    total_q = int(surface["q"].shape[0])
    reason = _check(
        surface,
        q=torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16),
    )
    assert reason is not None
    assert "64 query heads" in reason, reason


@pytest.mark.parametrize("topk", [8, 32])
def test_topk_must_be_16(topk: int) -> None:
    """topk is the q2k row stride and the union-table capacity, not a hint.

    A topk of 8 would make the kernel read past the end of every q2k row, and a
    topk of 32 would let eight queries insert more block ids than the 128-slot
    union table can hold -- whose linear probe is unbounded because it cannot
    fill. Both are refused here and asserted again by the host binding.
    """

    surface = _cpu_surface()
    total_q = int(surface["q"].shape[0])
    reason = _check(
        surface,
        q2k_indices=torch.zeros((_NUM_KV_HEADS, total_q, topk), dtype=torch.int32),
    )
    assert reason is not None
    assert "q2k_indices" in reason, reason
    assert str(_TOPK) in reason, reason


@pytest.mark.parametrize("layout", ["wrong-leading-dim", "gappy-innermost"])
def test_q2k_indices_must_be_one_dense_row_per_query(layout: str) -> None:
    """What the union builder actually needs is a DENSE ROW, not a dense buffer.

    The kernel takes the two outer strides now, so a permuted view is read in
    place. What it cannot read is a row whose innermost dimension is itself
    strided -- the eight-query loop walks `slot` by one.
    """

    surface = _cpu_surface()
    total_q = int(surface["q"].shape[0])
    if layout == "wrong-leading-dim":
        q2k_indices = torch.zeros((2, total_q, _TOPK), dtype=torch.int32)
    else:
        q2k_indices = torch.zeros(
            (_NUM_KV_HEADS, _TOPK, total_q), dtype=torch.int32
        ).transpose(1, 2)
        assert tuple(q2k_indices.shape) == (_NUM_KV_HEADS, total_q, _TOPK)
        assert q2k_indices.stride(2) != 1
    reason = _check(surface, q2k_indices=q2k_indices)
    assert reason is not None
    assert "q2k_indices" in reason, reason


def test_a_head_major_view_of_a_token_major_buffer_is_admitted() -> None:
    """The layout the MSA indexer actually produces, admitted uncopied.

    `topk[nd:num_tokens].transpose(0, 1)` is non-contiguous with a token stride
    of `num_kv_heads * topk`, and every consumer had to call `.contiguous()` on
    it once per prefill call before the strides became kernel arguments.
    """

    surface = _cpu_surface()
    total_q = int(surface["q"].shape[0])
    token_major = torch.zeros((total_q + 9, _NUM_KV_HEADS, _TOPK), dtype=torch.int32)
    view = token_major[:total_q].transpose(0, 1)
    assert not view.is_contiguous()
    assert view.stride() == (_TOPK, _NUM_KV_HEADS * _TOPK, 1)
    # Everything else in this fixture is a host tensor, so reaching the
    # architecture conjunct is what says the layout was accepted -- the same
    # property `test_a_well_formed_call_is_turned_away_only_by_the_device_check`
    # relies on.
    assert _is_device_reason(_check(surface, q2k_indices=view))


@pytest.mark.parametrize("width", [128, 256, 257, 2048, 32768])
def test_the_block_table_width_is_a_free_parameter(width: int) -> None:
    """No power-of-two, no 128, no benchmarked width: any width is admitted.

    The per-tile block union is a 128-slot hash table keyed on the block id, not
    a bitmap indexed by it, so its size follows ``8 * topk`` and not the number
    of blocks that exist. 257 and 2048 are here because a route that narrowed
    the width axis would reject exactly those and still pass a 128-wide test;
    2048 is a 262,144-token context.
    """

    reason = _check(_cpu_surface(max_blocks=width))
    assert _is_device_reason(reason), reason
    assert "width" not in reason, reason


def test_the_only_width_ceiling_is_the_24_bit_block_id() -> None:
    """The width bound is the id packing, and it is checked rather than clamped.

    A selected block id is carried in the low 24 bits of a union-table entry, so
    a width of ``MAX_SELECTABLE_BLOCKS`` is admissible and one more is not.
    Clamping instead would drop selected blocks from the union with no
    diagnostic, which is why this is a rejection and not a min().
    """

    module = _guard_module()
    surface = _cpu_surface()
    assert module.MAX_SELECTABLE_BLOCKS == 0x00FFFFFF
    assert module.MAX_CONTEXT_TOKENS == module.MAX_SELECTABLE_BLOCKS * _PAGE_SIZE

    # ~64 MiB of host int32 per table, so they are built and released one at a
    # time rather than parametrized.
    page_table = torch.zeros((1, module.MAX_SELECTABLE_BLOCKS), dtype=torch.int32)
    reason = _check(surface, page_table=page_table)
    del page_table
    assert _is_device_reason(reason), reason
    assert "width" not in reason, reason

    page_table = torch.zeros((1, module.MAX_SELECTABLE_BLOCKS + 1), dtype=torch.int32)
    reason = _check(surface, page_table=page_table)
    del page_table
    assert reason is not None
    assert "width" in reason, reason
    assert str(module.MAX_SELECTABLE_BLOCKS) in reason, reason


@pytest.mark.parametrize("name", ["k", "v", "k_scale", "v_scale"])
@pytest.mark.parametrize("defect", ["dtype", "shape", "stride"])
def test_each_page_region_is_pinned_by_dtype_shape_and_stride(
    name: str, defect: str
) -> None:
    """The kernel asserts these strides instead of reading them.

    Every byte address in the kernel is derived from the page map rather than
    from the tensor's own strides, so a region that disagrees with the map by
    dtype, extent or stride would be addressed as though it did not. The
    rejection has to name the region, because four look-alike tensors are
    involved and "layout mismatch" would not locate the caller's bug.
    """

    num_pages = 2
    pool = torch.zeros(num_pages * _PAGE_BYTES, dtype=torch.uint8)
    surface = _cpu_surface(num_pages=num_pages)
    is_scale = name.endswith("_scale")
    inner = _SCALE_DIM if is_scale else _DATA_DIM
    head_stride = _SCALE_HEAD_STRIDE if is_scale else _DATA_HEAD_STRIDE
    offset = {
        "k": 0,
        "v": _V_DATA_BYTE_OFFSET,
        "k_scale": _K_SCALE_BYTE_OFFSET,
        "v_scale": _V_SCALE_BYTE_OFFSET,
    }[name]
    shape = (num_pages, _NUM_KV_HEADS, _PAGE_SIZE, inner)
    stride = (_PAGE_BYTES, head_stride, inner, 1)

    if defect == "dtype":
        # int8 is neither the uint8 the data regions require nor one of the two
        # spellings (uint8, float8_e4m3fn) a scale region accepts.
        broken = torch.as_strided(pool, shape, stride, offset).view(torch.int8)
    elif defect == "shape":
        broken = torch.as_strided(
            pool, (num_pages, _NUM_KV_HEADS, _PAGE_SIZE // 2, inner), stride, offset
        )
    else:
        broken = torch.as_strided(
            pool, shape, (_NUM_KV_HEADS * head_stride, head_stride, inner, 1), offset
        )
    reason = _check(surface, **{name: broken})
    assert reason is not None
    assert reason.startswith(name), reason
    if defect in ("shape", "stride"):
        assert defect in reason, reason


@pytest.mark.parametrize(
    ("name", "offset"),
    [
        pytest.param("k_scale", _K_SCALE_BYTE_OFFSET, id="k_scale"),
        pytest.param("v", _V_DATA_BYTE_OFFSET, id="v_data"),
        pytest.param("v_scale", _V_SCALE_BYTE_OFFSET, id="v_scale"),
    ],
)
def test_the_four_regions_must_be_views_of_one_packed_page(
    name: str, offset: int
) -> None:
    """Separate allocations are refused even when every stride matches.

    Shape, dtype and stride cannot tell four views of one planar page apart from
    four unrelated allocations strided the same way, and the ``(4, 4)`` V-scale
    swizzle is invisible to all three. The byte offsets between the four base
    pointers are what actually pins the inputs to the page map the cache writer
    used, and they are also what lets the kernel address a whole page from k's
    base pointer.
    """

    num_pages = 2
    surface = _cpu_surface(num_pages=num_pages)
    # A second pool of the same size, sliced at the same internal offsets: the
    # delta from k's base is then off by exactly the distance between two live
    # allocations, which cannot be zero.
    other = torch.zeros(num_pages * _PAGE_BYTES, dtype=torch.uint8)
    replacement = dict(
        zip(
            ("k", "v", "k_scale", "v_scale"),
            _page_views(other, num_pages),
            strict=True,
        )
    )[name]
    reason = _check(surface, **{name: replacement})
    assert reason is not None
    assert reason.startswith(name), reason
    assert f"+{offset} B" in reason, reason


def test_float8_e4m3fn_scale_views_are_accepted_as_the_same_bytes() -> None:
    """A splitter that hands back e4m3 views is serving the same page.

    ``nvfp4_split_data_scale`` returns the block-scale regions as
    ``float8_e4m3fn`` and the data regions as ``uint8``; both spell the same
    bytes, so the guard accepts either and ``as_scale_bytes`` reinterprets
    without copying -- which is what keeps the base pointers the layout proof
    just checked unchanged.
    """

    module = _guard_module()
    surface = _cpu_surface(scale_dtype=torch.float8_e4m3fn)
    assert surface["k_scale"].dtype == torch.float8_e4m3fn
    reason = _check(surface)
    assert _is_device_reason(reason), reason

    for name in ("k_scale", "v_scale"):
        as_bytes = module.as_scale_bytes(surface[name])
        assert as_bytes.dtype == torch.uint8
        assert as_bytes.data_ptr() == surface[name].data_ptr()
        assert tuple(as_bytes.stride()) == tuple(surface[name].stride())
    already_bytes = _cpu_surface()["k_scale"]
    assert module.as_scale_bytes(already_bytes) is already_bytes


def test_the_allowlist_is_one_geometry_and_matches_the_module() -> None:
    """The capability surface is a single row, and the file agrees with the code.

    The allowlist is a capability statement, not a benchmarked-shape gate: the
    one row is the model geometry the kernel body is built from. A field list
    that drifted from the module's would silently disable the route, because an
    unreadable allowlist is an empty allowlist.
    """

    module = _guard_module()
    payload = module._read_workload_file()
    assert tuple(payload["fields"]) == module._WORKLOAD_FIELDS
    assert payload["workloads"] == [
        [_NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM, _PAGE_SIZE, _TOPK]
    ]
    assert module._load_allowlist() == frozenset(
        {(_NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM, _PAGE_SIZE, _TOPK)}
    )


def test_the_page_map_agrees_with_the_storage_contract() -> None:
    """The offsets this file's fixtures write to are the ones the guard checks.

    Every device test below builds its pages from the constants at the top of
    this file. If those and the module's ever diverged, the fixtures would still
    look self-consistent and would be proving nothing about the shipped map.
    """

    module = _guard_module()
    for name, expected in (
        ("_HEAD_DIM", _HEAD_DIM),
        ("_PAGE_SIZE", _PAGE_SIZE),
        ("_NUM_QO_HEADS", _NUM_QO_HEADS),
        ("_NUM_KV_HEADS", _NUM_KV_HEADS),
        ("_TOPK", _TOPK),
        ("_DATA_DIM", _DATA_DIM),
        ("_SCALE_DIM", _SCALE_DIM),
        ("_DATA_HEAD_STRIDE", _DATA_HEAD_STRIDE),
        ("_SCALE_HEAD_STRIDE", _SCALE_HEAD_STRIDE),
        ("_K_SCALE_BYTE_OFFSET", _K_SCALE_BYTE_OFFSET),
        ("_V_DATA_BYTE_OFFSET", _V_DATA_BYTE_OFFSET),
        ("_V_SCALE_BYTE_OFFSET", _V_SCALE_BYTE_OFFSET),
        ("_PAGE_BYTES", _PAGE_BYTES),
    ):
        assert getattr(module, name) == expected, name
    assert _PAGE_BYTES == 73728
    layout = module._read_workload_file()["kv_cache_layout"]
    assert layout["page_bytes"] == _PAGE_BYTES
    assert layout["regions"] == [
        ["k_data", 0, _NUM_KV_HEADS * _DATA_HEAD_STRIDE],
        ["k_scale", _K_SCALE_BYTE_OFFSET, _NUM_KV_HEADS * _SCALE_HEAD_STRIDE],
        ["v_data", _V_DATA_BYTE_OFFSET, _NUM_KV_HEADS * _DATA_HEAD_STRIDE],
        ["v_scale", _V_SCALE_BYTE_OFFSET, _NUM_KV_HEADS * _SCALE_HEAD_STRIDE],
    ]


def test_the_fixture_v_scale_swizzle_inverts_the_reference_unswizzle() -> None:
    """The fixtures write the V-scale swizzle the reference reads back.

    This is the one part of the page map that shape, dtype and stride cannot
    check, so if the writer here and the reader in ``reference`` disagreed, every
    numerics test in Section B would compare a kernel against an oracle reading
    different bytes. The two spellings are independent -- a scalar transcription
    of the contract sentence here, vectorized index arithmetic there -- so the
    round trip is a real check and not a tautology.
    """

    module = _guard_module()
    logical = (
        torch.arange(_PAGE_SIZE * _SCALE_DIM, dtype=torch.int32)
        .reshape(1, 1, _PAGE_SIZE, _SCALE_DIM)
        .to(torch.uint8)
    )
    physical = torch.empty_like(logical)
    _write_page_scales(physical, logical, swizzled=True)
    unswizzle = module._v_scale_unswizzle_index("cpu")
    recovered = physical.reshape(-1)[unswizzle].reshape(logical.shape)
    assert torch.equal(recovered, logical)
    # A linear store would pass the round trip above only if the swizzle were
    # the identity, which it is not.
    assert not torch.equal(physical, logical)

    linear = torch.empty_like(logical)
    _write_page_scales(linear, logical, swizzled=False)
    assert torch.equal(linear, logical)

    decoded = (
        torch.tensor(_E4M3_SCALE_BYTES, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
    )
    assert decoded.tolist() == list(_E4M3_SCALE_VALUES)


# ===========================================================================
# SECTION B -- the kernel. Every test below requires compute capability
# 10.0 or 10.3 and is skipped otherwise.
# ===========================================================================
_NUMERICS_CASES = [
    pytest.param([512], [512], id="prefill-no-prior-context"),
    pytest.param([8], [4096], id="short-tail-over-long-context"),
    pytest.param([64, 17], [1024, 300], id="batch2-mixed-lengths"),
    pytest.param([100], [333], id="partial-last-block"),
    pytest.param([13], [900], id="query-length-not-a-multiple-of-8"),
]


@pytest.mark.parametrize(("q_lens", "kv_lens"), _NUMERICS_CASES)
def test_nvfp4_prefill_matches_the_composable_reference(
    q_lens: list[int], kv_lens: list[int]
) -> None:
    """The public API serves NVFP4 paged prefill, and it serves it correctly.

    The oracle is the FP32 composable reference, which is a peer of the kernel
    rather than an approximation of it: same dequant, same block-scale layouts,
    same right-aligned causal mask. The cases cover a request with no prior
    context, a short tail over a long one, a heterogeneous batch, a KV length
    that does not fill its last block, and a query length that is not a multiple
    of the kernel's 8-token row tile.
    """

    device = _require_supported_gpu()
    problem = _build_problem(q_lens=q_lens, kv_lens=kv_lens, device=device, seed=7)
    actual = _serve(problem)
    assert actual.shape == problem["q"].shape
    assert actual.dtype == torch.bfloat16

    cosine, relative_rms = _agreement(actual, _reference(problem))
    assert cosine >= 0.999, f"cosine similarity {cosine}"
    assert relative_rms <= 0.02, f"relative RMS {relative_rms}"


def test_the_block_table_width_does_not_change_the_answer() -> None:
    """A 2048-wide block table returns the 128-wide answer bit for bit.

    Not merely "close": nothing in the kernel is proportional to the width
    except one runtime row stride, and the per-tile union is a hash table keyed
    on the block id, so the wide table's unused columns cannot reach the
    arithmetic at all. Equality is the observable form of that claim, and this
    width is the capability the route exists to provide.
    """

    device = _require_supported_gpu()
    arguments = dict(q_lens=[24], kv_lens=[100 * _PAGE_SIZE], device=device, seed=11)
    narrow = _build_problem(**arguments, max_blocks=128)
    wide = _build_problem(**arguments, max_blocks=2048)
    assert torch.equal(narrow["pool"], wide["pool"])
    assert torch.equal(narrow["q"], wide["q"])
    assert torch.equal(narrow["q2k_indices"], wide["q2k_indices"])
    assert narrow["page_table"].shape[1] == 128
    assert wide["page_table"].shape[1] == 2048
    assert torch.equal(wide["page_table"][:, :128], narrow["page_table"])

    assert torch.equal(_serve(narrow), _serve(wide))


def test_a_262144_token_context_at_width_2048_is_served() -> None:
    """The width axis is unconstrained in practice, not just in principle.

    2048 blocks of 128 tokens is a 262,144-token context: 144 MiB of packed
    pages, which is the whole cost -- the route adds no scratch on top. The
    problem is skipped rather than scaled down when the device cannot hold the
    pool, because a smaller pool would no longer be this test.
    """

    device = _require_supported_gpu()
    num_pages = 2048
    pool_bytes = num_pages * _PAGE_BYTES
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 4 * pool_bytes:
        pytest.skip(
            f"a {pool_bytes} byte page pool does not fit in {free_bytes} free bytes"
        )
    problem = _build_problem(
        q_lens=[8],
        kv_lens=[num_pages * _PAGE_SIZE],
        device=device,
        seed=13,
        max_blocks=num_pages,
    )
    actual = _serve(problem)
    cosine, relative_rms = _agreement(actual, _reference(problem))
    assert cosine >= 0.999, f"cosine similarity {cosine}"
    assert relative_rms <= 0.02, f"relative RMS {relative_rms}"


def test_an_empty_union_produces_finite_zeros() -> None:
    """A tile that selects no block must return zeros, not uninitialized TMEM.

    ``tcgen05.alloc`` does not zero tensor memory, and a tile whose union is
    empty issues no PV at all, so its output accumulator is never written. The
    epilogue scales that accumulator by a zero denominator rather than replacing
    it, which is safe for finite garbage and NOT safe for a NaN or Inf bit
    pattern left behind by a previous occupant of those columns.

    Nothing in the numerics cases reaches this path -- ``msa_topk_select`` always
    returns at least one visible block -- so it is constructed here directly, and
    the columns are dirtied first with a real problem so that the accumulator
    holds something rather than whatever the allocator happened to hand out.
    """

    device = _require_supported_gpu()
    problem = _build_problem(q_lens=[16], kv_lens=[512], device=device, seed=29)

    # Dirty the tensor-memory columns with a real, large-magnitude problem.
    _serve(problem)
    torch.cuda.synchronize(device)

    empty = dict(problem)
    empty["q2k_indices"] = torch.full_like(problem["q2k_indices"], -1)
    served = _serve(empty)
    torch.cuda.synchronize(device)

    assert torch.isfinite(served).all(), "an empty union produced a non-finite output"
    assert torch.count_nonzero(served) == 0, "an empty union produced a non-zero output"


def test_repeated_calls_are_bit_identical_below_the_union_table_width() -> None:
    """Run-to-run reproducibility, where the route claims to have it.

    The per-tile union is consumed in ascending hash-slot order. While a request
    has at most ``UNION_TABLE_SLOTS`` blocks the slot is a permutation of the
    selected set, so no two selected ids collide, the insert is a commutative
    ``atomicOr``, and the accumulation order is a function of the selected set
    rather than of how the atomics resolved. Bit-identity is the observable form
    of that, and it is asserted at a context just under the boundary.

    Above the boundary the insert linear-probes under ``atomicCAS`` and the
    order is genuinely race-dependent; that case is exercised for CORRECTNESS by
    the 262,144-token test, and is deliberately not asserted to be bit-stable
    here. See the module docstring of ``_nvfp4_prefill_sm100`` for the contract.
    """

    device = _require_supported_gpu()
    guard = _guard_module()
    assert guard.DETERMINISTIC_CONTEXT_TOKENS == guard.UNION_TABLE_SLOTS * _PAGE_SIZE

    kv_len = guard.DETERMINISTIC_CONTEXT_TOKENS - _PAGE_SIZE
    problem = _build_problem(q_lens=[64], kv_lens=[kv_len], device=device, seed=31)
    assert math.ceil(kv_len / _PAGE_SIZE) <= guard.UNION_TABLE_SLOTS

    first = _serve(problem).clone()
    for _ in range(4):
        assert torch.equal(first, _serve(problem))

    stats = guard.msa_prefill_nvfp4_specialized_stats()
    assert (
        stats["run_to_run_bitwise_reproducible_up_to_context"]
        == guard.DETERMINISTIC_CONTEXT_TOKENS
    )


def test_stats_report_the_capability_and_count_one_dispatch() -> None:
    """The introspection a consumer reads before turning NVFP4 KV on.

    A framework enabling an NVFP4 MSA cache has to know that this route is
    causal-only, that ``seqused_k`` is the KV-length authority, how wide a block
    table it will accept and what it costs in scratch. ``dispatch_count`` is the
    only way a caller can prove the route -- rather than something adjacent to
    it -- served a particular call.
    """

    device = _require_supported_gpu()
    from flashinfer.msa_ops import msa_prefill_nvfp4_specialized_stats

    stats = msa_prefill_nvfp4_specialized_stats()
    assert stats["scratch_bytes_per_request_at_128_blocks"] == 0
    assert stats["max_selectable_blocks"] == 0x00FFFFFF
    assert stats["causal_only"] is True
    assert stats["kv_length_authority"] == "seqused_k"

    problem = _build_problem(q_lens=[16], kv_lens=[512], device=device, seed=19)
    # One warm-up outside the window: the first eager dispatch on a device also
    # builds and warms the module, and a warm launch is deliberately not counted
    # as a caller's dispatch.
    _serve(problem)
    before = msa_prefill_nvfp4_specialized_stats()["dispatch_count"]
    _serve(problem)
    after = msa_prefill_nvfp4_specialized_stats()["dispatch_count"]
    assert after - before == 1


def test_the_route_allocates_nothing_beyond_the_output() -> None:
    """Zero scratch, and zero at every block-table width.

    Each CTA dequantizes one selected page into its own dynamic shared memory,
    so nothing proportional to the block-table width, the batch or the context
    is ever allocated in HBM. Measuring the same problem at width 128 and at
    width 2048 and finding the same peak is the proof: a route that staged a
    dequantized copy, or sized any buffer from the width, could not produce two
    equal numbers.
    """

    device = _require_supported_gpu()
    arguments = dict(q_lens=[32], kv_lens=[100 * _PAGE_SIZE], device=device, seed=23)
    problems = {
        128: _build_problem(**arguments, max_blocks=128),
        2048: _build_problem(**arguments, max_blocks=2048),
    }
    peaks = {}
    for width, problem in problems.items():
        _serve(problem)  # build, warm and settle the caching allocator first
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        served = _serve(problem)
        torch.cuda.synchronize(device)
        peaks[width] = torch.cuda.max_memory_allocated(device) - before
        assert peaks[width] == served.numel() * served.element_size(), width
        del served
    assert peaks[128] == peaks[2048]


def test_a_strided_selection_is_bit_identical_to_its_contiguous_copy() -> None:
    """The copy removal on the prefill path, proved on the device.

    `topk[nd:num_tokens].transpose(0, 1)` is the view a consumer holds; the
    kernel now reads it in place. Both calls read the same integers in the same
    order, so the outputs must be EQUAL, not merely close -- a tolerance here
    would hide exactly the addressing bug this test exists to catch.
    """

    device = _require_supported_gpu()
    problem = _build_problem(
        q_lens=[48, 16, 1], kv_lens=[9000, 130, 4096], device=device, seed=31
    )
    contiguous = problem["q2k_indices"]
    total_q = int(problem["q"].shape[0])

    padded = total_q + 13
    token_major = torch.zeros(
        (padded, _NUM_KV_HEADS, _TOPK), dtype=torch.int32, device=device
    )
    token_major[:total_q] = contiguous.permute(1, 0, 2)
    strided = token_major[:total_q].transpose(0, 1)
    assert not strided.is_contiguous()
    assert strided.stride() == (_TOPK, _NUM_KV_HEADS * _TOPK, 1)
    assert strided.stride(1) != _TOPK  # the row stride the old kernel assumed
    assert torch.equal(strided, contiguous)

    base = _serve(problem).clone()
    problem["q2k_indices"] = strided
    served = _serve(problem).clone()
    assert torch.equal(served, base)

    cosine, relative_rms = _agreement(served, _reference(problem))
    assert cosine >= 0.999, cosine
    assert relative_rms <= 0.06, relative_rms


def test_a_declined_prefill_says_WHICH_axis_declined_it() -> None:
    """Same defect, same fix, on the route that has no decode-side sibling.

    Prefill is where this matters most: the decode hook and the prefill hook
    both fall through to the same blanket ``NotImplementedError``, and it is
    false in both -- NVFP4 K/V *is* supported here, one axis of one call is
    not. There is no other implementation of MSA over an NVFP4 cache, so the
    message the operator gets is the last word on the failure.

    No GPU: ``_validate_scale_arguments`` inspects dtypes only, so the whole
    fall-through is reachable on a host.
    """

    from flashinfer.msa_ops import _blackwell_sm100

    surface = _cpu_surface()
    total_q = int(surface["q"].shape[0])
    with pytest.raises(NotImplementedError) as excinfo:
        _blackwell_sm100.blackwell_msa_sparse_attention(
            surface["q"],
            surface["k"],
            surface["v"],
            torch.zeros((_NUM_KV_HEADS, total_q, 8), dtype=torch.int32),
            surface["cu_seqlens_q"],
            causal=True,
            softmax_scale=1.0,
            page_table=surface["page_table"],
            seqused_k=surface["seqused_k"],
            k_scale=surface["k_scale"],
            v_scale=surface["v_scale"],
            k_global_scale=surface["k_global_scale"],
            v_global_scale=surface["v_global_scale"],
        )
    text = str(excinfo.value)
    assert "q2k_indices" in text, text
    assert "IS supported on this architecture" in text, text
    assert "no other implementation" in text, text
