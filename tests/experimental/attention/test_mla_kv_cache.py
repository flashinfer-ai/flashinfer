from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.sm12x.attention._shared.mla.kernel import (
    run_unified_decode,
)
from flashinfer.experimental.sm12x.attention._shared.mla.kv_cache import (
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache,
    concat_and_cache_nvfp4_mla_fp8_rope,
)
from flashinfer.experimental.sm12x.attention._shared.mla.prefill_mg import (
    run_unified_prefill_mg,
)
from flashinfer.experimental.sm12x.attention._shared.mla.traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
)
from flashinfer.experimental.sm12x.attention.sparse_mla._scratch import (
    SM12XSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)

from .._reference.helpers import E2M1_TO_FLOAT32
from ..conftest import require_sm12x as require_sm120


_RECORD_BYTES = 368
_NOPE_BYTES = 256
_GROUP_SCALES_OFFSET = 256
_ROPE_SCALE_OFFSET = 288
_PAD_OFFSET = 292
_ROPE_OFFSET = 304
_PAGE_SIZE = 64
_HEADS = 64
_HEAD_DIM = 576
_V_HEAD_DIM = 512
_SENTINEL = 0xA5


@pytest.fixture(scope="module", autouse=True)
def _isolate_writer_kernel_cache():
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache()
    yield
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache()


def _valid_cpu_writer_args() -> list[torch.Tensor]:
    return [
        torch.empty((2, _V_HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((2, _HEAD_DIM - _V_HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((2, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.uint8),
        torch.arange(2, dtype=torch.int64),
    ]


def _invalid_writer_args(case: str) -> tuple[torch.Tensor, ...]:
    args = _valid_cpu_writer_args()
    if case == "kv_c_shape":
        args[0] = torch.empty((2, _V_HEAD_DIM - 1), dtype=torch.bfloat16)
    elif case == "k_pe_shape":
        args[1] = torch.empty((2, _HEAD_DIM - _V_HEAD_DIM - 1), dtype=torch.bfloat16)
    elif case == "kv_c_dtype":
        args[0] = torch.empty((2, _V_HEAD_DIM), dtype=torch.float32)
    elif case == "k_pe_dtype":
        args[1] = torch.empty((2, _HEAD_DIM - _V_HEAD_DIM), dtype=torch.float16)
    elif case == "cache_shape":
        args[2] = torch.empty((2, _PAGE_SIZE, _RECORD_BYTES - 1), dtype=torch.uint8)
    elif case == "cache_dtype":
        args[2] = torch.empty((2, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.bfloat16)
    elif case == "zero_num_blocks":
        args[2] = torch.empty((0, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.uint8)
    elif case == "zero_block_size":
        args[2] = torch.empty((2, 0, _RECORD_BYTES), dtype=torch.uint8)
    elif case == "slot_dtype":
        args[3] = torch.arange(2, dtype=torch.int32)
    elif case == "slot_layout":
        args[3] = torch.arange(4, dtype=torch.int64)[::2]
    elif case == "short_source":
        args[0] = args[0][:1]
    elif case == "kv_c_inner_layout":
        args[0] = torch.empty((2, 2 * _V_HEAD_DIM), dtype=torch.bfloat16)[:, ::2]
    elif case == "cache_inner_layout":
        args[2] = torch.empty((2, _PAGE_SIZE, 2 * _RECORD_BYTES), dtype=torch.uint8)[
            ..., ::2
        ]
    elif case == "kv_c_row_alignment":
        args[0] = torch.empty((2, _V_HEAD_DIM + 1), dtype=torch.bfloat16)[
            :, :_V_HEAD_DIM
        ]
    elif case == "k_pe_row_alignment":
        args[1] = torch.empty((2, _HEAD_DIM - _V_HEAD_DIM + 1), dtype=torch.bfloat16)[
            :, : _HEAD_DIM - _V_HEAD_DIM
        ]
    elif case == "cache_record_alignment":
        args[2] = torch.empty((2, _PAGE_SIZE, _RECORD_BYTES + 1), dtype=torch.uint8)[
            ..., :_RECORD_BYTES
        ]
    elif case == "cache_base_alignment":
        numel = 2 * _PAGE_SIZE * _RECORD_BYTES
        args[2] = torch.empty(numel + 8, dtype=torch.uint8)[8:].view(
            2, _PAGE_SIZE, _RECORD_BYTES
        )
    elif case != "cpu_device":
        raise AssertionError(f"unknown invalid writer case {case}")
    return tuple(args)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("kv_c_shape", ValueError, r"kv_c must be \(num_tokens, 512\)"),
        ("k_pe_shape", ValueError, r"k_pe must be \(num_tokens, 64\)"),
        ("kv_c_dtype", TypeError, "kv_c must be bf16/f16"),
        ("k_pe_dtype", TypeError, "must match kv_c dtype"),
        ("cache_shape", ValueError, "kv_cache must be"),
        ("cache_dtype", TypeError, "kv_cache must be uint8"),
        ("zero_num_blocks", ValueError, "num_blocks.*positive"),
        ("zero_block_size", ValueError, "block_size.*positive"),
        ("slot_dtype", TypeError, "slot_mapping must be a 1-D int64 tensor"),
        ("slot_layout", ValueError, "slot_mapping must be contiguous"),
        ("short_source", ValueError, "must cover slot_mapping"),
        ("kv_c_inner_layout", ValueError, "must be innermost-contiguous"),
        ("cache_inner_layout", ValueError, "must be innermost-contiguous"),
        ("kv_c_row_alignment", ValueError, "must be 4-byte aligned"),
        ("k_pe_row_alignment", ValueError, "k_pe.*4-byte aligned"),
        ("cache_record_alignment", ValueError, "must be 16-byte aligned"),
        ("cache_base_alignment", ValueError, "must be 16-byte aligned"),
        ("cpu_device", ValueError, "all tensors must be on CUDA"),
    ],
)
def test_writer_rejects_invalid_dtype_shape_device_and_layout(
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        concat_and_cache_nvfp4_mla_fp8_rope(*_invalid_writer_args(case))


def _make_exactly_quantizable_inputs(
    *,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    e2m1_values = torch.tensor(
        [
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            2.0,
            -2.0,
        ],
        dtype=torch.float32,
    )
    e2m1_codes = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 4, 12],
        dtype=torch.uint8,
    )
    group_scales = torch.empty((num_tokens, 32), dtype=torch.float32)
    nope = torch.empty((num_tokens, 32, 16), dtype=torch.float32)
    codes = torch.empty((num_tokens, 32, 16), dtype=torch.uint8)
    for token in range(num_tokens):
        for group in range(32):
            scale = 2.0 ** ((token + group) % 4 - 2)
            shift = (5 * token + 3 * group) % 16
            group_scales[token, group] = scale
            nope[token, group] = torch.roll(e2m1_values, shifts=shift) * scale
            codes[token, group] = torch.roll(e2m1_codes, shifts=shift)

    packed_nope = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(
        num_tokens, _NOPE_BYTES
    )

    e4m3_values = torch.tensor(
        [
            448.0,
            -448.0,
            416.0,
            -416.0,
            240.0,
            -240.0,
            120.0,
            -120.0,
            6.0,
            -6.0,
            1.5,
            -1.5,
            0.5,
            -0.5,
            0.0,
            2.0,
        ],
        dtype=torch.float32,
    ).repeat(4)
    rope_scales = torch.tensor(
        [2.0 ** (token - 4) for token in range(num_tokens)],
        dtype=torch.float32,
    )
    rope = torch.stack(
        [
            torch.roll(e4m3_values, shifts=7 * token) * rope_scales[token]
            for token in range(num_tokens)
        ]
    )
    return (
        nope.reshape(num_tokens, _V_HEAD_DIM).to(device=device, dtype=dtype),
        rope.to(device=device, dtype=dtype),
        packed_nope.to(device=device),
        group_scales.to(device=device),
        rope_scales.to(device=device),
    )


def _dequantize_records(
    records: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = records[:, :_NOPE_BYTES]
    codes = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=-1).reshape(
        records.shape[0], _V_HEAD_DIM
    )
    e2m1 = torch.tensor(E2M1_TO_FLOAT32, dtype=torch.float32, device=records.device)
    group_scales = (
        records[:, _GROUP_SCALES_OFFSET:_ROPE_SCALE_OFFSET]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    nope = (
        e2m1[codes.long()].reshape(records.shape[0], 32, 16)
        * group_scales.unsqueeze(-1)
    ).reshape(records.shape[0], _V_HEAD_DIM)

    rope_scales = (
        records[:, _ROPE_SCALE_OFFSET:_PAD_OFFSET]
        .contiguous()
        .view(torch.float32)
        .reshape(-1)
    )
    rope_q = (
        records[:, _ROPE_OFFSET:_RECORD_BYTES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    rope = rope_q * rope_scales.unsqueeze(-1)
    return nope, group_scales, rope, rope_scales


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_writer_preserves_skipped_slots_and_writes_the_record_abi(
    dtype: torch.dtype,
) -> None:
    device = require_sm120()
    num_tokens = 6
    kv_c, k_pe, expected_packed, expected_group_scales, expected_rope_scales = (
        _make_exactly_quantizable_inputs(
            num_tokens=num_tokens,
            dtype=dtype,
            device=device,
        )
    )

    # The cache view has a full guard page on each side. The capacity slot
    # targets the first trailing guard record without an upper-bound check. The
    # huge slot's low 32 bits name an otherwise untouched in-cache record, so a
    # check performed only after Int32 narrowing is also observable.
    backing = torch.full(
        (5, _PAGE_SIZE, _RECORD_BYTES),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    cache = backing[1:4]
    capacity = cache.shape[0] * cache.shape[1]
    slot_mapping = torch.tensor(
        [0, -1, capacity, 65, 2**40 + 17, 130],
        dtype=torch.int64,
        device=device,
    )
    if dtype == torch.float16:
        concat_and_cache_nvfp4_mla_fp8_rope(
            kv_c,
            k_pe,
            cache,
            slot_mapping,
            scale=torch.tensor([37.0], dtype=torch.float32, device=device),
        )
    else:
        concat_and_cache_nvfp4_mla_fp8_rope(kv_c, k_pe, cache, slot_mapping)
    torch.cuda.synchronize(device)

    changed_records = (
        (backing != _SENTINEL)
        .any(dim=-1)
        .reshape(-1)
        .nonzero(as_tuple=False)
        .reshape(-1)
    )
    expected_changed = torch.tensor(
        [_PAGE_SIZE, 2 * _PAGE_SIZE + 1, 3 * _PAGE_SIZE + 2],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(changed_records, expected_changed)

    live_slots = torch.tensor([0, 65, 130], dtype=torch.int64, device=device)
    live_tokens = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    records = cache.reshape(-1, _RECORD_BYTES).index_select(0, live_slots)

    # These byte ranges are the public record ABI consumed by the real readers.
    assert torch.equal(
        records[:, :_NOPE_BYTES], expected_packed.index_select(0, live_tokens)
    )
    assert torch.count_nonzero(records[:, _PAD_OFFSET:_ROPE_OFFSET]).item() == 0

    nope, group_scales, rope, rope_scales = _dequantize_records(records)
    torch.testing.assert_close(
        group_scales,
        expected_group_scales.index_select(0, live_tokens),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        rope_scales,
        expected_rope_scales.index_select(0, live_tokens),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        nope,
        kv_c.index_select(0, live_tokens).float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        rope,
        k_pe.index_select(0, live_tokens).float(),
        rtol=0.0,
        atol=1e-5,
    )


def _make_written_reader_case(
    *,
    rows: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kv_c = torch.randn(
        (topk, _V_HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(
        (topk, _HEAD_DIM - _V_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    q = torch.randn(
        (rows, _HEADS, _HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(topk, generator=generator) for _ in range(rows)]
    ).to(device=device, dtype=torch.int32)

    num_blocks = (topk + _PAGE_SIZE - 1) // _PAGE_SIZE
    cache = torch.full(
        (num_blocks, _PAGE_SIZE, _RECORD_BYTES),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(topk, dtype=torch.int64, device=device)
    concat_and_cache_nvfp4_mla_fp8_rope(kv_c, k_pe, cache, slots)
    return q, cache, indices


def _reference_attention_from_records(
    *,
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    nope, _, rope, _ = _dequantize_records(cache.reshape(-1, _RECORD_BYTES))
    keys = torch.cat((nope, rope), dim=-1)
    rows = []
    for row, length in enumerate(lengths.cpu().tolist()):
        selected = indices[row, :length].long()
        selected_keys = keys.index_select(0, selected)
        selected_values = nope.index_select(0, selected)
        scores = q[row].float() @ selected_keys.T
        probabilities = torch.softmax(scores * sm_scale, dim=-1)
        rows.append(probabilities @ selected_values)
    return torch.stack(rows)


def _assert_reader_matches_dequantized_records(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_f = actual.float()
    max_abs = (actual_f - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), expected.reshape(-1), dim=0
    ).item()
    assert max_abs < 0.03, f"max absolute error {max_abs:.8f} exceeded 0.03"
    assert cosine > 0.999, f"cosine similarity {cosine:.8f} did not exceed 0.999"


@torch.inference_mode()
def test_writer_records_feed_production_head_multisplit_decode() -> None:
    device = require_sm120()
    topk = 129
    q, cache, indices = _make_written_reader_case(
        rows=1,
        topk=topk,
        seed=1301,
        device=device,
    )
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    sm_scale = _HEAD_DIM**-0.5

    caps = SM12XSparseMLAScratchCaps(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_HEADS,
        max_q_rows=1,
        max_batch=1,
        max_width=topk,
        max_kv_rows=topk,
        head_dim=_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        max_chunks_per_row=8,
        page_size=_PAGE_SIZE,
    )
    plan = plan_sparse_mla_scratch(caps)
    (scratch_spec,) = plan.scratch_specs()
    scratch_storage = torch.zeros(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    binding = plan.bind(
        scratch=scratch_storage,
        q=q,
        selected_indices=indices,
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=lengths,
    )

    actual = run_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=indices,
        swa_topk_lengths=lengths,
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=_PAGE_SIZE,
        forced_num_splits=2,
        scale_format_override=ScaleFormat.NVFP4_E4M3,
        fp8_rope_override=None,
    )
    expected = _reference_attention_from_records(
        q=q,
        cache=cache,
        indices=indices,
        lengths=lengths,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize(device)
    _assert_reader_matches_dequantized_records(actual, expected)


@torch.inference_mode()
def test_writer_records_feed_production_head_multitile_prefill_mg() -> None:
    device = require_sm120()
    topk = 129
    q, cache, indices = _make_written_reader_case(
        rows=2,
        topk=topk,
        seed=2302,
        device=device,
    )
    lengths = torch.tensor([topk, 65], dtype=torch.int32, device=device)
    sm_scale = _HEAD_DIM**-0.5

    actual, _ = run_unified_prefill_mg(
        q=q,
        kv_cache=cache,
        topk_indices=indices,
        topk_length=lengths,
        sm_scale=sm_scale,
        page_block_size=_PAGE_SIZE,
        compute_mode=ComputeMode.BF16,
        mg_n_hg=2,
        model_type=ModelType.GLM_NSA,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=True,
    )
    expected = _reference_attention_from_records(
        q=q,
        cache=cache,
        indices=indices,
        lengths=lengths,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize(device)
    _assert_reader_matches_dequantized_records(actual, expected)
