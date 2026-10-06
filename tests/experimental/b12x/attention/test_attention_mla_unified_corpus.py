from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_HEAD_DIM,
    COMPRESSED_SPARSE_MLA_NOPE_DIM,
    COMPRESSED_SPARSE_MLA_ROPE_DIM,
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.kernel import run_unified_decode, run_unified_prefill
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)
from b12x.attention._shared.mla.traits import ComputeMode, ScaleFormat
from b12x._lib.intrinsics import pack_grouped_fp4_values
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps,
    _compressed_sparse_mla_scratch_layout,
    _materialize_compressed_sparse_mla_scratch,
)

from b12x.testing.reference.helpers import (
    dequantize_token_major_nvfp4,
    ref_fp4_quant,
    require_b12x,
)


_PAGE_SIZE = 64
_SM_SCALE = 1.0 / math.sqrt(COMPRESSED_SPARSE_MLA_HEAD_DIM)
_GLM_Q_DIM = 576
_GLM_V_DIM = 512
_GLM_SM_SCALE = 1.0 / math.sqrt(_GLM_Q_DIM)
_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


@dataclass(frozen=True)
class _UnifiedInputs:
    q: torch.Tensor
    q_scenarios: tuple[torch.Tensor, torch.Tensor]
    main_cache: torch.Tensor
    main_indices: torch.Tensor
    main_index_scenarios: tuple[torch.Tensor, torch.Tensor]
    main_lengths: torch.Tensor | None
    main_length_scenarios: tuple[torch.Tensor, torch.Tensor]
    extra_cache: torch.Tensor | None
    extra_indices: torch.Tensor | None
    extra_index_scenarios: tuple[torch.Tensor, torch.Tensor] | None
    extra_lengths: torch.Tensor | None
    extra_length_scenarios: tuple[torch.Tensor, torch.Tensor] | None
    extra_page_size: int | None


def _make_cache(
    *,
    tokens: int,
    page_size: int = _PAGE_SIZE,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    k_nope = torch.empty(
        (tokens, COMPRESSED_SPARSE_MLA_NOPE_DIM),
        dtype=torch.float32,
        device=device,
    ).normal_(mean=0.0, std=0.35, generator=generator)
    k_rope = torch.empty(
        (tokens, COMPRESSED_SPARSE_MLA_ROPE_DIM),
        dtype=torch.float32,
        device=device,
    ).normal_(mean=0.0, std=0.35, generator=generator)
    return pack_compressed_sparse_mla_kv_cache_reference(
        k_nope,
        k_rope,
        page_size=page_size,
    )


def _make_inputs(
    *,
    rows: int,
    heads: int,
    main_width: int,
    extra_width: int,
    extra_page_size: int = _PAGE_SIZE,
    per_token: bool,
    device: torch.device,
) -> _UnifiedInputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(46_120 + rows * 100 + heads)

    q_a = torch.empty(
        (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    q_b = torch.empty_like(q_a).normal_(
        mean=0.05,
        std=0.25,
        generator=generator,
    )
    q = torch.empty_like(q_a)

    main_tokens = max(main_width, _PAGE_SIZE)
    main_cache = _make_cache(
        tokens=main_tokens,
        device=device,
        generator=generator,
    )
    main_indices_a = torch.arange(
        main_width,
        dtype=torch.int32,
        device=device,
    ).repeat(rows, 1)
    if rows > 1:
        main_indices_a[1] = torch.roll(main_indices_a[1], shifts=7)
    main_indices_b = torch.roll(main_indices_a, shifts=19, dims=1)
    if rows > 1:
        main_indices_b[1] = torch.flip(main_indices_a[1], dims=(0,))
    main_indices = torch.empty_like(main_indices_a)

    main_lengths_a = torch.full((rows,), main_width, dtype=torch.int32, device=device)
    main_lengths_b = main_lengths_a.clone()
    if rows > 1:
        main_lengths_a[1] = max(1, main_width - 29)
        main_lengths_b[0] = max(1, main_width // 3)
    main_lengths = torch.empty_like(main_lengths_a) if per_token else None

    extra_cache: torch.Tensor | None = None
    extra_indices: torch.Tensor | None = None
    extra_index_scenarios: tuple[torch.Tensor, torch.Tensor] | None = None
    extra_lengths: torch.Tensor | None = None
    extra_length_scenarios: tuple[torch.Tensor, torch.Tensor] | None = None
    if extra_width:
        extra_tokens = max(extra_width, extra_page_size)
        extra_cache = _make_cache(
            tokens=extra_tokens,
            page_size=extra_page_size,
            device=device,
            generator=generator,
        )
        extra_indices_a = torch.arange(
            extra_width,
            dtype=torch.int32,
            device=device,
        ).repeat(rows, 1)
        if rows > 1:
            extra_indices_a[0] = torch.roll(extra_indices_a[0], shifts=11)
        extra_indices_b = torch.roll(extra_indices_a, shifts=23, dims=1)
        if rows > 1:
            extra_indices_b[1] = torch.flip(extra_indices_a[1], dims=(0,))
        extra_indices = torch.empty_like(extra_indices_a)
        extra_index_scenarios = (extra_indices_a, extra_indices_b)
        extra_lengths_a = torch.full(
            (rows,), extra_width, dtype=torch.int32, device=device
        )
        extra_lengths_b = extra_lengths_a.clone()
        if rows > 1:
            extra_lengths_a[0] = max(1, extra_width // 2)
            extra_lengths_b[1] = max(1, extra_width - 17)
        extra_length_scenarios = (extra_lengths_a, extra_lengths_b)
        extra_lengths = torch.empty_like(extra_lengths_a) if per_token else None

    return _UnifiedInputs(
        q=q,
        q_scenarios=(q_a, q_b),
        main_cache=main_cache,
        main_indices=main_indices,
        main_index_scenarios=(main_indices_a, main_indices_b),
        main_lengths=main_lengths,
        main_length_scenarios=(main_lengths_a, main_lengths_b),
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_index_scenarios=extra_index_scenarios,
        extra_lengths=extra_lengths,
        extra_length_scenarios=extra_length_scenarios,
        extra_page_size=extra_page_size if extra_width else None,
    )


def _install_scenario(inputs: _UnifiedInputs, scenario: int) -> None:
    inputs.q.copy_(inputs.q_scenarios[scenario])
    inputs.main_indices.copy_(inputs.main_index_scenarios[scenario])
    if inputs.main_lengths is not None:
        inputs.main_lengths.copy_(inputs.main_length_scenarios[scenario])
    if inputs.extra_indices is not None:
        assert inputs.extra_index_scenarios is not None
        inputs.extra_indices.copy_(inputs.extra_index_scenarios[scenario])
    if inputs.extra_lengths is not None:
        assert inputs.extra_length_scenarios is not None
        inputs.extra_lengths.copy_(inputs.extra_length_scenarios[scenario])


def _reference(
    inputs: _UnifiedInputs,
    scenario: int,
    *,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.main_lengths is None:
        main_lengths = torch.full_like(
            inputs.main_length_scenarios[scenario],
            inputs.main_indices.shape[1],
        )
    else:
        main_lengths = inputs.main_length_scenarios[scenario]
    extra_lengths = None
    if inputs.extra_cache is not None:
        assert inputs.extra_indices is not None
        assert inputs.extra_length_scenarios is not None
        if inputs.extra_lengths is None:
            extra_lengths = torch.full_like(
                inputs.extra_length_scenarios[scenario],
                inputs.extra_indices.shape[1],
            )
        else:
            extra_lengths = inputs.extra_length_scenarios[scenario]
    result = compressed_sparse_mla_reference(
        inputs.q_scenarios[scenario],
        inputs.main_cache,
        inputs.main_index_scenarios[scenario],
        main_lengths,
        sm_scale=_SM_SCALE,
        attn_sink=attn_sink,
        extra_k_cache=inputs.extra_cache,
        extra_indices=(
            None
            if inputs.extra_index_scenarios is None
            else inputs.extra_index_scenarios[scenario]
        ),
        extra_topk_lengths=extra_lengths,
        swa_page_size=_PAGE_SIZE,
        extra_page_size=inputs.extra_page_size,
        return_lse=True,
    )
    assert isinstance(result, tuple)
    return result


def _assert_output(
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    got_fp32 = got.float()
    expected_fp32 = expected.float()
    assert torch.isfinite(got_fp32).all(), f"{label}: non-finite output"
    assert torch.count_nonzero(got_fp32).item() > 0, f"{label}: zero output"
    cosine = torch.nn.functional.cosine_similarity(
        got_fp32.flatten(),
        expected_fp32.flatten(),
        dim=0,
    ).item()
    assert cosine > 0.998, f"{label}: cosine={cosine}"
    torch.testing.assert_close(
        got_fp32,
        expected_fp32,
        atol=3.0e-2,
        rtol=3.0e-2,
    )


def _make_decode_workspace(
    *,
    rows: int,
    heads: int,
    width: int,
    device: torch.device,
):
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=width,
        head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
        v_head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
        max_chunks_per_row=8,
        page_size=_PAGE_SIZE,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(layout.nbytes, dtype=torch.uint8, device=device)
    return _materialize_compressed_sparse_mla_scratch(caps, storage, layout)


@torch.inference_mode()
@pytest.mark.parametrize(
    "entrypoint",
    ["main", "main-per-token", "extra", "extra-per-token"],
)
def test_unified_decode_entrypoint_live_graph_oracle(entrypoint: str) -> None:
    """Cover all four active UnifiedDecodeKernel @cute.kernel entrypoints."""
    device = require_b12x()
    rows, heads, main_width = 2, 8, 64
    has_extra = entrypoint.startswith("extra")
    per_token = entrypoint.endswith("per-token")
    extra_width = 64 if has_extra else 0
    inputs = _make_inputs(
        rows=rows,
        heads=heads,
        main_width=main_width,
        extra_width=extra_width,
        per_token=per_token,
        device=device,
    )
    workspace = _make_decode_workspace(
        rows=rows,
        heads=heads,
        width=main_width + extra_width,
        device=device,
    )
    output = torch.empty(
        (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    expected = tuple(_reference(inputs, scenario)[0] for scenario in range(2))
    assert not torch.allclose(expected[0], expected[1])

    def launch() -> None:
        run_unified_decode(
            q_all=inputs.q,
            swa_k_cache=inputs.main_cache,
            swa_indices=inputs.main_indices,
            swa_topk_lengths=inputs.main_lengths,
            workspace=workspace,
            sm_scale=_SM_SCALE,
            swa_page_size=_PAGE_SIZE,
            indexed_k_cache=inputs.extra_cache,
            indexed_indices=inputs.extra_indices,
            indexed_topk_lengths=inputs.extra_lengths,
            indexed_page_size=inputs.extra_page_size,
            forced_num_splits=1,
            out=output,
        )

    _install_scenario(inputs, 0)
    launch()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for scenario in range(2):
        _install_scenario(inputs, scenario)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_output(
            output,
            expected[scenario],
            label=f"decode {entrypoint} replay {scenario}",
        )


@torch.inference_mode()
@pytest.mark.parametrize("entrypoint", ["main", "dual"])
def test_unified_prefill_entrypoint_live_graph_oracle(entrypoint: str) -> None:
    """Cover both active UnifiedPrefillMGKernel @cute.kernel entrypoints."""
    device = require_b12x()
    rows, heads, main_width = 2, 16, 128
    has_extra = entrypoint == "dual"
    extra_width = 64 if has_extra else 0
    inputs = _make_inputs(
        rows=rows,
        heads=heads,
        main_width=main_width,
        extra_width=extra_width,
        per_token=True,
        device=device,
    )
    output = torch.empty(
        (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    lse_base2 = torch.empty((rows, heads), dtype=torch.float32, device=device)
    # Supplying caller-owned sink storage avoids the launcher's no-sink
    # placeholder allocation and also exercises live sink normalization.
    attn_sink = torch.linspace(
        -0.8,
        0.6,
        heads,
        dtype=torch.float32,
        device=device,
    )
    expected = tuple(
        _reference(inputs, scenario, attn_sink=attn_sink) for scenario in range(2)
    )
    assert not torch.allclose(expected[0][0], expected[1][0])

    def launch() -> None:
        run_unified_prefill(
            q=inputs.q,
            kv_cache=inputs.main_cache,
            topk_indices=inputs.main_indices,
            topk_length=inputs.main_lengths,
            sm_scale=_SM_SCALE,
            page_block_size=_PAGE_SIZE,
            attn_sink=attn_sink,
            output=output,
            lse_out=lse_base2,
            extra_kv_cache=inputs.extra_cache,
            extra_indices=inputs.extra_indices,
            extra_topk_length=inputs.extra_lengths,
            extra_page_block_size=inputs.extra_page_size,
        )

    _install_scenario(inputs, 0)
    launch()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for scenario in range(2):
        _install_scenario(inputs, scenario)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_output(
            output,
            expected[scenario][0],
            label=f"prefill {entrypoint} replay {scenario}",
        )
        assert torch.isfinite(lse_base2).all()
        torch.testing.assert_close(
            lse_base2,
            expected[scenario][1] / math.log(2.0),
            atol=6.0e-2,
            rtol=2.0e-2,
        )


@dataclass(frozen=True)
class _GlmInputs:
    q: torch.Tensor
    q_scenarios: tuple[torch.Tensor, torch.Tensor]
    packed_tokens: torch.Tensor
    launch_cache: torch.Tensor
    indices: torch.Tensor
    index_scenarios: tuple[torch.Tensor, torch.Tensor]
    lengths: torch.Tensor | None
    length_scenarios: tuple[torch.Tensor, torch.Tensor]


def _make_glm_inputs(
    *,
    rows: int,
    heads: int,
    width: int,
    per_token: bool,
    device: torch.device,
) -> _GlmInputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(46_576 + rows * 100 + heads)
    q_a = torch.empty(
        (rows, heads, _GLM_Q_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    q_b = torch.empty_like(q_a).normal_(
        mean=-0.04,
        std=0.24,
        generator=generator,
    )
    q = torch.empty_like(q_a)
    token_count = math.ceil(width / _PAGE_SIZE) * _PAGE_SIZE
    k_nope = torch.empty(
        (token_count, _GLM_V_DIM),
        dtype=torch.float32,
        device=device,
    ).normal_(mean=0.0, std=0.3, generator=generator)
    k_rope = torch.empty(
        (token_count, _GLM_Q_DIM - _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.3, generator=generator)
    packed_tokens = pack_mla_kv_cache_reference(k_nope, k_rope)
    launch_cache = packed_tokens.view(
        token_count // _PAGE_SIZE,
        _PAGE_SIZE,
        packed_tokens.shape[-1],
    )
    indices_a = torch.arange(width, dtype=torch.int32, device=device).repeat(rows, 1)
    if rows > 1:
        indices_a[1] = torch.roll(indices_a[1], shifts=13)
    indices_b = torch.roll(indices_a, shifts=29, dims=1)
    if rows > 1:
        indices_b[1] = torch.flip(indices_a[1], dims=(0,))
    indices = torch.empty_like(indices_a)
    lengths_a = torch.full((rows,), width, dtype=torch.int32, device=device)
    lengths_b = lengths_a.clone()
    if rows > 1:
        lengths_a[1] = max(1, width - 73)
        lengths_b[0] = max(1, width // 2 + 7)
    lengths = torch.empty_like(lengths_a) if per_token else None
    return _GlmInputs(
        q=q,
        q_scenarios=(q_a, q_b),
        packed_tokens=packed_tokens,
        launch_cache=launch_cache,
        indices=indices,
        index_scenarios=(indices_a, indices_b),
        lengths=lengths,
        length_scenarios=(lengths_a, lengths_b),
    )


def _glm_reference(
    inputs: _GlmInputs,
    scenario: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.lengths is None:
        active_token_counts = torch.full_like(
            inputs.length_scenarios[scenario],
            inputs.indices.shape[1],
        )
    else:
        active_token_counts = inputs.length_scenarios[scenario]
    result = sparse_mla_reference(
        q_all=inputs.q_scenarios[scenario],
        kv_cache=inputs.packed_tokens,
        page_table_1=inputs.index_scenarios[scenario],
        active_token_counts=active_token_counts,
        sm_scale=_GLM_SM_SCALE,
        v_head_dim=_GLM_V_DIM,
        return_lse=True,
    )
    assert isinstance(result, tuple)
    return result


def _install_glm_scenario(inputs: _GlmInputs, scenario: int) -> None:
    inputs.q.copy_(inputs.q_scenarios[scenario])
    inputs.indices.copy_(inputs.index_scenarios[scenario])
    if inputs.lengths is not None:
        inputs.lengths.copy_(inputs.length_scenarios[scenario])


@dataclass(frozen=True)
class _Nvfp4GlmInputs:
    q: torch.Tensor
    q_scenarios: tuple[torch.Tensor, torch.Tensor]
    launch_cache: torch.Tensor
    dequant_nope: torch.Tensor
    rope: torch.Tensor
    indices: torch.Tensor
    index_scenarios: tuple[torch.Tensor, torch.Tensor]
    lengths: torch.Tensor
    length_scenarios: tuple[torch.Tensor, torch.Tensor]


def _make_nvfp4_glm_inputs(
    *,
    rows: int,
    heads: int,
    width: int,
    device: torch.device,
) -> _Nvfp4GlmInputs:
    """Build the production 432-byte NVFP4 MLA record from generic helpers."""
    generator = torch.Generator(device=device)
    generator.manual_seed(46_432 + rows * 100 + heads)
    q_a = torch.empty(
        (rows, heads, _GLM_Q_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.2, generator=generator)
    q_b = torch.empty_like(q_a).normal_(
        mean=0.03,
        std=0.23,
        generator=generator,
    )
    q = torch.empty_like(q_a)

    token_count = math.ceil(width / _PAGE_SIZE) * _PAGE_SIZE
    latent = torch.empty(
        (token_count, _GLM_V_DIM),
        dtype=torch.float32,
        device=device,
    ).normal_(mean=0.0, std=0.3, generator=generator)
    rope = torch.empty(
        (token_count, _GLM_Q_DIM - _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.3, generator=generator)

    # The attention ABI stores 512 E2M1 values as 256 packed bytes followed by
    # 32 row-major E4M3 group-16 scales, 16 bytes of padding, and 64 BF16 RoPE
    # values.  A unit outer scale makes the generic test quantizer's dequant
    # exactly the kernel's ``fp4 * inline_scale * latent_scale`` contract.
    quantized, scales = ref_fp4_quant(latent, 1.0, block_size=16)
    packed_fp4 = pack_grouped_fp4_values(quantized.unsqueeze(0)).squeeze(-1)
    scales_fp8 = scales.to(torch.float8_e4m3fn)
    padding = torch.zeros((token_count, 16), dtype=torch.uint8, device=device)
    records = torch.cat(
        (
            packed_fp4,
            scales_fp8.view(torch.uint8),
            padding,
            rope.view(torch.uint8).reshape(token_count, 128),
        ),
        dim=1,
    ).contiguous()
    assert records.shape == (token_count, 432)
    launch_cache = records.view(
        token_count // _PAGE_SIZE,
        _PAGE_SIZE,
        432,
    )
    dequant_nope = dequantize_token_major_nvfp4(
        packed_fp4,
        scales_fp8,
        hidden_size=_GLM_V_DIM,
        global_scale=torch.ones((1,), dtype=torch.float32, device=device),
    )

    indices_a = torch.arange(width, dtype=torch.int32, device=device).repeat(rows, 1)
    if rows > 1:
        indices_a[1] = torch.roll(indices_a[1], shifts=17)
    indices_b = torch.roll(indices_a, shifts=31, dims=1)
    if rows > 1:
        indices_b[1] = torch.flip(indices_a[1], dims=(0,))
    indices = torch.empty_like(indices_a)
    lengths_a = torch.full((rows,), width, dtype=torch.int32, device=device)
    lengths_b = lengths_a.clone()
    if rows > 1:
        lengths_a[1] = max(1, width - 61)
        lengths_b[0] = max(1, width // 2 + 11)
    lengths = torch.empty_like(lengths_a)
    return _Nvfp4GlmInputs(
        q=q,
        q_scenarios=(q_a, q_b),
        launch_cache=launch_cache,
        dequant_nope=dequant_nope,
        rope=rope,
        indices=indices,
        index_scenarios=(indices_a, indices_b),
        lengths=lengths,
        length_scenarios=(lengths_a, lengths_b),
    )


def _nvfp4_glm_reference(
    inputs: _Nvfp4GlmInputs,
    scenario: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent GPU attention over the generic-helper dequantized record."""
    q = inputs.q_scenarios[scenario].float()
    k_all = torch.cat((inputs.dequant_nope, inputs.rope.float()), dim=1)
    output = torch.empty(
        (q.shape[0], q.shape[1], _GLM_V_DIM),
        dtype=torch.float32,
        device=q.device,
    )
    lse_base2 = torch.empty(
        (q.shape[0], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    for row in range(int(q.shape[0])):
        length = int(inputs.length_scenarios[scenario][row].item())
        selected = inputs.index_scenarios[scenario][row, :length].to(torch.int64)
        assert bool((selected >= 0).all())
        k_selected = k_all.index_select(0, selected)
        v_selected = inputs.dequant_nope.index_select(0, selected)
        scores = torch.matmul(q[row], k_selected.t()) * _GLM_SM_SCALE
        row_max = scores.amax(dim=-1, keepdim=True)
        weights = torch.exp(scores - row_max)
        denom = weights.sum(dim=-1, keepdim=True)
        output[row] = torch.matmul(weights, v_selected) / denom
        lse_base2[row] = (
            row_max.squeeze(-1) + torch.log(denom.squeeze(-1))
        ) / math.log(2.0)
    return output.to(torch.bfloat16), lse_base2


def _install_nvfp4_glm_scenario(
    inputs: _Nvfp4GlmInputs,
    scenario: int,
) -> None:
    inputs.q.copy_(inputs.q_scenarios[scenario])
    inputs.indices.copy_(inputs.index_scenarios[scenario])
    inputs.lengths.copy_(inputs.length_scenarios[scenario])


@dataclass(frozen=True)
class _MGPrefillServingCase:
    family: str
    compute: str
    heads: int
    topk: int
    n_hg: int

    @property
    def test_id(self) -> str:
        return (
            f"{self.family}-{self.compute}-hg{self.n_hg}-h{self.heads}-topk{self.topk}"
        )


_MG_PREFILL_SERVING_CASES = (
    _MGPrefillServingCase("dsv4", "fp8", 16, 512, 1),
    _MGPrefillServingCase("dsv4", "fp8", 32, 512, 2),
    _MGPrefillServingCase("dsv4-dual", "bf16", 16, 128, 1),
    _MGPrefillServingCase("dsv4-dual", "bf16", 32, 128, 2),
    _MGPrefillServingCase("dsv4-dual", "bf16", 16, 512, 1),
    _MGPrefillServingCase("dsv4-dual", "bf16", 32, 512, 2),
    _MGPrefillServingCase("dsv4-dual", "fp8", 16, 1024, 1),
    _MGPrefillServingCase("dsv4-dual", "fp8", 32, 1024, 2),
    _MGPrefillServingCase("dsv4-dual", "fp8", 16, 2048, 1),
    _MGPrefillServingCase("dsv4-dual", "fp8", 32, 2048, 2),
    _MGPrefillServingCase("dsv4", "bf16", 16, 128, 1),
    _MGPrefillServingCase("dsv4", "bf16", 32, 128, 2),
    _MGPrefillServingCase("glm", "fp8", 16, 512, 1),
    _MGPrefillServingCase("glm", "fp8", 32, 512, 2),
    _MGPrefillServingCase("glm-nvfp4", "bf16", 16, 512, 1),
    _MGPrefillServingCase("glm-nvfp4", "bf16", 32, 512, 2),
)


def _allocator_counters(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {name: int(stats.get(name, 0)) for name in _ALLOCATOR_COUNTERS}


def _poison_inactive_topk_tails(
    index_scenarios: tuple[torch.Tensor, torch.Tensor],
    length_scenarios: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Put an invalid sentinel immediately beyond every live prefix boundary."""
    for indices, lengths in zip(index_scenarios, length_scenarios, strict=True):
        width = int(indices.shape[1])
        for row in range(int(indices.shape[0])):
            length = int(lengths[row].item())
            assert 0 < length <= width
            if length < width:
                indices[row, length:].fill_(-1)


def _assert_live_topk_contract(
    *,
    live_indices: torch.Tensor,
    live_lengths: torch.Tensor,
    expected_indices: torch.Tensor,
    expected_lengths: torch.Tensor,
    topk: int,
) -> None:
    assert live_indices.shape == (2, topk)
    assert live_lengths.shape == (2,)
    assert torch.equal(live_indices, expected_indices)
    assert torch.equal(live_lengths, expected_lengths)
    for row in range(2):
        length = int(live_lengths[row].item())
        assert 0 < length <= topk
        active = live_indices[row, :length]
        assert bool((active >= 0).all())
        assert bool((active < topk).all())
        if length < topk:
            assert int(live_indices[row, length].item()) == -1


def _assert_prefill_boundary_heads(
    got: torch.Tensor,
    expected: torch.Tensor,
    *,
    n_hg: int,
) -> None:
    heads = int(got.shape[1])
    boundary_heads = [0, heads - 1]
    if n_hg == 2:
        boundary_heads.extend((15, 16))
    head_ids = torch.tensor(
        sorted(set(boundary_heads)),
        dtype=torch.int64,
        device=got.device,
    )
    torch.testing.assert_close(
        got.index_select(1, head_ids).float(),
        expected.index_select(1, head_ids).float(),
        atol=3.0e-2,
        rtol=3.0e-2,
    )


@torch.inference_mode()
@pytest.mark.parametrize(
    "case",
    _MG_PREFILL_SERVING_CASES,
    ids=lambda case: case.test_id,
)
def test_unified_prefill_mg_specialization_live_graph_oracle(
    case: _MGPrefillServingCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate each typed-SMEM MG group-count/compute arm under live replay.

    Cases cover single-cache and dual-cache DSV4 FP8 and BF16-QK, GLM FP8,
    and GLM NVFP4/BF16 with one and two head groups. DSV4 dual-cache windows
    of 128 and 512 use BF16-QK; widths 1024 and 2048 use FP8-QK.
    """
    device = require_b12x()
    rows = 2

    has_extra = case.family == "dsv4-dual"
    if case.family in ("dsv4", "dsv4-dual"):
        inputs = _make_inputs(
            rows=rows,
            heads=case.heads,
            main_width=case.topk,
            extra_width=64 if has_extra else 0,
            extra_page_size=2 if has_extra else _PAGE_SIZE,
            per_token=True,
            device=device,
        )
        assert inputs.main_lengths is not None
        _poison_inactive_topk_tails(
            inputs.main_index_scenarios,
            inputs.main_length_scenarios,
        )
        if has_extra:
            assert inputs.extra_cache is not None
            assert inputs.extra_indices is not None
            assert inputs.extra_lengths is not None
            assert inputs.extra_index_scenarios is not None
            assert inputs.extra_length_scenarios is not None
            _poison_inactive_topk_tails(
                inputs.extra_index_scenarios,
                inputs.extra_length_scenarios,
            )
        expected = tuple(_reference(inputs, scenario) for scenario in range(2))
        index_scenarios = inputs.main_index_scenarios
        length_scenarios = inputs.main_length_scenarios
        q_scenarios = inputs.q_scenarios
        live_q = inputs.q
        live_indices = inputs.main_indices
        live_lengths = inputs.main_lengths
        kv_cache = inputs.main_cache
        sm_scale = _SM_SCALE
        expected_lse = tuple(result[1] / math.log(2.0) for result in expected)
        scale_format: int | None = None

        def install(scenario: int) -> None:
            _install_scenario(inputs, scenario)

    elif case.family == "glm":
        inputs_glm = _make_glm_inputs(
            rows=rows,
            heads=case.heads,
            width=case.topk,
            per_token=True,
            device=device,
        )
        assert inputs_glm.lengths is not None
        _poison_inactive_topk_tails(
            inputs_glm.index_scenarios,
            inputs_glm.length_scenarios,
        )
        expected = tuple(_glm_reference(inputs_glm, scenario) for scenario in range(2))
        index_scenarios = inputs_glm.index_scenarios
        length_scenarios = inputs_glm.length_scenarios
        q_scenarios = inputs_glm.q_scenarios
        live_q = inputs_glm.q
        live_indices = inputs_glm.indices
        live_lengths = inputs_glm.lengths
        kv_cache = inputs_glm.launch_cache
        sm_scale = _GLM_SM_SCALE
        expected_lse = tuple(result[1] for result in expected)
        scale_format = None

        def install(scenario: int) -> None:
            _install_glm_scenario(inputs_glm, scenario)

    else:
        assert case.family == "glm-nvfp4"
        inputs_nvfp4 = _make_nvfp4_glm_inputs(
            rows=rows,
            heads=case.heads,
            width=case.topk,
            device=device,
        )
        _poison_inactive_topk_tails(
            inputs_nvfp4.index_scenarios,
            inputs_nvfp4.length_scenarios,
        )
        expected = tuple(
            _nvfp4_glm_reference(inputs_nvfp4, scenario) for scenario in range(2)
        )
        index_scenarios = inputs_nvfp4.index_scenarios
        length_scenarios = inputs_nvfp4.length_scenarios
        q_scenarios = inputs_nvfp4.q_scenarios
        live_q = inputs_nvfp4.q
        live_indices = inputs_nvfp4.indices
        live_lengths = inputs_nvfp4.lengths
        kv_cache = inputs_nvfp4.launch_cache
        sm_scale = _GLM_SM_SCALE
        expected_lse = tuple(result[1] for result in expected)
        scale_format = int(ScaleFormat.NVFP4_E4M3)

        def install(scenario: int) -> None:
            _install_nvfp4_glm_scenario(inputs_nvfp4, scenario)

    assert case.heads == case.n_hg * 16
    import b12x.attention._shared.mla.prefill_mg as prefill_mg

    original_run_mg = prefill_mg.run_unified_prefill_mg
    launched_modes = []
    expected_compute = ComputeMode.BF16 if case.compute == "bf16" else ComputeMode.FP8

    def checked_run_mg(**kwargs):
        assert kwargs["compute_mode"] == expected_compute
        assert kwargs["traits_override"].compute_mode == expected_compute
        launched_modes.append(kwargs["compute_mode"])
        return original_run_mg(**kwargs)

    monkeypatch.setattr(prefill_mg, "run_unified_prefill_mg", checked_run_mg)
    assert not torch.equal(q_scenarios[0], q_scenarios[1])
    assert not torch.equal(index_scenarios[0], index_scenarios[1])
    assert not torch.equal(length_scenarios[0], length_scenarios[1])
    assert any(
        int(length.item()) == case.topk
        for lengths in length_scenarios
        for length in lengths
    )
    assert any(
        int(length.item()) % 64 != 0
        for lengths in length_scenarios
        for length in lengths
    )
    assert not torch.allclose(expected[0][0], expected[1][0])

    output = torch.empty(
        (rows, case.heads, _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    lse_base2 = torch.empty(
        (rows, case.heads),
        dtype=torch.float32,
        device=device,
    )
    # MG prefill is single-pass and owns no scratch.  Pass a fixed caller-owned
    # sentinel through the symmetric workspace API and prove replay does not
    # allocate a hidden replacement.
    fixed_workspace = torch.empty((1,), dtype=torch.uint8, device=device)

    stable_tensors = {
        "q": live_q,
        "kv_cache": kv_cache,
        "indices": live_indices,
        "lengths": live_lengths,
        "output": output,
        "lse": lse_base2,
        "workspace": fixed_workspace,
    }
    if has_extra:
        assert case.family == "dsv4-dual"
        assert inputs.extra_cache is not None
        assert inputs.extra_indices is not None
        assert inputs.extra_lengths is not None
        stable_tensors.update(
            {
                "extra_cache": inputs.extra_cache,
                "extra_indices": inputs.extra_indices,
                "extra_lengths": inputs.extra_lengths,
            }
        )
    stable_ptrs = {name: tensor.data_ptr() for name, tensor in stable_tensors.items()}

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = dict(
            q=live_q,
            kv_cache=kv_cache,
            topk_indices=live_indices,
            topk_length=live_lengths,
            sm_scale=sm_scale,
            page_block_size=_PAGE_SIZE,
            output=output,
            lse_out=lse_base2,
            workspace=fixed_workspace,
            scale_format=scale_format,
        )
        if has_extra:
            assert inputs.extra_cache is not None
            assert inputs.extra_indices is not None
            assert inputs.extra_lengths is not None
            kwargs.update(
                extra_kv_cache=inputs.extra_cache,
                extra_indices=inputs.extra_indices,
                extra_topk_length=inputs.extra_lengths,
                extra_page_block_size=inputs.extra_page_size,
            )
        return run_unified_prefill(**kwargs)

    install(0)
    warm_output, warm_lse = launch()
    assert launched_modes == [expected_compute]
    torch.cuda.synchronize(device)
    assert warm_output.data_ptr() == output.data_ptr()
    assert warm_lse.data_ptr() == lse_base2.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize(device)

    first_output: torch.Tensor | None = None
    for scenario in range(2):
        install(scenario)
        assert torch.equal(live_q, q_scenarios[scenario])
        _assert_live_topk_contract(
            live_indices=live_indices,
            live_lengths=live_lengths,
            expected_indices=index_scenarios[scenario],
            expected_lengths=length_scenarios[scenario],
            topk=case.topk,
        )
        if has_extra:
            assert inputs.extra_indices is not None
            assert inputs.extra_lengths is not None
            assert inputs.extra_index_scenarios is not None
            assert inputs.extra_length_scenarios is not None
            _assert_live_topk_contract(
                live_indices=inputs.extra_indices,
                live_lengths=inputs.extra_lengths,
                expected_indices=inputs.extra_index_scenarios[scenario],
                expected_lengths=inputs.extra_length_scenarios[scenario],
                topk=64,
            )
        assert {
            name: tensor.data_ptr() for name, tensor in stable_tensors.items()
        } == stable_ptrs

        output.fill_(float("nan"))
        lse_base2.fill_(float("nan"))
        allocator_before = _allocator_counters(device)
        graph.replay()
        torch.cuda.synchronize(device)
        assert _allocator_counters(device) == allocator_before
        assert {
            name: tensor.data_ptr() for name, tensor in stable_tensors.items()
        } == stable_ptrs

        _assert_output(
            output,
            expected[scenario][0],
            label=f"{case.test_id} replay {scenario}",
        )
        _assert_prefill_boundary_heads(
            output,
            expected[scenario][0],
            n_hg=case.n_hg,
        )
        assert torch.isfinite(lse_base2).all()
        torch.testing.assert_close(
            lse_base2,
            expected_lse[scenario],
            atol=6.0e-2,
            rtol=2.0e-2,
        )
        if first_output is None:
            first_output = output.clone()
        else:
            assert not torch.equal(output, first_output)


@torch.inference_mode()
@pytest.mark.parametrize("entrypoint", ["main", "per-token"])
def test_unified_glm_decode_live_graph_oracle(entrypoint: str) -> None:
    """Compile and validate GLM's distinct 101-KiB typed-SMEM decode body."""
    device = require_b12x()
    rows, heads, width = 2, 8, 128
    per_token = entrypoint == "per-token"
    inputs = _make_glm_inputs(
        rows=rows,
        heads=heads,
        width=width,
        per_token=per_token,
        device=device,
    )
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=width,
        head_dim=_GLM_Q_DIM,
        v_head_dim=_GLM_V_DIM,
        max_chunks_per_row=8,
        page_size=_PAGE_SIZE,
    )
    layout = _compressed_sparse_mla_scratch_layout(caps)
    storage = torch.zeros(layout.nbytes, dtype=torch.uint8, device=device)
    workspace = _materialize_compressed_sparse_mla_scratch(caps, storage, layout)
    output = torch.empty(
        (rows, heads, _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    expected = tuple(_glm_reference(inputs, scenario)[0] for scenario in range(2))
    assert not torch.allclose(expected[0], expected[1])

    def launch() -> None:
        run_unified_decode(
            q_all=inputs.q,
            swa_k_cache=inputs.launch_cache,
            swa_indices=inputs.indices,
            swa_topk_lengths=inputs.lengths,
            workspace=workspace,
            sm_scale=_GLM_SM_SCALE,
            swa_page_size=_PAGE_SIZE,
            forced_num_splits=1,
            out=output,
        )

    _install_glm_scenario(inputs, 0)
    launch()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for scenario in range(2):
        _install_glm_scenario(inputs, scenario)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_output(
            output,
            expected[scenario],
            label=f"GLM decode {entrypoint} replay {scenario}",
        )


@torch.inference_mode()
def test_unified_glm_prefill_live_graph_oracle() -> None:
    """Validate GLM MG prefill with live inputs and fixed caller-owned output."""
    device = require_b12x()
    rows, heads, width = 2, 8, 512
    inputs = _make_glm_inputs(
        rows=rows,
        heads=heads,
        width=width,
        per_token=True,
        device=device,
    )
    output = torch.empty(
        (rows, heads, _GLM_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    lse_base2 = torch.empty((rows, heads), dtype=torch.float32, device=device)
    expected = tuple(_glm_reference(inputs, scenario) for scenario in range(2))
    assert not torch.allclose(expected[0][0], expected[1][0])

    def launch() -> None:
        run_unified_prefill(
            q=inputs.q,
            kv_cache=inputs.launch_cache,
            topk_indices=inputs.indices,
            topk_length=inputs.lengths,
            sm_scale=_GLM_SM_SCALE,
            page_block_size=_PAGE_SIZE,
            output=output,
            lse_out=lse_base2,
        )

    _install_glm_scenario(inputs, 0)
    launch()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for scenario in range(2):
        _install_glm_scenario(inputs, scenario)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_output(
            output,
            expected[scenario][0],
            label=f"GLM prefill replay {scenario}",
        )
        assert torch.isfinite(lse_base2).all()
        torch.testing.assert_close(
            lse_base2,
            expected[scenario][1],
            atol=6.0e-2,
            rtol=2.0e-2,
        )
