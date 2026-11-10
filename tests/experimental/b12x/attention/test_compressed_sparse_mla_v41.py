"""Prepared heterogeneous attention numerical and graph boundaries."""

from dataclasses import replace

import pytest
import torch

from b12x.attention import compressed_sparse_mla
from b12x.attention.compressed_sparse_mla._tuning import TUNING
from b12x.attention._shared.mla.compressed_api import _validate_compressed_cache_layout
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
)
from b12x.preparation import PreparationSession, PreparedCall
from tests.experimental.b12x.conftest import require_b12x as require_sm120

_SM_SCALE = 512**-0.5


@pytest.fixture
def mla_session():
    require_sm120()
    device = torch.device("cuda", torch.cuda.current_device())
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        yield session


def _prepare_binding(session, plan, bind_args, run_args, output):
    def prepare(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=output.device)
        trial_output = torch.empty_like(output)
        binding = state.bind_for_preparation(scratch=scratch, **bind_args)
        return PreparedCall(
            run=lambda: state.run(binding, **run_args, out=trial_output),
            output=trial_output,
            owners=(scratch, binding),
        )

    session.prepare((plan.request(name="compressed-attention", prepare_call=prepare),))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=output.device)
    return compressed_sparse_mla.bind(plan, scratch=scratch, **bind_args), scratch


@pytest.mark.parametrize(
    "heads,mode,large_pool,fp8",
    [
        (heads, mode, False, False)
        for heads in (8, 16, 24, 32, 64)
        for mode in ("decode", "extend")
    ]
    + [(8, mode, True, False) for mode in ("decode", "extend")]
    + [(heads, "decode", False, True) for heads in (1, 2, 4, 8, 12, 16, 20, 24, 32, 64)]
    + [(heads, "extend", False, True) for heads in (8, 16, 24, 32, 64)]
    + [(8, mode, True, True) for mode in ("decode", "extend")]
    + [
        (heads, mode, True, "default")
        for heads in (16, 24)
        for mode in ("decode", "extend")
    ],
)
@torch.inference_mode()
def test_v41_heterogeneous_attention_replay_and_live_rows(
    heads: int,
    mode: str,
    large_pool: bool,
    fp8: bool | str,
    monkeypatch,
    mla_session,
) -> None:
    from b12x.attention._shared.mla.compressed_reference import (
        pack_deepseek_v41_cache_reference,
    )

    device = require_sm120()
    rows, width, page_size = 19, 128, 64
    gen = torch.Generator(device=device).manual_seed(4100 + heads)
    group_scales = torch.linspace(0.03, 0.7, 32, device=device).repeat_interleave(16)
    swa_kv = (
        torch.randn((width, 512), device=device, generator=gen) * group_scales
    ).bfloat16()
    indexed_kv = (
        torch.randn((width, 512), device=device, generator=gen)
        * group_scales.flip(0)
        * 3
    ).bfloat16()
    swa_packed = pack_deepseek_v41_cache_reference(
        swa_kv, page_size=page_size, cache_kind="swa"
    )
    indexed_packed = pack_deepseek_v41_cache_reference(
        indexed_kv, page_size=page_size, cache_kind="indexed"
    )
    swa_cache = torch.empty((3, swa_packed.shape[1]), device=device, dtype=torch.uint8)
    swa_cache[0].fill_(127)  # FP8 NaNs in the inactive fallback page.
    swa_cache[1:].copy_(swa_packed)
    indexed_pid = (2**31 // int(indexed_packed.stride(0)) + 1) if large_pool else 1
    required_bytes = (indexed_pid + 2) * indexed_packed.shape[1]
    if (
        large_pool
        and torch.cuda.mem_get_info(device)[0] < required_bytes + 512 * 1024**2
    ):
        pytest.skip("insufficient free memory for mapped physical offsets beyond 2 GiB")
    indexed_cache = torch.empty(
        (indexed_pid + 2, indexed_packed.shape[1]), device=device, dtype=torch.uint8
    )
    indexed_cache[0].fill_(127)
    indexed_cache[indexed_pid:].copy_(indexed_packed)
    q = (torch.randn((rows, heads, 512), device=device, generator=gen) * 0.2).bfloat16()
    # Tail-only heads make an accidental V4 448+64 layout visibly wrong.
    q[:, 0, :448].zero_()
    q[:, 0, 448:].mul_(8)
    logical = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
    swa_indices = logical + page_size
    logical[:, 100:].fill_(-1)
    lengths = torch.full((rows,), width, device=device, dtype=torch.int32)
    lengths[0] = 0
    lengths[1] = 65
    indexed_lengths = lengths.clone()
    table = torch.tensor(
        [indexed_pid, indexed_pid + 1], device=device, dtype=torch.int32
    ).expand(rows, -1)
    caps = compressed_sparse_mla.Caps(
        device=q.device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=2 * width,
        swa_width=width,
        indexed_width=width,
        max_page_table_width=2,
        swa_page_size=page_size,
        indexed_page_size=page_size,
        cache_format="deepseek_v41",
        mode=mode,
        max_chunks_per_row=4,
        use_cuda_graph=True,
    )
    output = torch.empty_like(q)
    invocation = compressed_sparse_mla.invocation_from_tensors(
        q=q,
        swa_k_cache=swa_cache,
        indexed_k_cache=indexed_cache,
        out=output,
        return_lse=True,
        lse_scale="natural",
    )
    plan = compressed_sparse_mla.plan(caps, invocation=invocation)
    if fp8 != "default":
        execution = replace(
            TUNING.configure(plan.query, device=mla_session.device.identity).default,
            v41_compute_mode="fp8" if fp8 else "bf16",
            v41_heads_per_block=8 if fp8 and mode == "decode" and heads % 16 else 16,
        )
        plan = compressed_sparse_mla.plan(
            caps, invocation=invocation, override=execution
        )
    bind_args = dict(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=lengths,
        indexed_indices=logical,
        indexed_lengths=indexed_lengths,
        indexed_page_table=table,
    )
    run_args = dict(
        swa_k_cache=swa_cache,
        indexed_k_cache=indexed_cache,
        sm_scale=_SM_SCALE,
        return_lse=True,
        lse_scale="natural",
    )
    binding, storage = _prepare_binding(mla_session, plan, bind_args, run_args, output)
    if heads == 16 and mode == "decode" and not large_pool and fp8 is True:
        offset_storage = torch.empty(q.numel() + 1, dtype=q.dtype, device=device)
        offset_q = offset_storage[1:].view_as(q)
        with pytest.raises(ValueError, match="Q alignment"):
            compressed_sparse_mla.bind(
                plan, scratch=storage, **dict(bind_args, q=offset_q)
            )

    def run(active_binding, active_output):
        return compressed_sparse_mla.run(
            binding=active_binding,
            swa_k_cache=swa_cache,
            swa_page_size=page_size,
            indexed_k_cache=indexed_cache,
            indexed_page_size=page_size,
            sm_scale=_SM_SCALE,
            return_lse=True,
            lse_scale="natural",
            out=active_output,
        )

    def check(actual, lse, live):
        physical = (
            table[:live]
            .long()
            .gather(1, logical[:live].long().clamp_min(0) // page_size)
        )
        physical = (physical * page_size + logical[:live].long() % page_size).int()
        physical.masked_fill_(logical[:live] < 0, -1)
        expected, expected_lse = compressed_sparse_mla_reference(
            q[:live],
            swa_cache,
            swa_indices[:live],
            lengths[:live],
            extra_k_cache=indexed_cache,
            extra_indices=physical,
            extra_topk_lengths=indexed_lengths[:live],
            swa_page_size=page_size,
            extra_page_size=page_size,
            sm_scale=_SM_SCALE,
            return_lse=True,
            cache_format="deepseek_v41",
        )
        if fp8 and live:
            from b12x.testing.reference.v41_fp8 import (
                canonical_fp8_rows,
                split64_fp8_attention,
            )

            swa_data, swa_scales = canonical_fp8_rows(swa_packed, "swa")
            main_data, main_scales = canonical_fp8_rows(indexed_packed, "indexed")
            swa_local = swa_indices[:live].long() - page_size
            main_local = physical.long() - indexed_pid * page_size
            columns = torch.arange(width, device=device)[None]
            swa_valid = (columns < lengths[:live, None]) & (swa_local >= 0)
            main_valid = (columns < indexed_lengths[:live, None]) & (physical >= 0)
            key_data = torch.cat(
                (swa_data[swa_local.clamp_min(0)], main_data[main_local.clamp_min(0)]),
                dim=1,
            )
            key_scales = torch.cat(
                (
                    swa_scales[swa_local.clamp_min(0)],
                    main_scales[main_local.clamp_min(0)],
                ),
                dim=1,
            )
            expected, expected_lse = split64_fp8_attention(
                q[:live],
                key_data,
                key_scales,
                torch.cat((swa_valid, main_valid), dim=1),
                _SM_SCALE,
                qk_fp8=mode == "decode",
                round_split_outputs=mode == "decode",
            )
        tolerance = 0.008 if fp8 else 0.035
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            lse, expected_lse, atol=0.005 if fp8 else 0.025, rtol=0.005 if fp8 else 0.01
        )
        if live:
            assert torch.equal(actual[0], torch.zeros_like(actual[0]))
            assert torch.isneginf(lse[0]).all()

    warm_output, warm_lse = run(binding, output)
    check(warm_output, warm_lse, rows)
    # Runtime environment mutations must not change an already-selected plan.
    monkeypatch.setenv("B12X_MLA_SM120_DSV41_NATIVE", "0" if fp8 else "h8")

    mla_session.freeze()
    graph = torch.cuda.CUDAGraph()
    try:
        for live in (3, 1, 0):
            live_binding = compressed_sparse_mla.bind(
                plan,
                scratch=storage,
                q=q[:live],
                swa_indices=swa_indices[:live],
                swa_lengths=lengths[:live],
                indexed_indices=logical[:live],
                indexed_lengths=indexed_lengths[:live],
                indexed_page_table=table[:live],
            )
            actual, lse = run(live_binding, output[:live])
            check(actual, lse, live)
        with torch.cuda.graph(graph):
            actual, lse = run(binding, output)
        pointers = (
            output.data_ptr(),
            binding.scratch.final_lse.data_ptr(),
            binding.scratch.mapped_indices.data_ptr(),
        )
        # Mutate both page mapping and live source lengths after capture.
        table[0].copy_(table[0].flip(0))
        lengths[1:].fill_(3)
        indexed_lengths[1:].fill_(7)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        check(actual, lse, rows)
        assert pointers == (
            output.data_ptr(),
            binding.scratch.final_lse.data_ptr(),
            binding.scratch.mapped_indices.data_ptr(),
        )
    finally:
        graph.reset()


@pytest.mark.parametrize("cache_format", ["deepseek_v4", "deepseek_v41"])
@torch.inference_mode()
def test_prepared_compressed_swa_only_matches_bound_recipe(
    cache_format: str, mla_session
) -> None:
    from b12x.attention._shared.mla.compressed_reference import (
        pack_deepseek_v41_cache_reference,
    )

    device = require_sm120()
    page_size, heads = 64, 8
    q = torch.full((2, heads, 512), 0.125, device=device, dtype=torch.bfloat16)
    kv = torch.linspace(-0.5, 0.75, 64 * 512, device=device).reshape(64, 512).bfloat16()
    cache = (
        pack_deepseek_v41_cache_reference(kv, page_size=page_size, cache_kind="swa")
        if cache_format == "deepseek_v41"
        else pack_compressed_sparse_mla_kv_cache_reference(
            kv[:, :448], kv[:, 448:], page_size=page_size
        )
    )
    indices = torch.arange(64, device=device, dtype=torch.int32).repeat(2, 1)
    lengths = torch.tensor([64, 0], device=device, dtype=torch.int32)
    expected = compressed_sparse_mla_reference(
        q,
        cache,
        indices,
        lengths,
        swa_page_size=page_size,
        cache_format=cache_format,
        sm_scale=_SM_SCALE,
    )
    output = torch.empty_like(q)
    plan = compressed_sparse_mla.plan(
        compressed_sparse_mla.Caps(
            device=q.device,
            num_q_heads=heads,
            max_q_rows=2,
            max_width=64,
            swa_width=64,
            indexed_width=0,
            cache_format=cache_format,
        ),
        invocation=compressed_sparse_mla.invocation_from_tensors(
            q=q, swa_k_cache=cache, out=output
        ),
    )
    binding, storage = _prepare_binding(
        mla_session,
        plan,
        dict(q=q, swa_indices=indices, swa_lengths=lengths),
        dict(swa_k_cache=cache, sm_scale=_SM_SCALE),
        output,
    )
    mla_session.freeze()
    bound_result = compressed_sparse_mla.run(
        binding=binding,
        swa_k_cache=cache,
        swa_page_size=page_size,
        sm_scale=_SM_SCALE,
        out=output,
    )
    torch.testing.assert_close(bound_result, expected, atol=0.035, rtol=0.035)
    with pytest.raises(ValueError, match="cache_format differs"):
        compressed_sparse_mla.run(
            binding=binding,
            swa_k_cache=cache,
            swa_page_size=page_size,
            cache_format="deepseek_v4"
            if cache_format == "deepseek_v41"
            else "deepseek_v41",
            sm_scale=_SM_SCALE,
        )


@pytest.mark.parametrize("cache_kind,record_bytes", [("swa", 528), ("indexed", 288)])
def test_v41_cache_width_is_source_specific(cache_kind, record_bytes) -> None:
    page_size = 3
    cache = torch.empty((2, page_size * record_bytes), dtype=torch.uint8)
    _validate_compressed_cache_layout(
        cache,
        page_size=page_size,
        name="cache",
        cache_format="deepseek_v41",
        cache_kind=cache_kind,
    )
    with pytest.raises(ValueError, match="page byte width"):
        _validate_compressed_cache_layout(
            cache,
            page_size=page_size,
            name="cache",
            cache_format="deepseek_v41",
            cache_kind="indexed" if cache_kind == "swa" else "swa",
        )
