"""Prepared MXFP4 DSA numerical contracts."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from b12x.attention import dsa_indexer
from b12x.preparation import PreparationSession, PreparedCall


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="native prepared MXFP4 DSA requires SM12x",
)


def _oracle_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (amax / 6).contiguous().view(torch.int32)
    scales = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    scale = (scales.int() << 23).view(torch.float32)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    pick = (groups.div(scale[..., None]).abs()[..., None] - lut[order]).abs().argmin(-1)
    code = order[pick].to(torch.uint8) | (torch.signbit(groups).to(torch.uint8) << 3)
    return code.reshape(*x.shape[:-1], 128)[..., ::2] | (code.reshape(*x.shape[:-1], 128)[..., 1::2] << 4), scales, (lut[(code & 7).long()] * torch.where((code & 8) != 0, -1, 1) * scale[..., None]).bfloat16().reshape_as(x)


@contextmanager
def _prepared(caps: dsa_indexer.Caps, *, q, keys, slots, lengths, weights):
    page_bytes = dsa_indexer.index_mxfp4_page_bytes(caps.page_size)
    pool = torch.empty((max(1, int(slots.max().item()) // caps.page_size + 1), page_bytes), dtype=torch.uint8, device=q.device)
    packed = torch.empty((*q.shape[:-1], 64), dtype=torch.uint8, device=q.device)
    scales = torch.empty((*q.shape[:-1], 4), dtype=torch.uint8, device=q.device)
    output = torch.empty((q.shape[0], caps.topk), dtype=torch.int32, device=q.device)
    values = torch.empty((q.shape[0], caps.topk), dtype=torch.float32, device=q.device)
    page_table = (slots[::caps.page_size] // caps.page_size).int()[None]
    declaration = dsa_indexer.plan(caps)

    def prime(state):
        state.write_index_keys(keys, index_k_cache=pool, slot_mapping=slots)
        state.quantize_query(q, q_mxfp4=packed, q_scales=scales)
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
        binding = state.bind(scratch=scratch, q_mxfp4=packed, q_scales=scales,
                             query_weights=weights, index_k_cache=pool,
                             page_table=page_table, cache_lengths=lengths,
                             active_width=torch.tensor([keys.shape[0]], dtype=torch.int32, device=q.device),
                             output_indices=output, output_scores=values)
        return PreparedCall(run=lambda: state.run(binding), owners=(pool, packed, scales, scratch, binding))

    request = declaration.request(name="mxfp4", prepare_call=prime)
    with PreparationSession(device=q.device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        yield declaration, pool, packed, scales, page_table, output, values


def test_prepared_quantizers_preserve_adversarial_e2m1_and_writer_layout() -> None:
    device = torch.device("cuda")
    q = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device=device)
    q[0, 0, :8] = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6], device=device)
    keys = torch.randn((128, 128), dtype=torch.bfloat16, device=device)
    slots = torch.arange(128, dtype=torch.int64, device=device) + 64
    lengths = torch.tensor([128], dtype=torch.int32, device=device)
    weights = torch.ones((1, 4), dtype=torch.bfloat16, device=device) / 64
    caps = dsa_indexer.Caps(device=device, num_q_heads=4, max_q_rows=1,
                            max_page_table_width=2, topk=512, cache_format="mxfp4")
    with _prepared(caps, q=q, keys=keys, slots=slots, lengths=lengths, weights=weights) as (plan, pool, packed, scales, _, _, _):
        expected_packed, expected_scales, _ = _oracle_quant(q)
        torch.testing.assert_close(packed, expected_packed)
        torch.testing.assert_close(scales, expected_scales)
        expected_k, expected_ks, _ = _oracle_quant(keys)
        page = slots // 64
        token = slots % 64
        torch.testing.assert_close(pool[page[:, None], token[:, None] * 64 + torch.arange(64, device=device)], expected_k)
        torch.testing.assert_close(pool[page[:, None], 64 * 64 + token[:, None] * 4 + torch.arange(4, device=device)], expected_ks)
        with pytest.raises(TypeError, match="session-prepared"):
            dsa_indexer.quantize_q_mxfp4(caps, q, q_mxfp4=packed, q_scales=scales)
        assert plan.component_id == "attention.dsa_indexer"


def test_prepared_score_reduces_bf16_before_selecting_logical_indices() -> None:
    torch.manual_seed(555)
    device = torch.device("cuda")
    q = torch.randn((2, 4, 128), dtype=torch.bfloat16, device=device)
    keys = torch.randn((128, 128), dtype=torch.bfloat16, device=device)
    slots = torch.arange(128, dtype=torch.int64, device=device) + 64
    lengths = torch.tensor([128, 91], dtype=torch.int32, device=device)
    weights = torch.randn((2, 4), dtype=torch.bfloat16, device=device) / 64
    caps = dsa_indexer.Caps(device=device, num_q_heads=4, max_q_rows=2,
                            max_page_table_width=2, topk=512, mode="prefill", cache_format="mxfp4")
    with _prepared(caps, q=q, keys=keys, slots=slots, lengths=lengths, weights=weights) as (plan, pool, packed, scales, pages, output, values):
        (spec,) = dsa_indexer.scratch_specs(plan, device=q.device)
        binding = dsa_indexer.bind(plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), q_mxfp4=packed, q_scales=scales, query_weights=weights, index_k_cache=pool, page_table=pages, cache_lengths=lengths, active_width=torch.tensor([128], dtype=torch.int32, device=device), output_indices=output, output_scores=values)
        scores = dsa_indexer.score(binding)
        peer = torch.randn_like(scores).bfloat16()
        scores.add_(peer)
        dsa_indexer.select(binding)
        for row, length in enumerate(lengths.tolist()):
            expected = (scores[row, :length].float().topk(min(512, length)).indices.sort().values).int()
            torch.testing.assert_close(output[row, :expected.numel()], expected)
            assert output[row, expected.numel():].eq(-1).all()


def test_prepared_writer_requires_int64_physical_slot_mapping() -> None:
    device = torch.device("cuda")
    q = torch.ones((1, 1, 128), dtype=torch.bfloat16, device=device)
    keys = torch.ones((64, 128), dtype=torch.bfloat16, device=device)
    slots = torch.arange(64, dtype=torch.int64, device=device) + 64
    caps = dsa_indexer.Caps(device=device, num_q_heads=1, max_q_rows=1,
                            max_page_table_width=1, topk=512, cache_format="mxfp4")
    with _prepared(caps, q=q, keys=keys, slots=slots, lengths=torch.tensor([64], dtype=torch.int32, device=device), weights=torch.ones((1, 1), dtype=torch.bfloat16, device=device)) as (plan, pool, _, _, _, _, _):
        with pytest.raises(TypeError, match="int64"):
            dsa_indexer.quantize_write_index_k_mxfp4(plan, keys, index_k_cache=pool, slot_mapping=slots.int())
