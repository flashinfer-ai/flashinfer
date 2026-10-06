"""Ordinary dense MXFP8 choices execute through block-scaled linear plans."""
from dataclasses import replace

import pytest
import torch

from b12x.gemm import DenseGemmConfig, block_fp8_linear as bfl
from b12x.gemm._tuning import TUNING as DENSE
from b12x.gemm.block_fp8_linear._tuning import TUNING, BlockFp8LinearQuery, dense_query
from b12x.preparation import DeviceIdentity, PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x
from tests.experimental.b12x.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
    _make_block_fp8_weight,
)

DEVICE = DeviceIdentity(vendor="nvidia", compute_capability=(12, 0), sm_count=188,
                        product_name="RTX PRO 6000 Blackwell Max-Q")
BASE = DenseGemmConfig(backend="cutedsl", tile_m=16, tile_n=64, tile_k=128,
                       load_path="tma", swap_ab=False, split_k_slices=1,
                       large_m_unroll=False, target_occupancy=None)


def query(m=8, block=32):
    return BlockFp8LinearQuery(max_tokens=m, in_features=5120, out_features=1792,
                               source_dtype="bfloat16", output_dtype="bfloat16",
                               output_mode="provided", weight_block_size=block)


@pytest.mark.parametrize("m", (1, 8, 64, 4096))
@pytest.mark.parametrize("block", (32, 128))
@pytest.mark.parametrize("exhaustive", (False, True))
def test_block_scaled_linear_searches_the_complete_dense_domain(m, block, exhaustive):
    declaration = replace(query(m, block), exhaustive=exhaustive)
    wrapper = TUNING.eligible_plan(declaration, DEVICE)
    ordinary = DENSE.eligible_plan(dense_query(declaration), DEVICE)
    assert {config for _, config in wrapper.candidates} == {config for _, config in ordinary.candidates}
    configs = [config for _, config in wrapper.candidates]
    assert {config.tile_k for config in configs} == {64, 128}
    assert {config.swap_ab for config in configs} == {False, True}
    assert {config.large_m_unroll for config in configs} == {False, True}
    if exhaustive:
        assert {config.tile_n for config in configs} == {16, 32, 64, 128}


@pytest.mark.parametrize("n", (6400, 25600))
@pytest.mark.parametrize("m", (1, 6, 8))
def test_fp32_reduction_has_a_legal_default_and_search(monkeypatch, m, n):
    """Disabling atomic BF16 must not leave an unsupported four-slice default."""
    import b12x._lib.dense_gemm as dense

    monkeypatch.setattr(dense, "_B12X_DENSE_SPLITK_TURBO", False)
    declaration = replace(query(m), in_features=6144, out_features=n)
    device = replace(DEVICE, compute_capability=(12, 1), sm_count=48,
                     product_name="NVIDIA GB10")
    configuration = TUNING.configure(declaration, device=device)
    assert configuration.default.split_k_slices in (1, 2)
    candidates = TUNING.eligible_plan(declaration, device).candidates
    assert {config.split_k_slices for _, config in candidates} == {1, 2}


@pytest.mark.parametrize("config", (
    replace(BASE, split_k_slices=2),
    replace(BASE, split_k_slices=4),
    replace(BASE, tile_m=64, tile_n=16, swap_ab=True),
    replace(BASE, tile_m=128, tile_k=64),
    replace(BASE, tile_m=64, tile_n=128, large_m_unroll=True),
))
def test_dense_choices_preserve_quantization_and_allocation_free_replay(config):
    import b12x._lib.dense_gemm as dense

    device = require_b12x()
    capacity, k, n = (64 if config.large_m_unroll else 8), 5120, 1792
    torch.manual_seed(414112)
    source = torch.randn((capacity, k), dtype=torch.bfloat16, device=device)
    weight, scales = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    plan = bfl.plan(bfl.Caps(device=device, max_tokens=capacity, in_features=k,
                            out_features=n, block_size=(32, 32)), override=config)
    output = torch.empty((capacity, n, 1), dtype=source.dtype, device=device)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

    def prepare(state):
        binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=output)
        return PreparedCall(run=lambda: state.run_binding(binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="dense-linear", prepare_call=prepare),))
        state = plan.prepared.state
        assert state.dense.lowering.tile_k == config.tile_k
        assert state.dense.lowering.swap_ab == config.swap_ab
        assert state.dense.lowering.policy.large_m_unroll == config.large_m_unroll
        if config.split_k_slices > 1:
            assert dense._B12X_DENSE_SPLITK_TURBO
            assert state.dense.lowering.policy.split_k_atomic_bf16
            assert state.scratch.workspace_nbytes == 0
        session.freeze()
        pointers = tuple(t.data_ptr() for t in (source, scratch, output, weight, scales))
        for rows in sorted({1, capacity - 1, capacity}):
            binding = bfl.bind(plan, scratch=scratch, source=source[:rows],
                               packed_weight=packed, output=output[:rows])
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    bfl.run(binding=binding)
                for _ in range(2):
                    source.normal_()
                    source[:, :32].mul_(1e-5)
                    scratch.fill_(255)
                    output.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
                    assert pointers == tuple(t.data_ptr() for t in (source, scratch, output, weight, scales))
                    assert torch.isnan(output[rows:]).all()
                    _assert_v41_accumulation_matches_reference(
                        source[:rows], weight, scales, output[:rows, :, 0],
                        atomic_slices=config.split_k_slices,
                    )
            finally:
                graph.reset()
