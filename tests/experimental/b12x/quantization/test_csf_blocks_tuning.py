"""Compact expansion uses the session tuner and retains graph-safe selections."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.csf_blocks_tuning import (
    TUNING,
    CsfExpansionQuery,
    plan_csf_block_expansion,
)
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x
from .test_csf_blocks import planes_and_native


@pytest.mark.parametrize("routes,count", [(5, 14), (65, 28), (129, 14)])
def test_expansion_tuner_enumerates_kernel_parameters(routes, count):
    geometry = (128, 4, 4, 4, 128, 4, 8, 0, 1)
    query = CsfExpansionQuery(geometry, geometry, routes, "int32")
    eligible = TUNING.eligible_plan(query, None)
    configs = [config for _, config in eligible.candidates]
    assert len(configs) == count
    assert {c.groups_per_task for c in configs} == {1, 2, 4, 8, 16, 32, 64}
    assert {c.shared_exceptions for c in configs} == {False, True}
    assert {c.bitmap_routes for c in configs} == (
        {False, True} if count == 28 else {False}
    )
    with pytest.raises(ValueError, match="four-bit"):
        TUNING.configure(
            replace(query, first=(128, 4, 2, 4, 128, 4, 8, 0, 1)), device=None
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_expansion_autotunes_caches_and_replays_without_allocation(tmp_path, dtype):
    device = require_b12x()
    planes, outputs, expected = [], [], []
    for rows, columns in ((320, 160), (2560, 10)):
        source, native = planes_and_native(rows, columns, 8, dense=False)
        planes.append(Nvfp4CompactBlockPlane(upload_csf_block_payload(source, device)))
        outputs.append(
            torch.empty_like(native, device=device).view(torch.float8_e4m3fn)
        )
        expected.append(native.to(device))
    ids = torch.tensor([0, 3, 3, -1, 7] * 13, dtype=dtype, device=device)

    def prepare(state):
        active = torch.empty(8, dtype=torch.int32, device=device)
        binding = state.bind(ids=ids, out13=outputs[0], out2=outputs[1], scratch=active)
        return PreparedCall(
            run=binding.run, produce=lambda: None, owners=(active, binding)
        )

    for source in ("tuned", "cached"):
        plan = plan_csf_block_expansion(*planes, routes=ids.numel(), ids_dtype=dtype)
        with PreparationSession(
            device=device, autotune=True, compile_workers=2, cache_dir=tmp_path
        ) as session:
            session.prepare(
                (
                    plan.request(
                        name="expand", prepare_call=prepare, benchmark_call=prepare
                    ),
                )
            )
            assert plan.selection.source == source
            assert plan.scratch_specs()[0].nbytes == 8 * 4
            active = torch.empty(8, dtype=torch.int32, device=device)
            binding = plan.prepared.state.bind(
                ids=ids, out13=outputs[0], out2=outputs[1], scratch=active
            )
            with pytest.raises(ValueError, match="capacity"):
                plan.prepared.state.bind(
                    ids=ids[:1], out13=outputs[0], out2=outputs[1], scratch=active
                )
            session.freeze()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.run()
            for expert in (0, 6, -1):
                ids.fill_(expert)
                for output in outputs:
                    output.view(torch.uint8).fill_(0xD3)
                active.fill_(42)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                for output, native in zip(outputs, expected, strict=True):
                    reference = torch.full_like(native, 0xD3)
                    if expert >= 0:
                        reference[expert] = native[expert]
                    torch.testing.assert_close(
                        output.view(torch.uint8), reference, rtol=0, atol=0
                    )
            graph.reset()


def test_layers_with_different_exception_payloads_share_tuning(tmp_path):
    from b12x.preparation.session import _coalesce_requests

    device = require_b12x()
    layers = []
    ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    for dense in (False, True):
        source, native = planes_and_native(128, 8, 2, dense=dense)
        plane = Nvfp4CompactBlockPlane(upload_csf_block_payload(source, device))
        outputs = tuple(
            torch.empty_like(native, device=device).view(torch.float8_e4m3fn)
            for _ in range(2)
        )
        plan = plan_csf_block_expansion(plane, plane, routes=ids.numel())
        layers.append((plane, outputs, native.to(device), plan))
    assert layers[0][0].fixed.numel() != layers[1][0].fixed.numel()
    assert layers[0][3].query == layers[1][3].query
    benchmarked = set()

    def prepare(state, outputs):
        active = torch.empty(2, dtype=torch.int32, device=device)
        binding = state.bind(ids=ids, out13=outputs[0], out2=outputs[1], scratch=active)
        return PreparedCall(
            run=binding.run, produce=lambda: None, owners=(active, binding)
        )

    def benchmark(state, index, outputs):
        benchmarked.add(index)
        return prepare(state, outputs)

    from functools import partial

    requests = tuple(
        plan.request(
            name=f"layer{index}/scales",
            prepare_call=partial(prepare, outputs=outputs),
            benchmark_call=partial(benchmark, index=index, outputs=outputs),
        )
        for index, (_, outputs, _, plan) in enumerate(layers)
    )
    assert len(_coalesce_requests(requests)) == 1
    with PreparationSession(
        device=device, autotune=True, compile_workers=2, cache_dir=tmp_path
    ) as session:
        session.prepare(requests)
        assert benchmarked == {0}
        assert layers[0][3].selection == layers[1][3].selection
        assert layers[0][3].prepared.state.first.fixed is layers[0][0].fixed
        assert layers[1][3].prepared.state.first.fixed is layers[1][0].fixed
        bindings = []
        for _, outputs, _, plan in layers:
            active = torch.empty(2, dtype=torch.int32, device=device)
            bindings.append(
                plan.prepared.state.bind(
                    ids=ids, out13=outputs[0], out2=outputs[1], scratch=active
                )
            )
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for binding in bindings:
                binding.run()
        for _, outputs, _, _ in layers:
            for output in outputs:
                output.view(torch.uint8).fill_(0xD3)
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        for _, outputs, expected, _ in layers:
            for output in outputs:
                torch.testing.assert_close(
                    output.view(torch.uint8), expected, rtol=0, atol=0
                )
        graph.reset()
