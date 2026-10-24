"""Projection rounding, capacity, aliasing, and captured stream-gate execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.gemm import hyperconnection_gate as gate
from b12x.preparation import PreparationSession, PreparedCall
from b12x._lib.runtime_control import kernel_resolution_guard
from ..conftest import require_b12x


def operands(rows, hidden=2560, streams=4, rank=320, capacity=None):
    require_b12x()
    torch.manual_seed(2153)
    options = dict(device="cuda", dtype=torch.bfloat16)
    return (
        torch.randn(rows, rank, **options) * 0.125,
        torch.randn(streams * hidden, rank, **options) * 0.125,
        torch.randn(rows, streams * hidden, **options),
        torch.empty(rows, hidden, **options),
        torch.empty(capacity or rows, streams * hidden, **options),
    )


def prepared(session, tensors, config):
    x, w, v, out, workspace = tensors
    plan = gate.plan(
        gate.Query(
            max_rows=x.shape[0],
            hidden_size=out.shape[1],
            streams=v.shape[1] // out.shape[1],
            rank=x.shape[1],
        ),
        override=config,
    )
    session.prepare(
        (
            plan.request(
                name="gate",
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(x, w, v, out, workspace)
                ),
            ),
        )
    )
    return plan


@pytest.mark.parametrize(
    "shape",
    [
        (1, 17, 1, 32),
        (6, 33, 2, 48),
        (48, 64, 8, 32),
        (129, 2560, 4, 320),
        (4096, 2560, 4, 320),
    ],
)
def test_fused_projection_matches_bf16_rounding_and_stream_tails(shape):
    tensors = operands(*shape)
    x, w, v, out, workspace = tensors
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        plan = prepared(session, tensors, gate.Config())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gate.run(x, w, v, out=out, workspace=workspace, plan=plan)
        for scale in (0.01, 0.125, 1, 4):
            x.normal_(std=scale)
            v.normal_(std=scale)
            v[0].zero_()
            reference = gate.reference(x, w, v, streams=shape[2])
            out.fill_(float("nan"))
            workspace.fill_(float("nan"))
            before = torch.cuda.memory_allocated()
            with kernel_resolution_guard("captured hyperconnection gate"):
                graph.replay()
                torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            torch.testing.assert_close(out, reference, rtol=0.008, atol=0.002)
            assert not out[0].count_nonzero()


@pytest.mark.parametrize("implementation", ["torch", "mma", "simt", "tc"])
def test_existing_projection_paths_reuse_the_same_contract(implementation):
    tensors = operands(6)
    x, w, v, out, workspace = tensors
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        plan = prepared(session, tensors, gate.Config(implementation=implementation))
        torch.testing.assert_close(
            out, gate.reference(x, w, v, streams=4), rtol=0.008, atol=0.002
        )
        assert plan.selection.config.implementation == implementation


def test_live_prefix_reuses_compiled_program_and_preserves_workspace_tail():
    x, w, v, out, workspace = tensors = operands(17, capacity=32)
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        plan = prepared(session, tensors, gate.Config())
        for rows in (1, 6, 0, 17, 3):
            workspace.fill_(19)
            with kernel_resolution_guard("live gate prefix"):
                gate.run(
                    x[:rows],
                    w,
                    v[:rows],
                    out=out[:rows],
                    workspace=workspace,
                    plan=plan,
                )
            torch.testing.assert_close(
                out[:rows],
                gate.reference(x[:rows], w, v[:rows], streams=4),
                rtol=0.008,
                atol=0.002,
            )
            assert torch.all(workspace == 19)


def test_gate_rejects_aliases_and_mismatched_geometry():
    x, w, v, out, workspace = tensors = operands(6)
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        plan = prepared(session, tensors, gate.Config())
        with pytest.raises(ValueError, match="overlap"):
            gate.run(x, w, v, out=out, workspace=v, plan=plan)
        with pytest.raises(ValueError, match="prepared shapes"):
            gate.run(x[:, :16], w, v, out=out, workspace=workspace, plan=plan)
        with pytest.raises(ValueError, match="workspace"):
            gate.run(x, w, v, out=out, workspace=workspace[:3], plan=plan)


def test_complete_gate_tuning_includes_fusion_and_eligible_projection_plans():
    from b12x.gemm.hyperconnection_gate._tuning import TUNING

    query = gate.Query(max_rows=6, hidden_size=2560, streams=4, rank=320)
    choices = tuple(TUNING.parameter_space(query, None).configurations())
    assert {c["implementation"] for c in choices} == {
        "fused",
        "simt",
        "mma",
        "tc",
        "torch",
    }
    for choice in choices:
        TUNING.validate_config(query, gate.Config(**dict(choice)), None)
    with pytest.raises(ValueError, match="streams"):
        TUNING.validate_query(replace(query, streams=3), None)
    with pytest.raises(ValueError, match="multiple of 16"):
        TUNING.validate_query(replace(query, rank=17), None)
    for field, value in (("hidden_size", 65537), ("max_rows", 2**31), ("rank", 2**31)):
        with pytest.raises(ValueError, match="supported indexing"):
            TUNING.validate_query(replace(query, **{field: value}), None)


def test_other_architectures_retain_existing_projection_candidates():
    from b12x.gemm.hyperconnection_gate._tuning import TUNING

    device = SimpleNamespace(compute_capability=(10, 3))
    query = gate.Query(max_rows=6, hidden_size=2560, streams=4, rank=320)
    choices = tuple(TUNING.parameter_space(query, device).configurations())
    assert {c["implementation"] for c in choices} == {"simt", "mma", "tc", "torch"}
    assert TUNING.default_config(query, device).implementation != "fused"
    with pytest.raises(ValueError, match="SM120 or SM121"):
        TUNING.validate_config(query, gate.Config(), device)


def test_custom_op_survives_fullgraph_compilation():
    x, w, v, out, workspace = tensors = operands(6)
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        plan = prepared(session, tensors, gate.Config())

        def run(x, v):
            return gate.run(x, w, v, out=out, workspace=workspace, plan=plan)

        compiled = torch.compile(run, fullgraph=True)
        for _ in range(2):
            x.normal_(std=0.125)
            v.normal_()
            torch.testing.assert_close(
                compiled(x, v),
                gate.reference(x, w, v, streams=4),
                rtol=0.008,
                atol=0.002,
            )


@pytest.mark.parametrize("implementation", ["fused", "simt"])
def test_compilation_does_not_copy_logit_workspace_capacity(implementation):
    tensors = operands(6, capacity=4096)
    x, w, v, out, workspace = tensors
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        plan = prepared(session, tensors, gate.Config(implementation=implementation))

        def run(x, v):
            return gate.run(x, w, v, out=out, workspace=workspace, plan=plan)

        compiled = torch.compile(run, fullgraph=True)
        compiled(x, v)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            compiled(x, v)
            torch.cuda.synchronize()
        kernels = [
            event.name
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(kernels) == (1 if implementation == "fused" else 2), kernels
        if implementation == "fused":
            assert "projection_gate" in kernels[0]
