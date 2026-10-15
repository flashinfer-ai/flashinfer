"""Distributed correctness of the NVFP4-transport all-reduce (world size 2 one-shot, world size 4 two-shot) and of
the fused residual + RMSNorm family (world size 2; bfloat16, NVFP4 and MXFP8 outputs)."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import time
from typing import Callable

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm.fp4_allreduce import (
    Fp4AllReduceFusedNormOutput,
    Fp4AllReduceFusedNormResult,
    Fp4AllReduceFusedNormSpec,
    Fp4AllReduceWorkspace,
    fp4_allreduce,
    fp4_allreduce_nvfp4,
    fp4_allreduce_rmsnorm,
    fp4_allreduce_rmsnorm_nvfp4,
    fused_norm_scale_bytes,
    window_plan,
)
from flashinfer.jit.fp4_allreduce import CAPACITY, HIDDEN, RECORD_BYTES, RMSNORM_EPS
from tests.comm import fp4_allreduce_reference as codec

ROWS = {2: (64, 1000, 1001, 4096), 4: (64, 1000, 4096)}
NORM_ROWS = (64, 1000, 1004, 4096)  # the fused-norm family needs a multiple of 4 rows
NORM_PROBE_ROWS = 8192  # the smallest measured tier that runs the probe-all receive-poll program (variant 1)
SUPPORTED_CAPABILITIES = {(10, 0), (10, 3), (10, 7)}
PROCESS_JOIN_TIMEOUT_S = 2400.0  # the first run compiles up to 24 generated programs
NORM_PROCESS_JOIN_TIMEOUT_S = (
    3600.0  # 8 generated programs plus the quantiser modules of the reference
)
PROCESS_CLEANUP_TIMEOUT_S = 30.0
SEED = 1000
NORM_SEED = 2000


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gather(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, tensor.contiguous(), group=group)
    return peers


def _check(out: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if out.shape != expected.shape or out.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: got {tuple(out.shape)} {out.dtype}, expected {tuple(expected.shape)} {expected.dtype}"
        )
    if not torch.equal(out, expected):
        differing = int((out != expected).sum().item())
        raise AssertionError(
            f"{label}: {differing} of {out.numel()} elements differ from the reference"
        )


def _check_identical_across_ranks(
    out: torch.Tensor, group: dist.ProcessGroup, label: str
) -> None:
    for peer_rank, peer in enumerate(_gather(out, group)):
        if not torch.equal(peer, out):
            raise AssertionError(
                f"{label}: rank {dist.get_rank(group)} and rank {peer_rank} disagree"
            )


def _expect(exc_type: type, fn, label: str) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - the test reports the wrong type
        raise AssertionError(
            f"{label}: raised {type(exc).__name__} instead of {exc_type.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: no {exc_type.__name__} was raised")


# ---------------------------------------------------------------------------
# all-reduce family
# ---------------------------------------------------------------------------


def _bf16_cases(workspace, world_size, rank, device, group):
    for launch_with_pdl in (False, True):
        for rows in ROWS[world_size]:
            x = codec.generate_rank_input(rows, rank, SEED + 17 * rows, device)
            expected = codec.reference_allreduce(_gather(x, group))
            out = fp4_allreduce(x, workspace, launch_with_pdl=launch_with_pdl)
            torch.cuda.synchronize(device)
            label = f"bf16 rows={rows} pdl={launch_with_pdl}"
            _check(out, expected, label)
            _check_identical_across_ranks(out, group, label)
            assert workspace.kernel_name("bf16", rows).startswith(
                f"kernel_fp4_allreduce_tp{world_size}_"
            )


def _back_to_back(workspace, world_size, rank, device, group):
    """Four calls without host synchronisation: the three-slot rotation wraps around once."""
    rows = ROWS[world_size][-1]
    inputs = [
        codec.generate_rank_input(rows, rank, SEED + 7777 * call, device)
        for call in range(4)
    ]
    expected = [codec.reference_allreduce(_gather(x, group)) for x in inputs]
    outputs = [
        torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device) for _ in inputs
    ]
    for x, out in zip(inputs, outputs, strict=True):
        fp4_allreduce(x, workspace, out=out)
    torch.cuda.synchronize(device)
    for call, (out, ref) in enumerate(zip(outputs, expected, strict=True)):
        _check(out, ref, f"back-to-back call {call}")


def _alternating_rows(workspace, world_size, rank, device, group):
    """Changing row counts between calls: the in-kernel reset must cover the previous call's ranges."""
    sequence = (
        ROWS[world_size][-1],
        ROWS[world_size][0],
        ROWS[world_size][-1],
        ROWS[world_size][1],
    )
    for call, rows in enumerate(sequence):
        x = codec.generate_rank_input(rows, rank, SEED + 31 * call, device)
        expected = codec.reference_allreduce(_gather(x, group))
        out = fp4_allreduce(x, workspace)
        torch.cuda.synchronize(device)
        _check(out, expected, f"alternating rows call {call} rows={rows}")


def _nvfp4_records(workspace, world_size, rank, device, group):
    for rows in ROWS[world_size][1:]:
        x = codec.generate_rank_input(rows, rank, SEED + 101 * rows, device)
        scale = codec.tensor_global_scale(x)
        records = codec.encode_nvfp4_records(x, scale)
        scales = [
            float(value.item())
            for value in _gather(
                torch.tensor([scale], dtype=torch.float32, device=device), group
            )
        ]
        expected = codec.reference_allreduce_nvfp4(_gather(records, group), scales)
        out = fp4_allreduce_nvfp4(
            records, torch.tensor(scales, dtype=torch.float32, device=device), workspace
        )
        torch.cuda.synchronize(device)
        label = f"nvfp4 rows={rows}"
        _check(out, expected, label)
        _check_identical_across_ranks(out, group, label)


def _graph_replay(workspace, world_size, rank, device, group):
    rows = ROWS[world_size][1]
    x_static = codec.generate_rank_input(rows, rank, SEED + 4242, device)
    out_static = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    expected = codec.reference_allreduce(_gather(x_static, group))
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        fp4_allreduce(x_static, workspace, out=out_static)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(3):
            fp4_allreduce(x_static, workspace, out=out_static)
    for replay in range(3):
        out_static.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        _check(out_static, expected, f"graph replay {replay}")
    # Eager calls after the replays continue the same slot rotation.
    out = fp4_allreduce(x_static, workspace)
    torch.cuda.synchronize(device)
    _check(out, expected, "eager call after graph replays")
    del graph


def _negative_control(workspace, world_size, rank, device, group):
    """A perturbed partial on the last rank must change every rank's result."""
    rows = ROWS[world_size][1]
    x = codec.generate_rank_input(rows, rank, SEED + 99, device)
    expected = codec.reference_allreduce(_gather(x, group))
    perturbed = x.clone()
    if rank == world_size - 1:
        perturbed[rows // 2, HIDDEN // 2] = perturbed[rows // 2, HIDDEN // 2] + 1000.0
    out = fp4_allreduce(perturbed, workspace)
    torch.cuda.synchronize(device)
    if torch.equal(out, expected):
        raise AssertionError(
            "the perturbed partial did not change the all-reduce result"
        )


def _validation(workspace, world_size, device):
    bf16 = dict(dtype=torch.bfloat16, device=device)
    _expect(
        ValueError,
        lambda: fp4_allreduce(
            torch.zeros((64, HIDDEN), dtype=torch.float32, device=device), workspace
        ),
        "float32 input",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((64, HIDDEN - 1), **bf16), workspace),
        "hidden size",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((CAPACITY + 1, HIDDEN), **bf16), workspace),
        "rows above capacity",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((64, 2 * HIDDEN), **bf16)[:, ::2], workspace),
        "non-contiguous input",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((0, HIDDEN), **bf16), workspace),
        "zero rows",
    )
    if world_size == 4:
        _expect(
            ValueError,
            lambda: fp4_allreduce(torch.zeros((6, HIDDEN), **bf16), workspace),
            "rows not a multiple of 4",
        )
    x = torch.zeros((64, HIDDEN), **bf16)
    _expect(
        ValueError,
        lambda: fp4_allreduce(
            x,
            workspace,
            out=torch.zeros((64, HIDDEN), dtype=torch.float32, device=device),
        ),
        "float32 out",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(x, workspace, out=torch.zeros((65, HIDDEN), **bf16)),
        "out rows",
    )
    scales = torch.ones((world_size,), dtype=torch.float32, device=device)
    _expect(
        ValueError,
        lambda: fp4_allreduce_nvfp4(
            torch.zeros((64, RECORD_BYTES - 1), dtype=torch.uint8, device=device),
            scales,
            workspace,
        ),
        "record width",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_nvfp4(
            torch.zeros((64, RECORD_BYTES), dtype=torch.uint8, device=device),
            scales[:-1],
            workspace,
        ),
        "scale count",
    )
    _expect(TypeError, lambda: fp4_allreduce(x, object()), "workspace type")
    spec = Fp4AllReduceFusedNormSpec(weight=torch.ones((HIDDEN,), **bf16))
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(x, x.clone(), spec, workspace),
        "fused-norm call on an all-reduce workspace",
    )


def _check_workspace(workspace, world_size, device):
    plan = window_plan(world_size)
    assert workspace.family == "allreduce"
    assert workspace.world_size == world_size
    assert workspace.capacity == CAPACITY
    assert workspace.table.numel() == plan.table_entries
    assert workspace.table.dtype == torch.int64
    assert workspace.ctrl.numel() == plan.ctrl_words
    assert workspace.ctrl.dtype == torch.int32
    assert (
        workspace.grid == torch.cuda.get_device_properties(device).multi_processor_count
    )
    assert all(int(pointer) != 0 for pointer in workspace.table.tolist())
    for rows in ROWS[world_size]:
        assert 0 <= workspace.variant(rows) < 3


def _worker(world_size: int, rank: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    workspace = Fp4AllReduceWorkspace(group, device=device)
    try:
        _check_workspace(workspace, world_size, device)
        _bf16_cases(workspace, world_size, rank, device, group)
        _back_to_back(workspace, world_size, rank, device, group)
        _alternating_rows(workspace, world_size, rank, device, group)
        _nvfp4_records(workspace, world_size, rank, device, group)
        _graph_replay(workspace, world_size, rank, device, group)
        _negative_control(workspace, world_size, rank, device, group)
        _validation(workspace, world_size, device)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
        workspace.destroy()  # a second destroy is a no-op
    _expect(
        RuntimeError,
        lambda: fp4_allreduce(
            torch.zeros((64, HIDDEN), dtype=torch.bfloat16, device=device), workspace
        ),
        "call after destroy",
    )
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# fused residual + RMSNorm family (world size 2)
# ---------------------------------------------------------------------------


def _norm_inputs(rows: int, rank: int, seed: int, device):
    """Per-rank partial, rank-identical residual and the norm weight of one call."""
    x = codec.generate_rank_input(rows, rank, seed, device)
    residual = codec.generate_residual(rows, seed, device)
    weight = codec.generate_weight(seed, device)
    return x, residual, weight


def _check_norm_result(
    result: Fp4AllReduceFusedNormResult,
    ref: codec.FusedNormReference,
    label: str,
    group: dist.ProcessGroup,
) -> dict:
    """Residual bitwise, y inside atol = rtol = 1e-2 of the float32 reference, both identical on every rank."""
    assert result.output == Fp4AllReduceFusedNormOutput.BF16
    assert result.values is None and result.scales is None
    _check(result.residual_out, ref.residual_out, f"{label}: residual_out")
    report = codec.check_normalised_output(result.out, ref)
    if not report["passed"]:
        raise AssertionError(
            f"{label}: y has {report['violations']} elements outside atol = rtol = 1e-2 and "
            f"{report['non_finite']} non-finite elements (max bfloat16 distance {report['max_ulp']})"
        )
    _check_identical_across_ranks(result.out, group, f"{label}: y")
    _check_identical_across_ranks(result.residual_out, group, f"{label}: residual_out")
    return report


def _rmsnorm_bf16_cases(workspace, rank, device, group):
    for launch_with_pdl in (False, True):
        for rows in NORM_ROWS:
            x, residual, weight = _norm_inputs(
                rows, rank, NORM_SEED + 17 * rows, device
            )
            ref = codec.reference_fused_rmsnorm(_gather(x, group), residual, weight)
            spec = Fp4AllReduceFusedNormSpec(weight=weight)
            result = fp4_allreduce_rmsnorm(
                x, residual, spec, workspace, launch_with_pdl=launch_with_pdl
            )
            torch.cuda.synchronize(device)
            label = f"rmsnorm bf16 rows={rows} pdl={launch_with_pdl}"
            _check_norm_result(result, ref, label, group)
            assert workspace.kernel_name("bf16", rows, "bf16").startswith(
                "kernel_fp4_allreduce_tp2_"
            )


def _rmsnorm_negative_control(workspace, rank, device, group):
    """A reference with a wrong weight bias (1 + 2^-7 instead of 1) must fail the y tolerance."""
    rows = NORM_ROWS[1]
    x, residual, weight = _norm_inputs(rows, rank, NORM_SEED + 99, device)
    biased = codec.reference_fused_rmsnorm(
        _gather(x, group),
        residual,
        weight,
        weight_bias=codec.NEGATIVE_CONTROL_WEIGHT_BIAS,
    )
    result = fp4_allreduce_rmsnorm(
        x, residual, Fp4AllReduceFusedNormSpec(weight=weight), workspace
    )
    torch.cuda.synchronize(device)
    _check(result.residual_out, biased.residual_out, "negative control: residual_out")
    report = codec.check_normalised_output(result.out, biased)
    if report["violations"] == 0:
        raise AssertionError(
            "a reference with weight bias 1 + 2^-7 stayed inside the y tolerance"
        )


def _rmsnorm_records(workspace, rank, device, group):
    for rows in NORM_ROWS[1:]:
        x, residual, weight = _norm_inputs(rows, rank, NORM_SEED + 101 * rows, device)
        scale = codec.tensor_global_scale(x)
        records = codec.encode_nvfp4_records(x, scale)
        scales = [
            float(value.item())
            for value in _gather(
                torch.tensor([scale], dtype=torch.float32, device=device), group
            )
        ]
        ref = codec.reference_fused_rmsnorm(
            residual_bf16=residual,
            weight_bf16=weight,
            records_by_rank=_gather(records, group),
            global_scales=scales,
        )
        result = fp4_allreduce_rmsnorm_nvfp4(
            records,
            torch.tensor(scales, dtype=torch.float32, device=device),
            residual,
            Fp4AllReduceFusedNormSpec(weight=weight),
            workspace,
        )
        torch.cuda.synchronize(device)
        _check_norm_result(result, ref, f"rmsnorm nvfp4 records rows={rows}", group)
        assert workspace.kernel_name("nvfp4", rows, "bf16").startswith(
            "kernel_fp4_allreduce_tp2_"
        )


def _quantiser_environment_is_default() -> None:
    from flashinfer.quantization.fp4_quantization import NVFP4_QUANT_ENV_VARS

    active = [name for name in NVFP4_QUANT_ENV_VARS if os.environ.get(name)]
    if active:
        raise AssertionError(
            "the quantised outputs are compared bitwise against fp4_quantize / mxfp8_quantize in "
            f"their default configuration; unset {active}"
        )


def _rmsnorm_quantised_outputs(workspace, rank, device, group):
    """NVFP4 / MXFP8 outputs equal ``fp4_quantize`` / ``mxfp8_quantize`` of the fused kernel's own bfloat16 ``y``.

    The bfloat16-output program on the same inputs is the probe: both programs
    compute the same ``bf16(y)`` and differ only in how they store it.
    """
    _quantiser_environment_is_default()
    for output in (
        Fp4AllReduceFusedNormOutput.NVFP4,
        Fp4AllReduceFusedNormOutput.MXFP8,
    ):
        for rows in (NORM_ROWS[1], NORM_ROWS[3], NORM_PROBE_ROWS):
            seed = NORM_SEED + 1000 * len(output.value) + rows
            x, residual, weight = _norm_inputs(rows, rank, seed, device)
            ref = codec.reference_fused_rmsnorm(_gather(x, group), residual, weight)
            probe = fp4_allreduce_rmsnorm(
                x, residual, Fp4AllReduceFusedNormSpec(weight=weight), workspace
            )
            torch.cuda.synchronize(device)
            label = f"rmsnorm {output.value} rows={rows}"
            _check_norm_result(probe, ref, f"{label} (probe)", group)
            y = probe.out
            if output == Fp4AllReduceFusedNormOutput.NVFP4:
                global_scale = codec.output_global_scale(y)
                spec = Fp4AllReduceFusedNormSpec(
                    weight=weight, output=output, global_scale=global_scale
                )
                expected_values, expected_scales = codec.nvfp4_output_reference(
                    y, global_scale
                )
                value_dtype, value_bytes = torch.uint8, codec.VALUE_BYTES
            else:
                spec = Fp4AllReduceFusedNormSpec(weight=weight, output=output)
                expected_values, expected_scales = codec.mxfp8_output_reference(y)
                value_dtype, value_bytes = torch.float8_e4m3fn, HIDDEN
            result = fp4_allreduce_rmsnorm(x, residual, spec, workspace)
            torch.cuda.synchronize(device)
            # receive-poll variant 1 (probe-all program, symbol ``_pa_``) above 28 rows per CTA: 1000 and 4096 rows
            # run the grouped program, 8192 rows the probe-all program on 148 / 152 / 212 SMs
            variant = workspace.variant(rows, output.value)
            assert variant == (1 if rows // workspace.grid > 28 else 0)
            kernel = workspace.kernel_name("bf16", rows, output.value)
            assert kernel.startswith("kernel_fp4_allreduce_tp2_lnorm_bf16_")
            assert ("_pa_" in kernel) == (variant == 1)
            assert result.output == output and result.out is None
            assert result.values.dtype == value_dtype
            assert tuple(result.values.shape) == (rows, value_bytes)
            assert result.scales.dtype == torch.uint8
            assert result.scales.numel() == fused_norm_scale_bytes(rows, output)
            _check(result.residual_out, ref.residual_out, f"{label}: residual_out")
            values = result.values.view(torch.uint8)
            _check(values, expected_values, f"{label}: values")
            _check(result.scales, expected_scales, f"{label}: swizzled scales")
            _check_identical_across_ranks(values, group, f"{label}: values")
            _check_identical_across_ranks(result.scales, group, f"{label}: scales")
            # The consumer view of the scales (block order) agrees as well.
            per_row = (
                codec.NVFP4_OUTPUT_SCALES_PER_ROW
                if output == Fp4AllReduceFusedNormOutput.NVFP4
                else codec.MXFP8_OUTPUT_SCALES_PER_ROW
            )
            _check(
                codec.unswizzle_scales_128x4(result.scales, rows, per_row),
                codec.unswizzle_scales_128x4(expected_scales, rows, per_row),
                f"{label}: scales in block order",
            )


def _rmsnorm_back_to_back(workspace, rank, device, group):
    """Four fused calls without host synchronisation into caller-provided buffers (rotation wrap-around)."""
    rows = NORM_ROWS[-1]
    calls = []
    for call in range(4):
        x, residual, weight = _norm_inputs(rows, rank, NORM_SEED + 7777 * call, device)
        ref = codec.reference_fused_rmsnorm(_gather(x, group), residual, weight)
        out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        residual_out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
        calls.append((x, residual, weight, ref, out, residual_out))
    for x, residual, weight, _ref, out, residual_out in calls:
        fp4_allreduce_rmsnorm(
            x,
            residual,
            Fp4AllReduceFusedNormSpec(weight=weight),
            workspace,
            out=out,
            residual_out=residual_out,
        )
    torch.cuda.synchronize(device)
    for call, (_x, _residual, _weight, ref, out, residual_out) in enumerate(calls):
        _check(
            residual_out,
            ref.residual_out,
            f"rmsnorm back-to-back call {call}: residual_out",
        )
        report = codec.check_normalised_output(out, ref)
        assert report["passed"], (call, report)


def _rmsnorm_alternating_rows(workspace, rank, device, group):
    sequence = (NORM_ROWS[-1], NORM_ROWS[0], NORM_ROWS[-1], NORM_ROWS[1])
    for call, rows in enumerate(sequence):
        x, residual, weight = _norm_inputs(rows, rank, NORM_SEED + 31 * call, device)
        ref = codec.reference_fused_rmsnorm(_gather(x, group), residual, weight)
        result = fp4_allreduce_rmsnorm(
            x, residual, Fp4AllReduceFusedNormSpec(weight=weight), workspace
        )
        torch.cuda.synchronize(device)
        _check_norm_result(
            result, ref, f"rmsnorm alternating rows call {call} rows={rows}", group
        )


def _rmsnorm_graph_replay(workspace, rank, device, group):
    rows = NORM_ROWS[1]
    x, residual, weight = _norm_inputs(rows, rank, NORM_SEED + 4242, device)
    ref = codec.reference_fused_rmsnorm(_gather(x, group), residual, weight)
    spec = Fp4AllReduceFusedNormSpec(weight=weight)
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    residual_out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        fp4_allreduce_rmsnorm(
            x, residual, spec, workspace, out=out, residual_out=residual_out
        )
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(3):
            fp4_allreduce_rmsnorm(
                x, residual, spec, workspace, out=out, residual_out=residual_out
            )
    for replay in range(3):
        out.zero_()
        residual_out.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        _check(
            residual_out,
            ref.residual_out,
            f"rmsnorm graph replay {replay}: residual_out",
        )
        _check(out, first, f"rmsnorm graph replay {replay}: y (deterministic)")
    result = fp4_allreduce_rmsnorm(x, residual, spec, workspace)
    torch.cuda.synchronize(device)
    _check(result.out, first, "rmsnorm eager call after graph replays")
    del graph


def _rmsnorm_validation(workspace, device):
    bf16 = dict(dtype=torch.bfloat16, device=device)
    rows = NORM_ROWS[0]
    x = torch.zeros((rows, HIDDEN), **bf16)
    residual = torch.zeros((rows, HIDDEN), **bf16)
    weight = torch.zeros((HIDDEN,), **bf16)
    spec = Fp4AllReduceFusedNormSpec(weight=weight)
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x, residual, Fp4AllReduceFusedNormSpec(weight=weight, eps=1e-5), workspace
        ),
        "eps other than the compiled 1e-6",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x, residual, spec, workspace, residual_out=residual
        ),
        "residual_out aliasing residual",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            torch.zeros((6, HIDDEN), **bf16),
            torch.zeros((6, HIDDEN), **bf16),
            spec,
            workspace,
        ),
        "rows not a multiple of 4",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x, torch.zeros((rows + 4, HIDDEN), **bf16), spec, workspace
        ),
        "residual rows",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x,
            residual,
            Fp4AllReduceFusedNormSpec(
                weight=torch.zeros((HIDDEN,), dtype=torch.float32, device=device)
            ),
            workspace,
        ),
        "float32 weight",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x, residual, Fp4AllReduceFusedNormSpec(weight=weight[:-1]), workspace
        ),
        "weight length",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x,
            residual,
            Fp4AllReduceFusedNormSpec(
                weight=weight, output=Fp4AllReduceFusedNormOutput.NVFP4
            ),
            workspace,
        ),
        "NVFP4 output without a global scale",
    )
    global_scale = torch.ones((1,), dtype=torch.float32, device=device)
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x,
            residual,
            Fp4AllReduceFusedNormSpec(weight=weight, global_scale=global_scale),
            workspace,
        ),
        "global scale with the bfloat16 output",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x,
            residual,
            Fp4AllReduceFusedNormSpec(
                weight=weight,
                output=Fp4AllReduceFusedNormOutput.MXFP8,
            ),
            workspace,
            values=torch.zeros(
                (rows, HIDDEN - 1), dtype=torch.float8_e4m3fn, device=device
            ),
        ),
        "values buffer shape",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x,
            residual,
            Fp4AllReduceFusedNormSpec(
                weight=weight,
                output=Fp4AllReduceFusedNormOutput.MXFP8,
            ),
            workspace,
            scales=torch.zeros((16,), dtype=torch.uint8, device=device),
        ),
        "scales buffer too small",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm(
            x,
            residual,
            spec,
            workspace,
            values=torch.zeros((rows, 3072), dtype=torch.uint8, device=device),
        ),
        "values buffer with the bfloat16 output",
    )
    records = torch.zeros((rows, RECORD_BYTES), dtype=torch.uint8, device=device)
    scales = torch.ones((2,), dtype=torch.float32, device=device)
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm_nvfp4(
            records,
            scales,
            residual,
            Fp4AllReduceFusedNormSpec(
                weight=weight, output=Fp4AllReduceFusedNormOutput.MXFP8
            ),
            workspace,
        ),
        "quantised output for NVFP4 records",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_rmsnorm_nvfp4(
            records, scales[:-1], residual, spec, workspace
        ),
        "record scale count",
    )
    _expect(
        TypeError,
        lambda: fp4_allreduce_rmsnorm(x, residual, object(), workspace),
        "spec type",
    )
    _expect(
        TypeError,
        lambda: fp4_allreduce_rmsnorm(x, residual, spec, object()),
        "workspace type",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(x, workspace),
        "all-reduce call on a fused-norm workspace",
    )


def _check_rmsnorm_workspace(workspace, device):
    plan = window_plan(2, family="rmsnorm")
    assert workspace.family == "rmsnorm"
    assert workspace.world_size == 2 and workspace.capacity == CAPACITY
    assert workspace.table.numel() == plan.table_entries == 2
    assert workspace.ctrl.numel() == plan.ctrl_words
    assert workspace.ctrl.dtype == torch.int32
    assert (
        workspace.grid == torch.cuda.get_device_properties(device).multi_processor_count
    )
    assert all(int(pointer) != 0 for pointer in workspace.table.tolist())
    assert workspace.variant(NORM_ROWS[-1]) == 0
    assert workspace.variant(NORM_ROWS[-1], "nvfp4") == 1
    assert workspace.variant(NORM_ROWS[0], "mxfp8") == 0
    assert RMSNORM_EPS == 1e-6


def _rmsnorm_worker(world_size: int, rank: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    workspace = Fp4AllReduceWorkspace(group, device=device, family="rmsnorm")
    try:
        _check_rmsnorm_workspace(workspace, device)
        _rmsnorm_bf16_cases(workspace, rank, device, group)
        _rmsnorm_negative_control(workspace, rank, device, group)
        _rmsnorm_records(workspace, rank, device, group)
        _rmsnorm_quantised_outputs(workspace, rank, device, group)
        _rmsnorm_back_to_back(workspace, rank, device, group)
        _rmsnorm_alternating_rows(workspace, rank, device, group)
        _rmsnorm_graph_replay(workspace, rank, device, group)
        _rmsnorm_validation(workspace, device)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
        workspace.destroy()
    _expect(
        RuntimeError,
        lambda: fp4_allreduce_rmsnorm(
            torch.zeros((64, HIDDEN), dtype=torch.bfloat16, device=device),
            torch.zeros((64, HIDDEN), dtype=torch.bfloat16, device=device),
            Fp4AllReduceFusedNormSpec(
                weight=torch.zeros((HIDDEN,), dtype=torch.bfloat16, device=device)
            ),
            workspace,
        ),
        "call after destroy",
    )
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# process harness
# ---------------------------------------------------------------------------


def _run_workers(world_size: int, target: Callable, join_timeout_s: float) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    capabilities = {
        torch.cuda.get_device_capability(index) for index in range(world_size)
    }
    if len(capabilities) != 1 or not capabilities.issubset(SUPPORTED_CAPABILITIES):
        pytest.skip("test requires one homogeneous SM100, SM103 or SM107 node")

    port = _open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=target, args=(world_size, rank, port))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + join_timeout_s
    try:
        for rank, process in enumerate(processes):
            process.join(timeout=max(1.0, deadline - time.monotonic()))
            if process.is_alive():
                raise AssertionError(
                    f"rank {rank} did not finish within {join_timeout_s}s"
                )
            assert process.exitcode == 0, f"rank {rank} exited with {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=PROCESS_CLEANUP_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=PROCESS_CLEANUP_TIMEOUT_S)


@pytest.mark.parametrize("world_size", [2, 4])
def test_fp4_allreduce(world_size: int) -> None:
    _run_workers(world_size, _worker, PROCESS_JOIN_TIMEOUT_S)


@pytest.mark.parametrize("world_size", [2])
def test_fp4_allreduce_rmsnorm(world_size: int) -> None:
    _run_workers(world_size, _rmsnorm_worker, NORM_PROCESS_JOIN_TIMEOUT_S)
