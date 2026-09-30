from __future__ import annotations

import os
import socket
from math import gcd

import pytest
import torch
import torch.distributed as dist
from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
from b12x.comm.pcie._dma_kernels import DmaKernels, compile_launchers
from b12x.comm.pcie._dma_preparation import (
    plan as dma_plan,
    prepared_call,
    query_from_runtime,
)
from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce
from b12x.preparation import PreparationSession


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DMA_TEST") != "1",
    reason="set B12X_RUN_PCIE_DMA_TEST=1 to run PCIe ring allreduce GPU tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_input(
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    iteration: int,
) -> torch.Tensor:
    values = torch.arange(rows * hidden, device=device, dtype=torch.float32)
    values = values.reshape(rows, hidden)
    return (torch.sin(values * 0.001 + rank * 0.31 + iteration * 0.17) * 0.5).to(dtype)


def _reference(inp: torch.Tensor) -> torch.Tensor:
    # clone: .float() aliases the input for fp32 and all_reduce is in-place.
    ref = inp.detach().clone().float()
    dist.all_reduce(ref)
    return ref


def _assert_close(actual: torch.Tensor, ref: torch.Tensor, world_size: int) -> None:
    # Stepwise low-precision ring adds; allow world_size half-ulps around the
    # fp32 reference. Compressed wire modes need a wider band because the ring
    # can requantize partial sums at each reduce-scatter hop.
    if os.getenv("B12X_PCIE_DMA_FP8", "0") not in ("", "0"):
        torch.testing.assert_close(
            actual.float(), ref, rtol=1.5e-1, atol=6e-2 * world_size
        )
    elif actual.dtype == torch.float32:
        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-5 * world_size)
    else:
        torch.testing.assert_close(
            actual.float(), ref, rtol=3e-2, atol=3e-2 * world_size
        )



def _prepare_dma(ring: PCIeDmaAllReduce):
    session = PreparationSession(device=ring.device, autotune=False)
    query = query_from_runtime(ring, surface="DmaAllReduce.all_reduce", call={})
    prime_inp = torch.zeros(
        ring.world_size * 128, dtype=torch.bfloat16, device=ring.device
    )
    prime_out = torch.empty_like(prime_inp)
    declaration = dma_plan(query, runtime=ring)
    request = declaration.request(
        name="dma",
        prepare_call=lambda state: prepared_call(state, inp=prime_inp, out=prime_out),
    )
    result = session.prepare((request,))
    return session, result, declaration


def _prepared_mx_kernels() -> DmaKernels:
    kernels = DmaKernels(CudaRTLibrary())
    kernels.install(compile_launchers(world_size=2, wire_mode="mx"))
    return kernels
def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    hidden, max_rows = 6144, 512
    row_multiple = (world_size * 8) // gcd(hidden, world_size * 8)

    def valid_rows(rows: int) -> int:
        return ((rows + row_multiple - 1) // row_multiple) * row_multiple

    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD, device=device, max_bytes=max_rows * hidden * 4
    )
    session, result, plan = _prepare_dma(ring)
    try:
        explicit_inp = _make_input(valid_rows(8), hidden, torch.bfloat16, device, rank, 0)
        explicit_ref = _reference(explicit_inp)
        explicit_out = torch.empty_like(explicit_inp)
        with pytest.raises(ValueError, match="device"):
            ring.all_reduce(
                explicit_inp, plan=plan,
                out=torch.empty_like(explicit_inp, device="cpu"),
            )
        with pytest.raises(ValueError, match="device"):
            ring.all_reduce(
                explicit_inp, plan=plan,
                out=torch.empty_like(explicit_inp, device=(rank + 1) % world_size),
            )
        wrong_device = (rank + 1) % world_size
        torch.cuda.set_device(wrong_device)
        ring.all_reduce(explicit_inp, plan=plan, out=explicit_out)
        assert torch.cuda.current_device() == rank
        torch.cuda.synchronize(device)
        _assert_close(explicit_out, explicit_ref, world_size)

        retained_outputs: list[torch.Tensor] = []
        retained_refs: list[torch.Tensor] = []
        for dtype in (torch.bfloat16,):
            for requested_rows in (8, 64, 256):
                rows = valid_rows(requested_rows)
                inp = _make_input(rows, hidden, dtype, device, rank, 0)
                ref = _reference(inp)
                if requested_rows == 8:
                    out = ring.all_reduce(inp, plan=plan)
                else:
                    out = torch.empty_like(inp)
                    ring.all_reduce(inp, plan=plan, out=out)
                torch.cuda.synchronize(device)
                _assert_close(out, ref, world_size)
                retained_outputs.append(out)
                retained_refs.append(ref)
                for retained, retained_ref in zip(retained_outputs, retained_refs, strict=True):
                    _assert_close(retained, retained_ref, world_size)

        rows = valid_rows(256)
        inp = _make_input(rows, hidden, torch.bfloat16, device, rank, 0)
        out = torch.empty_like(inp)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            ring.all_reduce(inp, plan=plan, out=out)
        for iteration in range(1, 4):
            inp.copy_(_make_input(rows, hidden, torch.bfloat16, device, rank, iteration))
            ref = _reference(inp)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_close(out, ref, world_size)
        dist.barrier()
    finally:
        result.close()
        session.close()
        ring.close()
        dist.destroy_process_group()


def test_pcie_dma_all_reduce_eager_and_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("B12X_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def _fp8_worker(rank: int, world_size: int, port: int, mode: str) -> None:
    os.environ["B12X_PCIE_DMA_FP8"] = mode
    _worker(rank, world_size, port)


@pytest.mark.parametrize(
    "mode",
    [
        "ag",
        "a2a",
        "ring",
        "i8",
        "i8_a2a",
        "i8_ring",
        "mx",
        "mx_a2a",
        "mx_ring",
    ],
)
def test_pcie_dma_all_reduce_compressed_wire(mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    world_size = int(os.getenv("B12X_PCIE_DMA_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _fp8_worker,
        args=(world_size, _free_port(), mode),
        nprocs=world_size,
        join=True,
    )


def test_pcie_dma_mxfp8_codec_matches_cpu_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda:0")
    elems = 4 * 128
    source = (
        torch.sin(torch.arange(elems, device=device, dtype=torch.float32) * 0.07)
        * torch.linspace(0.01, 900.0, elems, device=device)
    ).to(torch.bfloat16)
    source[17] = 1200.0
    source[131] = -0.0002
    storage = torch.empty(elems + elems // 32, dtype=torch.uint8, device=device)
    output = torch.empty_like(source)

    kernels = _prepared_mx_kernels()
    kernels.dma_quant_mx(
        source.data_ptr(), storage.data_ptr(), storage.data_ptr() + elems, elems
    )
    kernels.dma_dequant_store_mx(
        output.data_ptr(), storage.data_ptr(), storage.data_ptr() + elems, elems
    )
    torch.cuda.synchronize(device)

    groups = source.float().cpu().reshape(-1, 32)
    amax = groups.abs().amax(dim=1)
    exponents = torch.where(
        amax > 0,
        torch.ceil(torch.log2(amax / 448.0)),
        torch.zeros_like(amax),
    ).clamp(-127, 127)
    expected_scales = (exponents + 127).to(torch.uint8)
    scale_values = torch.pow(2.0, exponents)
    expected_payload = (groups / scale_values[:, None]).to(torch.float8_e4m3fn)
    expected_output = (expected_payload.float() * scale_values[:, None]).reshape(-1)

    actual_payload = storage[:elems].cpu()
    actual_scales = storage[elems:].cpu()
    assert torch.equal(actual_scales, expected_scales)
    assert torch.equal(actual_payload, expected_payload.reshape(-1).view(torch.uint8))
    assert torch.equal(output.cpu(), expected_output.to(torch.bfloat16))


def test_pcie_dma_mxfp8_quantize_is_exact_for_every_bf16_bit_pattern() -> None:
    """Lock the CUDA E8M0/SATFINITE edge semantics without native code."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda:0")
    bit_patterns = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16)
    source_values = bit_patterns.view(torch.bfloat16)
    source = source_values.to(device).repeat_interleave(32)
    payload = torch.empty(source.numel(), dtype=torch.uint8, device=device)
    scales = torch.empty(1 << 16, dtype=torch.uint8, device=device)

    kernels = _prepared_mx_kernels()
    kernels.dma_quant_mx(
        source.data_ptr(), payload.data_ptr(), scales.data_ptr(), source.numel()
    )
    torch.cuda.synchronize(device)

    # __nv_cvt_float_to_e8m0(amax / 448, SATFINITE, round-positive-inf).
    # For a finite normal BF16 value, division by 448 (= 1.75 * 2**8)
    # increments the resulting exponent exactly when its significand exceeds
    # 1.75. E8M0 reserves byte 0 for 2**-127 and saturates infinity to 254.
    absolute_bits = bit_patterns.to(torch.int32) & 0x7FFF
    exponent = absolute_bits >> 7
    fraction = absolute_bits & 0x7F
    expected_scales = torch.clamp(
        exponent - 8 + (fraction > 96).to(torch.int32), min=0
    )
    expected_scales = torch.where(
        absolute_bits == 0, torch.full_like(expected_scales, 127), expected_scales
    )
    expected_scales = torch.where(
        exponent == 255,
        torch.where(
            fraction == 0,
            torch.full_like(expected_scales, 254),
            torch.full_like(expected_scales, 255),
        ),
        expected_scales,
    ).to(torch.uint8)

    scale_exponents = expected_scales.to(torch.int32) - 127
    scale_values = torch.pow(2.0, scale_exponents.to(torch.float32))
    scale_values = torch.where(
        expected_scales == 0, torch.tensor(2.0**-127), scale_values
    )
    scale_values = torch.where(
        expected_scales == 255, torch.tensor(float("nan")), scale_values
    )
    expected_payload = (
        source_values.float() / scale_values
    ).to(torch.float8_e4m3fn).view(torch.uint8)
    is_inf = (exponent == 255) & (fraction == 0)
    is_nan = (exponent == 255) & (fraction != 0)
    expected_payload = torch.where(
        is_inf,
        torch.where(
            (bit_patterns.to(torch.int32) & 0x8000) != 0,
            torch.full_like(expected_payload, 0xFE),
            torch.full_like(expected_payload, 0x7E),
        ),
        expected_payload,
    )
    expected_payload = torch.where(
        is_nan, torch.full_like(expected_payload, 0x7F), expected_payload
    )

    assert torch.equal(scales.cpu(), expected_scales)
    assert torch.equal(
        payload.cpu(), expected_payload.repeat_interleave(32)
    )


def _preparation_skew_worker(rank: int, world_size: int, port: int) -> None:
    import time
    from datetime import timedelta
    from b12x.comm.pcie._dma_preparation import query_from_metadata

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=world_size, timeout=timedelta(seconds=45),
    )
    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD, device=device, max_bytes=80 << 20,
        fp8="",
    )
    session = PreparationSession(device=device, autotune=False)
    graph = torch.cuda.CUDAGraph()
    try:
        cases = []
        for index, rows in enumerate((4089, 4096)):
            source = torch.full(
                (rows, 5120), rank + 1, device=device, dtype=torch.bfloat16,
            )
            output = torch.empty_like(source)
            declaration = dma_plan(
                query_from_metadata(ring, shape=tuple(source.shape), dtype=source.dtype),
                runtime=ring,
            )

            def call(state):
                if rank == index + 1:
                    time.sleep(0.3)
                return prepared_call(state, inp=source, out=output)

            session.prepare((declaration.request(name=f"dma.m{rows}", prepare_call=call),))
            torch.testing.assert_close(output, torch.full_like(output, 10), rtol=0, atol=0)
            cases.append((declaration, source, output))
            dist.barrier()
        session.freeze()
        for declaration, source, output in cases:
            with session.capture(), torch.cuda.graph(graph):
                ring.all_reduce(source, plan=declaration, out=output)
            address = output.data_ptr()
            for iteration in (1, 2):
                source.fill_(rank + 1 + iteration)
                output.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert output.data_ptr() == address
                assert torch.cuda.memory_allocated(device) == allocated
                torch.testing.assert_close(
                    output, torch.full_like(output, 10 + world_size * iteration),
                    rtol=0, atol=0,
                )
            graph.reset()
        dist.barrier()
    finally:
        graph.reset()
        session.close()
        ring.close()
        dist.destroy_process_group()


def test_pcie_dma_preparation_tolerates_rank_skew_and_replays() -> None:
    import time
    import torch.multiprocessing as mp

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    workers = mp.spawn(
        _preparation_skew_worker, args=(4, _free_port()), nprocs=4, join=False,
    )
    finished = False
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            if workers.join(timeout=1):
                finished = True
                return
        raise TimeoutError("DMA preparation or replay stalled with delayed ranks")
    finally:
        if not finished:
            for process in workers.processes:
                if process.is_alive():
                    process.kill()
            for process in workers.processes:
                process.join(timeout=5)
