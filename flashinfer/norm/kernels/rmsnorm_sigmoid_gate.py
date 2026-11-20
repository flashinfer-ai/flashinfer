# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CuTe DSL fused output RMSNorm and sigmoid gate for per-head output projection.

Token counts are runtime values. Compiled kernels are shared by tensors with
the same head count, token strides, and launch configuration.
"""

import functools

import torch

from ...cute_dsl.availability import is_cute_dsl_arch_supported
from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    HAS_CUTEDSL = True
except ImportError:  # pragma: no cover - exercised without nvidia-cutlass-dsl
    HAS_CUTEDSL = False


HEAD_DIM = 64
_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
LOG2E = 1.4426950408889634
INV_HEAD_DIM = 1.0 / HEAD_DIM

LATENCY_LANES = 32
BANDWIDTH_LANES = 8
LATENCY_WARPS = 32
BANDWIDTH_WARPS = 8
LATENCY_CLUSTER = 16


if HAS_CUTEDSL:

    def _launch_config(rows: int) -> tuple[int, int, int]:
        if rows <= 256:
            rows_per_cta = max(1, rows // LATENCY_CLUSTER)
            warps = max(
                1,
                min(
                    LATENCY_WARPS,
                    rows_per_cta * LATENCY_LANES // 32,
                ),
            )
            return LATENCY_LANES, warps, LATENCY_CLUSTER
        return BANDWIDTH_LANES, BANDWIDTH_WARPS, 1

    def _row_view(
        address,
        row,
        lane_offset,
        token_stride,
        num_heads,
        vector_size,
        vector_bytes,
    ):
        token = row // num_heads
        head = row - token * num_heads
        byte_offset = (token * token_stride + head * HEAD_DIM) * 2 + lane_offset
        return cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16,
                address + byte_offset,
                cutlass.AddressSpace.gmem,
                assumed_align=vector_bytes,
            ),
            cute.make_layout((vector_size,), stride=(1,)),
        )

    @cute.jit
    def _load_rows(
        atom,
        address,
        rows,
        num_rows,
        lane_offset,
        token_stride,
        num_heads,
        vector_size,
        vector_bytes,
        full_tiles: cutlass.Constexpr,
    ):
        fragments = []
        for row in rows:
            fragment = cute.make_rmem_tensor((vector_size,), cutlass.BFloat16)
            if cutlass.const_expr(not full_tiles):
                # Keep every lane in the reduction, without reading padded rows.
                fragment.fill(0)
            if full_tiles or row < num_rows:
                cute.copy(
                    atom,
                    _row_view(
                        address,
                        row,
                        lane_offset,
                        token_stride,
                        num_heads,
                        vector_size,
                        vector_bytes,
                    ),
                    fragment,
                )
            fragments.append(fragment)
        return fragments

    @cute.jit
    def _store_rows(
        atom,
        address,
        rows,
        num_rows,
        lane_offset,
        token_stride,
        num_heads,
        fragments,
        vector_size,
        vector_bytes,
        full_tiles: cutlass.Constexpr,
    ):
        for row, fragment in zip(rows, fragments, strict=True):
            if full_tiles or row < num_rows:
                cute.copy(
                    atom,
                    fragment,
                    _row_view(
                        address,
                        row,
                        lane_offset,
                        token_stride,
                        num_heads,
                        vector_size,
                        vector_bytes,
                    ),
                )

    def _row_sum(fragment, vector_size, log2_lanes):
        values = [fragment[i] for i in range(vector_size)]
        while len(values) > 1:
            values = [
                values[2 * i] + values[2 * i + 1] for i in range(len(values) // 2)
            ]
        value = values[0]
        for index in range(log2_lanes):
            value = value + cute.arch.shuffle_sync_bfly(value, offset=(1 << index))
        return value

    def _epilogue(x_fragments, gate_fragments, eps, vector_size, log2_lanes):
        outputs = []
        for x_fragment, gate_fragment in zip(x_fragments, gate_fragments, strict=True):
            x = x_fragment.load().to(cutlass.Float32)
            gate = gate_fragment.load().to(cutlass.Float32)
            sigmoid = cute.rcp(
                cute.exp2(gate * (-LOG2E), fastmath=True) + 1.0,
                approx=True,
                ftz=True,
            )
            squares = cute.make_rmem_tensor((vector_size,), cutlass.Float32)
            squares.store(x * x)
            inv_rms = cute.rsqrt(
                _row_sum(squares, vector_size, log2_lanes) * INV_HEAD_DIM + eps,
                fastmath=True,
            )
            output = cute.make_rmem_tensor((vector_size,), cutlass.BFloat16)
            output.store((x * inv_rms * sigmoid).to(cutlass.BFloat16))
            outputs.append(output)
        return outputs

    @cute.kernel
    def _rmsnorm_sigmoid_gate_kernel(
        x_address: cutlass.Int64,
        gate_address: cutlass.Int64,
        output_address: cutlass.Int64,
        eps: cutlass.Float32,
        rows: cutlass.Int32,
        num_heads: cutlass.Constexpr,
        x_token_stride: cutlass.Constexpr,
        gate_token_stride: cutlass.Constexpr,
        output_token_stride: cutlass.Constexpr,
        lanes: cutlass.Constexpr,
        threads: cutlass.Constexpr,
        full_tiles: cutlass.Constexpr,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        vector_size = cutlass.const_expr(HEAD_DIM // lanes)
        vector_bytes = cutlass.const_expr(vector_size * 2)
        copy_bits = cutlass.const_expr(vector_size * 16)
        log2_lanes = cutlass.const_expr((lanes - 1).bit_length())
        rows_per_cta = cutlass.const_expr(threads // lanes)

        thread_idx = cute.arch.thread_idx()[0]
        block_idx = cute.arch.block_idx()[0]
        lane_offset = (thread_idx & (lanes - 1)) * vector_bytes
        row = block_idx * rows_per_cta + (thread_idx >> log2_lanes)
        tile_rows = [row]

        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        x_fragments = _load_rows(
            atom,
            x_address,
            tile_rows,
            rows,
            lane_offset,
            x_token_stride,
            num_heads,
            vector_size,
            vector_bytes,
            full_tiles,
        )
        gate_fragments = _load_rows(
            atom,
            gate_address,
            tile_rows,
            rows,
            lane_offset,
            gate_token_stride,
            num_heads,
            vector_size,
            vector_bytes,
            full_tiles,
        )
        outputs = _epilogue(
            x_fragments,
            gate_fragments,
            eps,
            vector_size,
            log2_lanes,
        )
        _store_rows(
            atom,
            output_address,
            tile_rows,
            rows,
            lane_offset,
            output_token_stride,
            num_heads,
            outputs,
            vector_size,
            vector_bytes,
            full_tiles,
        )

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _launch(
        x: cute.Tensor,
        gate: cute.Tensor,
        output: cute.Tensor,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
        num_heads: cutlass.Constexpr,
        x_token_stride: cutlass.Constexpr,
        gate_token_stride: cutlass.Constexpr,
        output_token_stride: cutlass.Constexpr,
        lanes: cutlass.Constexpr,
        warps: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        rows = x.shape[0] * num_heads
        rows_per_cta = cutlass.const_expr(32 * warps // lanes)
        # Whole clusters per token guarantee no padded rows for any runtime T.
        full_tiles = cutlass.const_expr(num_heads % (rows_per_cta * cluster) == 0)
        tiles = cute.ceil_div(rows, rows_per_cta)
        grid = cute.ceil_div(tiles, cluster) * cluster
        _rmsnorm_sigmoid_gate_kernel(
            x.iterator.toint(),
            gate.iterator.toint(),
            output.iterator.toint(),
            eps,
            rows,
            num_heads,
            x_token_stride,
            gate_token_stride,
            output_token_stride,
            lanes,
            32 * warps,
            full_tiles,
            enable_pdl,
        ).launch(
            grid=(grid, 1, 1),
            block=(32 * warps, 1, 1),
            cluster=(cluster, 1, 1),
            stream=stream,
            use_pdl=enable_pdl,
        )


def _as_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.view(tensor.shape[0], 1, tensor.shape[1])
    if tensor.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D tensor, got {tensor.shape}")
    return tensor


@functools.cache
def _get_compiled(
    num_heads: int,
    x_token_stride: int,
    gate_token_stride: int,
    lanes: int,
    warps: int,
    cluster: int,
    enable_pdl: bool = False,
):
    shape = (cute.sym_int(), num_heads, HEAD_DIM)
    x = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        shape,
        stride=(x_token_stride, HEAD_DIM, 1),
        assumed_align=16,
    )
    gate = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        shape,
        stride=(gate_token_stride, HEAD_DIM, 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_name = (
        f"h{num_heads}_xs{x_token_stride}_gs{gate_token_stride}_l{lanes}_"
        f"w{warps}_c{cluster}_pdl{int(enable_pdl)}"
    )
    return build_and_load_cute_dsl_kernel(
        "rmsnorm_sigmoid_gate",
        kernel_name,
        lambda: cute.compile(
            _launch,
            x,
            gate,
            x,
            cutlass.Float32(1e-5),
            stream,
            num_heads,
            x_token_stride,
            gate_token_stride,
            x_token_stride,
            lanes,
            warps,
            cluster,
            enable_pdl,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def rmsnorm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> None:
    if not HAS_CUTEDSL:
        raise RuntimeError("rmsnorm_sigmoid_gate requires nvidia-cutlass-dsl")
    x = _as_3d(x)
    gate = _as_3d(gate)
    if x.shape != gate.shape:
        raise ValueError(f"shape mismatch: x={x.shape}, gate={gate.shape}")
    if x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("x and gate must be bfloat16")
    if not x.is_cuda or not gate.is_cuda:
        raise ValueError("x and gate must be CUDA tensors")
    if x.device != gate.device:
        raise ValueError("x and gate must be on the same device")
    if x.shape[2] != HEAD_DIM:
        raise ValueError(f"expected head size {HEAD_DIM}, got {x.shape[2]}")
    if x.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("head elements must be contiguous")
    if x.stride(0) % 8 or gate.stride(0) % 8:
        raise ValueError("token strides must be multiples of 8 BF16 elements")
    if x.stride(1) != HEAD_DIM or gate.stride(1) != HEAD_DIM:
        raise ValueError("head rows must be contiguous")
    if x.data_ptr() % 16 or gate.data_ptr() % 16:
        raise ValueError("x and gate must be 16-byte aligned")
    if torch.cuda.get_device_capability(x.device) not in _COMPUTE_CAPABILITIES:
        raise ValueError("rmsnorm_sigmoid_gate requires an SM100, SM103, or SM107 GPU")
    if x.numel() == 0:
        return

    rows = x.shape[0] * x.shape[1]
    lanes, warps, cluster = _launch_config(rows)
    rows_per_cta = 32 * warps // lanes
    tiles = (rows + rows_per_cta - 1) // rows_per_cta
    cluster = min(cluster, tiles)

    compiled = _get_compiled(
        x.shape[1],
        x.stride(0),
        gate.stride(0),
        lanes,
        warps,
        cluster,
        enable_pdl,
    )
    compiled(x, gate, x, eps)


def supports_rmsnorm_sigmoid_gate(
    dtype: torch.dtype,
    head_dim: int,
    device: torch.device | str | None = None,
) -> bool:
    """Whether the SM100/SM103/SM107 fused kernel supports this dtype and head size."""
    if not HAS_CUTEDSL or dtype != torch.bfloat16 or head_dim != HEAD_DIM:
        return False
    if device is None:
        if not torch.cuda.is_available():
            return False
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    else:
        resolved_device = torch.device(device)
    return (
        resolved_device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(resolved_device) in _COMPUTE_CAPABILITIES
        and is_cute_dsl_arch_supported(
            *torch.cuda.get_device_capability(resolved_device)
        )
    )
