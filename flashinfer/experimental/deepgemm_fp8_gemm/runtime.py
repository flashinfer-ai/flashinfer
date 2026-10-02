"""Prepared FP8 E4M3 1D1D GEMM launches on the generated SM100a / SM103a programs.

Two routes share one physical schedule (BM128 / BN224 / BK128, six TMA stages,
``cta_group::2``): ``forward`` writes BF16 ``out = A @ B^T``; ``wgrad``
accumulates ``out += A @ B^T`` into FP32 in place.  Problem sizes, tile counts
and the persistent CTA count are launch scalars, so every ``M`` / ``N`` and
every ``K`` multiple of 128 runs on the same compiled program.
"""

from __future__ import annotations

import functools

import torch
import tvm_ffi

from . import cake_jit

BLOCK_M, BLOCK_N, BLOCK_K = 128, 224, 128
SF_WORD_K_BLOCKS = 4  # K128 UE8M0 bytes packed per uint32 scale word
# Grouped-raster M extent folded into the programs.  DeepGEMM's rule
# ``min((8, 16), key=g -> g * BLOCK_M + ceil(num_sms / g) * BLOCK_N)`` selects
# 16 on every part with at least 72 SMs; ``launch_geometry`` refuses others.
RASTER_GROUP_M = 16
_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def device_arch(device) -> str:
    """Generated-program architecture of ``device`` (raises when none is registered)."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("FP8 1D1D GEMM requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _device_facts(index)[0]


@functools.cache
def _device_facts(index: int) -> tuple[str, int]:
    capability = tuple(torch.cuda.get_device_capability(index))
    arch = _ARCHES.get(capability)
    if arch is None or any(
        arch not in record["arches"] for record in cake_jit.MODULES.values()
    ):
        raise RuntimeError(
            f"FP8 1D1D GEMM has no generated programs for compute capability {capability}"
        )
    return arch, int(torch.cuda.get_device_properties(index).multi_processor_count)


def raster_group_m(num_sms: int) -> int:
    """DeepGEMM's grouped-raster M extent for ``num_sms`` persistent CTAs."""
    return min(
        (8, 16),
        key=lambda group: group * BLOCK_M + (num_sms + group - 1) // group * BLOCK_N,
    )


def launch_geometry(M: int, N: int, K: int, num_sms: int) -> dict:
    """Launch scalars for one problem: tile counts, K tiles, grid, packed scale rows.

    ``grid_m`` is rounded up to an even tile count so each cluster pair owns two
    adjacent M tiles (a tile past ``M`` only zero-fills loads and clips stores).
    """
    if M < 1 or N < 1 or K < 1:
        raise ValueError(f"M, N and K must be positive, got M={M} N={N} K={K}")
    if K % BLOCK_K:
        raise ValueError(
            f"K={K} must be a multiple of {BLOCK_K} (one UE8M0 scale per {BLOCK_K} K elements)"
        )
    if raster_group_m(num_sms) != RASTER_GROUP_M:
        raise ValueError(
            f"{num_sms} SMs select a raster group of {raster_group_m(num_sms)} M tiles; "
            f"the generated programs fold {RASTER_GROUP_M}"
        )
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_m += grid_m % 2
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    k_tiles = K // BLOCK_K
    return dict(
        grid_m=grid_m,
        grid_n=grid_n,
        K_tiles=k_tiles,
        grid_x=min(num_sms - num_sms % 2, grid_m * grid_n),
        sf_words=(k_tiles + SF_WORD_K_BLOCKS - 1) // SF_WORD_K_BLOCKS,
    )


def pack_ue8m0_words(scales_u8: torch.Tensor, rows: int) -> torch.Tensor:
    """Pack ``[rows | ceil(rows/128), K/128]`` UE8M0 bytes into MN-major ``[ceil(K/512), rows]`` words.

    Byte 0 of each ``uint32`` is the lowest K block; a trailing partial word is
    zero-padded.  Per-block (128-row) scales are broadcast over their rows.
    This is the packed scale layout the programs read through TMA.
    """
    if scales_u8.dtype != torch.uint8 or scales_u8.dim() != 2:
        raise TypeError("scales must be a 2-D uint8 tensor of UE8M0 exponent bytes")
    if scales_u8.shape[0] != rows:
        if scales_u8.shape[0] != (rows + 127) // 128:
            raise ValueError(
                f"scales have {scales_u8.shape[0]} rows; expected {rows} or {(rows + 127) // 128}"
            )
        scales_u8 = scales_u8.repeat_interleave(128, dim=0)[:rows]
    k_blocks = scales_u8.shape[1]
    padded = (k_blocks + SF_WORD_K_BLOCKS - 1) // SF_WORD_K_BLOCKS * SF_WORD_K_BLOCKS
    if padded != k_blocks:
        scales_u8 = torch.nn.functional.pad(scales_u8, (0, padded - k_blocks))
    words = scales_u8.contiguous().view(torch.uint32)
    # Materialize into a fresh dense [words, rows] buffer: a transposed view of
    # a single-word column reports stride (1, 1), which TMA rejects as a
    # 4-byte row pitch, so ``transpose().contiguous()`` is not enough.
    packed = torch.empty(
        (padded // SF_WORD_K_BLOCKS, rows), dtype=torch.uint32, device=scales_u8.device
    )
    packed.copy_(words.transpose(0, 1))
    return packed


def _check_tma_operand(t: torch.Tensor, name: str) -> None:
    """Reject operands whose row pitch the TMA descriptor cannot encode."""
    pitch = int(t.stride(0)) * t.element_size()
    if (
        t.dim() != 2
        or t.stride(1) != 1
        or int(t.stride(0)) < int(t.shape[1])
        or pitch % 16
        or t.data_ptr() % 16
    ):
        raise ValueError(
            f"{name} must be a 2-D tensor with unit inner stride, a 16-byte aligned "
            f"base and a 16-byte multiple row pitch (got shape {tuple(t.shape)}, "
            f"strides {tuple(t.stride())})"
        )


class Fp8GemmPlan:
    """One prepared launch; ``run()`` submits it on the current PyTorch stream."""

    def __init__(self, entry, args, bindings, geometry):
        self._submission = (entry, args)
        self.bindings = bindings
        self.geometry = geometry

    def run(self):
        entry, args = self._submission
        with tvm_ffi.use_torch_stream():
            entry(*args)


def prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out, *, accumulate=False, cache_dir=None):
    """Prepare ``out = A @ B^T`` (BF16) or ``out += A @ B^T`` (FP32) with packed UE8M0 scale words.

    ``a`` ``[M, K]`` and ``b`` ``[N, K]`` hold FP8 E4M3 bytes (``uint8``).
    ``sfa`` ``[ceil(K/512), M]`` and ``sfb`` ``[ceil(K/512), N]`` are ``uint32``
    words packing four adjacent K128 UE8M0 scale bytes, stored MN-major
    (``pack_ue8m0_words``).  ``K`` is a multiple of 128; ``M`` and ``N`` are
    arbitrary.  Forward writes BF16 ``out`` ``[M, N]``; accumulation reads and
    updates FP32 ``out`` in place, so restore the initializer before each
    independent accumulated evaluation.  ``run()`` allocates nothing and can
    be captured into a CUDA graph.  ``cache_dir`` is accepted for signature
    stability only: programs are built by FlashInfer's JIT.
    """
    del cache_dir
    case = "wgrad" if accumulate else "forward"
    if a.dtype != torch.uint8 or b.dtype != torch.uint8 or a.dim() != 2 or b.dim() != 2:
        raise TypeError("a and b must be 2-D uint8 tensors of FP8 E4M3 bytes")
    M, K = (int(v) for v in a.shape)
    N = int(b.shape[0])
    if int(b.shape[1]) != K:
        raise ValueError(f"b has K={b.shape[1]}, a has K={K}")
    if out.device.type != "cuda" or any(
        t.device != out.device for t in (a, b, sfa, sfb)
    ):
        raise ValueError("All operands must live on the output's CUDA device")
    index = (
        out.device.index
        if out.device.index is not None
        else torch.cuda.current_device()
    )
    arch, num_sms = _device_facts(index)
    geometry = launch_geometry(M, N, K, num_sms)
    words = geometry["sf_words"]
    if (
        sfa.dtype != torch.uint32
        or sfb.dtype != torch.uint32
        or tuple(sfa.shape) != (words, M)
        or tuple(sfb.shape) != (words, N)
    ):
        raise TypeError(
            f"sfa/sfb must be uint32 MN-major packed UE8M0 words of shape [{words}, M] / [{words}, N]"
        )
    if tuple(out.shape) != (M, N):
        raise ValueError(f"out must be [{M}, {N}]")
    if out.dtype != (torch.float32 if accumulate else torch.bfloat16):
        raise TypeError("Accumulator/output dtype does not match the selected route")
    for operand_name, t in (
        ("a", a),
        ("b", b),
        ("sfa", sfa),
        ("sfb", sfb),
        ("out", out),
    ):
        _check_tma_operand(t, operand_name)
    name = cake_jit.KERNELS[case]
    record = cake_jit.MODULES[name]
    module = cake_jit.load_module(name, arch)
    bindings = dict(
        A=a,
        B=b,
        SFA=sfa,
        SFB=sfb,
        C_tma=out,
        M=M,
        N=N,
        K=K,
        grid_m=geometry["grid_m"],
        grid_n=geometry["grid_n"],
        K_tiles=geometry["K_tiles"],
        grid_x=geometry["grid_x"],
        grid_y=1,
        grid_z=1,
    )
    args = tuple(bindings[key] for _, key in record["arg_plan"])
    return Fp8GemmPlan(module[record["ffi_entry"]], args, bindings, geometry)
