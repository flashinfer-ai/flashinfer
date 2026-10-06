"""TMA B-tile swizzle probes for the w4a8 TMA-B staging change.

Pins, bit-tight and before any kernel change, the physical smem byte
placement produced by staging a packed-FP4 B tile through the SAME TMA
machinery the dynamic kernel builds (dense.py `_make_smem_layouts` K-major
atom + `_make_tma_atoms_and_tensors`), and validates the swizzle-corrected
scalar read formula the w4a8 consumer will use.

Primary hypothesis (byte-domain Swizzle<2,4,3>, hardware SWIZZLE_64B over
64-byte rows): for byte offset ``off = row*64 + b`` within one 8KB stage,

    off' = off ^ (((off >> 7) & 3) << 4)

i.e. 16B chunk ``ch`` of row ``row`` lands at chunk ``ch ^ ((row >> 1) & 3)``.
Consumer correction is then purely on the k32-chunk index:

    addr = b_buf + n_in*64 + ((kb ^ ((n_in >> 1) & 3)) << 4) + lane_c*4

Fallback hypothesis (element-domain FP4 swizzle):
    off' = off ^ (((off >> 6) & 3) << 3)
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_ptr

from b12x._lib.intrinsics import (
    e2m1x8_to_e4m3x8,
    ld_shared_u32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
)
from b12x._lib.dense_gemm import DenseGemmKernel
from tests.experimental.b12x.moe.test_w4a8_fragment_probe import _e4m3_bytes, _pack_b

from b12x.testing.reference.helpers import require_b12x

_TILE_N = 128
_TILE_K = 128  # fp4 positions -> 64 bytes per row
_ROW_BYTES = _TILE_K // 2
_STAGE_BYTES = _TILE_N * _ROW_BYTES
_AB_STAGE = 2


def _map_primary(row: int, ch: int) -> int:
    """Confirmed by dump: plain 64B rows, 16B chunk XORed with (row>>1)&3 —
    i.e. byte-domain Swizzle<2,4,3>: off ^= ((off>>7)&3)<<4."""
    return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4)


def _map_fallback(row: int, ch: int) -> int:
    """Element-domain variant (rejected by the dump; kept as the foil)."""
    off = row * 64 + ch * 16
    return off ^ (((off >> 6) & 3) << 3)


class _TmaBLayoutProbe:
    """Stage B tiles via TMA into the kernel's swizzled layout, then either
    dump smem linearly (mode="dump") or read via the candidate corrected
    formula (mode="read")."""

    num_threads = 32

    def __init__(self, *, mode: str, stage_idx: int, k_tile: int,
                 n_total: int, k_total: int):
        assert mode in ("dump", "read", "gemm")
        self.mode = mode
        self.stage_idx = int(stage_idx)
        self.k_tile = int(k_tile)
        self.n_total = int(n_total)
        self.k_total = int(k_total)

    @cute.jit
    def __call__(
        self,
        b_ptr: cute.Pointer,  # fp4 weights, (n_total, K_total, 1) k-major
        mA: cute.Tensor,    # gemm mode: [16, 32] u32 = 16x128 e4m3 bytes
        mOut: cute.Tensor,  # dump: [words] u32 ; read: [128,4,4] u32 ; gemm: [16,8] f32
        stream: cuda.CUstream,
    ):
        mB = cute.make_tensor(
            b_ptr,
            cute.make_ordered_layout(
                (self.n_total, self.k_total, 1), order=(1, 0, 2)
            ),
        )
        # Mirror dense.py `_make_smem_layouts` for the B operand exactly.
        b_layout = utils.LayoutEnum.from_tensor(mB)
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, mB.element_type, _TILE_K),
            mB.element_type,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append((_TILE_N, _TILE_K), _AB_STAGE),
            order=(0, 1, 2),
        )
        tma_b, gB = DenseGemmKernel._make_tma_atoms_and_tensors(
            mB, b_smem_layout_staged, (_TILE_N, _TILE_K), 1
        )
        self.kernel(tma_b, gB, b_smem_layout_staged, mA, mOut).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_b: cute.CopyAtom,
        mB: cute.Tensor,
        b_smem_staged: cute.ComposedLayout,
        mA: cute.Tensor,
        mOut: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            mbar: cute.struct.MemRange[cutlass.Int64, 1]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(b_smem_staged)
                ],
                1024,
            ]

        storage = smem.allocate(Storage)
        sB = storage.sB.get_tensor(
            b_smem_staged.outer, swizzle=b_smem_staged.inner
        )

        mbar_ptr = storage.mbar.data_ptr()
        if Int32(tidx) == Int32(0):
            cute.arch.mbarrier_init(mbar_ptr, Int32(1))
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Tile and index the coordinate tensor EXACTLY like dynamic.py
        # (3-D (n, K, E); slice to (grouped, k_tile) before the copy loop).
        gB_tiled = cute.local_tile(
            mB, (_TILE_N, _TILE_K), (None, None, None)
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_b,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_tiled, 0, 2),
        )
        tBgB_nk = tBgB[(None, 0, None, 0)]
        if Int32(tidx) == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr, Int32(2 * _STAGE_BYTES)
            )
        cute.copy(
            tma_b,
            tBgB_nk[(None, self.k_tile)],
            tBsB[(None, 0)],
            tma_bar_ptr=mbar_ptr,
        )
        cute.copy(
            tma_b,
            tBgB_nk[(None, self.k_tile + 1)],
            tBsB[(None, 1)],
            tma_bar_ptr=mbar_ptr,
        )
        cute.arch.mbarrier_wait(mbar_ptr, phase=0)
        cute.arch.sync_threads()

        sb_base = shared_ptr_to_u32(storage.sB.data_ptr())
        if cutlass.const_expr(self.mode == "dump"):
            # Dump BOTH stages linearly, never through the swizzled view.
            words = 2 * _STAGE_BYTES // 4
            w = Int32(tidx)
            while w < Int32(words):
                mOut[w] = ld_shared_u32(sb_base + (w << Int32(2)))
                w += Int32(self.num_threads)
        elif cutlass.const_expr(self.mode == "read"):
            # Candidate corrected read: for every (n_in, kb, c) fetch the u32
            # the consumer formula would fetch from the requested stage.
            stage_off = Int32(self.stage_idx * _STAGE_BYTES)
            n_in = Int32(tidx)
            while n_in < Int32(_TILE_N):
                for kb in cutlass.range_constexpr(4):
                    for c in cutlass.range_constexpr(4):
                        addr = (
                            sb_base
                            + stage_off
                            + n_in * Int32(_ROW_BYTES)
                            + (
                                (Int32(kb) ^ ((n_in >> Int32(1)) & Int32(3)))
                                << Int32(4)
                            )
                            + Int32(c * 4)
                        )
                        mOut[(n_in, kb, c)] = ld_shared_u32(addr)
                n_in += Int32(self.num_threads)
        if cutlass.const_expr(self.mode == "gemm"):
            # One m16n8 atom over the staged window: lane (c, g) reads B via
            # the corrected formula (n_in = g), A via the probe-pinned plain
            # loads, expands, and accumulates 4 k32 QMMAs. The residual path
            # only changes the expand call (orthogonal to where B bytes come
            # from) and is covered by the original fragment probe.
            stage_off = Int32(self.stage_idx * _STAGE_BYTES)
            lane = cute.arch.lane_idx()
            c = lane % Int32(4)
            g = lane // Int32(4)
            d0 = cutlass.Float32(0.0)
            d1 = cutlass.Float32(0.0)
            d2 = cutlass.Float32(0.0)
            d3 = cutlass.Float32(0.0)
            for kb in cutlass.range_constexpr(4):
                w_pk = ld_shared_u32(
                    sb_base
                    + stage_off
                    + g * Int32(_ROW_BYTES)
                    + ((Int32(kb) ^ ((g >> Int32(1)) & Int32(3))) << Int32(4))
                    + (c << Int32(2))
                )
                b0, b1 = e2m1x8_to_e4m3x8(w_pk)
                a0 = Uint32(mA[g, Int32(8 * kb) + Int32(2) * c])
                a1 = Uint32(mA[g + Int32(8), Int32(8 * kb) + Int32(2) * c])
                a2 = Uint32(mA[g, Int32(8 * kb) + Int32(2) * c + Int32(1)])
                a3 = Uint32(mA[g + Int32(8), Int32(8 * kb) + Int32(2) * c + Int32(1)])
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    d0, d1, d2, d3,
                    a0, a1, a2, a3,
                    b0, b1,
                    Uint32(0x7F7F), Uint32(0x7F7F),
                )
            col = Int32(2) * c
            mOut[g, col] = d0
            mOut[g, col + Int32(1)] = d1
            mOut[g + Int32(8), col] = d2
            mOut[g + Int32(8), col + Int32(1)] = d3


def _make_b_source(device: torch.device, n_tiles: int = 1, k_tiles: int = 2):
    """Global B bytes with a unique u32 per (row, 16B-chunk): word value =
    (global_k_chunk << 16) | row, replicated across the 4 words of a chunk
    with the word-in-chunk index in bits [8,16)."""
    n_total = _TILE_N * n_tiles
    k_chunks_total = k_tiles * 4  # 4 chunks of 16B per 64B tile-row
    words = torch.zeros(n_total, k_chunks_total, 4, dtype=torch.int64)
    rows = torch.arange(n_total, dtype=torch.int64)
    chunks = torch.arange(k_chunks_total, dtype=torch.int64)
    wis = torch.arange(4, dtype=torch.int64)
    words += rows[:, None, None]
    words += wis[None, None, :] << 8
    words += chunks[None, :, None] << 16
    u8 = (
        words.to(torch.int32)
        .view(torch.int32)
        .reshape(n_total, k_chunks_total * 4)
    )
    return u8.contiguous().to(device)


def _u32_tensor(x: torch.Tensor) -> cute.Tensor:
    t = from_dlpack(x, assumed_align=16)
    t.element_type = cutlass.Uint32
    return t


def _fp4_ptr(b_words: torch.Tensor) -> cute.Pointer:
    return make_ptr(
        cutlass.Float4E2M1FN, b_words.data_ptr(), cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _run(mode: str, b_words: torch.Tensor, stage_idx: int = 0, k_tile: int = 0,
         a_bytes: torch.Tensor | None = None):
    device = b_words.device
    if mode == "dump":
        out = torch.zeros(2 * _STAGE_BYTES // 4, dtype=torch.int32, device=device)
    elif mode == "read":
        out = torch.zeros(_TILE_N, 4, 4, dtype=torch.int32, device=device)
    else:
        out = torch.zeros(16, 8, dtype=torch.float32, device=device)
    if a_bytes is None:
        a_bytes = torch.zeros(16, 128, dtype=torch.uint8, device=device)
    n, k_words = b_words.shape
    args = (
        _fp4_ptr(b_words),
        _u32_tensor(a_bytes.view(torch.int32)),
        from_dlpack(out, assumed_align=16),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    probe = _TmaBLayoutProbe(
        mode=mode, stage_idx=stage_idx, k_tile=k_tile,
        n_total=n, k_total=k_words * 8,
    )
    compiled = cute.compile(probe, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return out


def test_tma_b_layout_dump_matches_swizzle_formula():
    """P1: the physical (row, chunk) -> smem offset mapping equals the
    primary byte-domain Swizzle<2,4,3> formula, for both stages, with an
    8192B stage stride."""
    require_b12x()
    device = torch.device("cuda")
    b_words = _make_b_source(device, n_tiles=1, k_tiles=2)
    dump = _run("dump", b_words).cpu()

    src = b_words.cpu().view(_TILE_N, 8, 4)  # (row, global k-chunk, word)
    hits_primary = 0
    hits_fallback = 0
    mismatches = []
    for stage in range(2):
        for row in range(_TILE_N):
            for ch in range(4):
                gch = stage * 4 + ch  # stage s staged k_tile s
                expect = int(src[row, gch, 0])
                off_p = _map_primary(row, ch) + stage * _STAGE_BYTES
                off_f = _map_fallback(row, ch) + stage * _STAGE_BYTES
                got_p = int(dump[off_p // 4])
                got_f = int(dump[off_f // 4])
                if got_p == expect:
                    hits_primary += 1
                if got_f == expect:
                    hits_fallback += 1
                if got_p != expect and len(mismatches) < 8:
                    # Locate where the word actually landed.
                    where = (dump == expect).nonzero()
                    mismatches.append(
                        (stage, row, ch, [int(x) * 4 for x in where[:4]])
                    )
    total = 2 * _TILE_N * 4
    assert hits_primary == total or hits_fallback == total, (
        f"primary {hits_primary}/{total}, fallback {hits_fallback}/{total}; "
        f"first mismatches (stage,row,chunk,found_at_bytes): {mismatches}"
    )
    assert hits_primary == total, (
        f"FALLBACK formula matched, not primary: element-domain swizzle. "
        f"primary {hits_primary}/{total}"
    )


def test_tma_b_corrected_read_formula():
    """P2: the consumer-shaped corrected read returns exactly the source
    word for every (n_in, kb, lane_c), on both stages (8192B apart) and a
    non-zero k_tile (FC2 pair second-tile analogue)."""
    require_b12x()
    device = torch.device("cuda")
    b_words = _make_b_source(device, n_tiles=1, k_tiles=4)
    src = b_words.cpu().view(_TILE_N, 16, 4)

    for stage_idx, k_tile in ((0, 0), (1, 0), (0, 2), (1, 2)):
        got = _run("read", b_words, stage_idx=stage_idx, k_tile=k_tile).cpu()
        gch0 = (k_tile + stage_idx) * 4  # stage s holds k_tile+s
        for row in range(_TILE_N):
            for kb in range(4):
                for c in range(4):
                    expect = int(src[row, gch0 + kb, c])
                    assert int(got[row, kb, c]) == expect, (
                        stage_idx, k_tile, row, kb, c,
                        hex(int(got[row, kb, c])), hex(expect),
                    )


def test_tma_b_gemm_bit_exact():
    """P3: TMA-staged swizzled B + corrected scalar reads + expand + QMMA
    reproduces the torch oracle bit-exactly (dyadic data, unit scales), on
    both stages (the fused gate/up granule halves are exactly stage 0/1 of
    this layout, 8192B apart)."""
    require_b12x()
    device = torch.device("cuda")
    torch.manual_seed(7)
    a_vals = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 4.0, -2.0], device=device)
    a = a_vals[torch.randint(0, 8, (16, 128), device=device)]
    fp4_grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    # Two k-tiles of B (256 fp4 positions per row): stage 0 <- k 0..127,
    # stage 1 <- k 128..255.
    b = fp4_grid[torch.randint(0, 15, (_TILE_N, 256), device=device)]
    b_words = _pack_b(b).view(torch.int32)

    for stage_idx in (0, 1):
        out = _run(
            "gemm", b_words, stage_idx=stage_idx,
            a_bytes=_e4m3_bytes(a).contiguous(),
        )
        k0 = stage_idx * 128
        ref = a @ b[0:8, k0:k0 + 128].T
        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
