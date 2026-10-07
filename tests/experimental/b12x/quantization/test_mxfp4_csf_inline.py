"""Inline MXFP4-CSF storage reproduces the native compact W4A8 scale words."""

import os
import subprocess
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import numpy as np
import pytest
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import get_ptr_as_int64, shared_ptr_to_u32
from b12x._lib.quant import mxfp4_csf_inline as inline
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe.fused_moe._impl import _e8m0_scale_to_w4a8_n64_sfb_inplace
from ..conftest import require_b12x


def logical_scales(rows, columns, *, experts=4, seed=0):
    """E8M0 grids with sparse and dense exceptions, clamped and zero rows."""
    rng = np.random.default_rng(seed + rows + columns)
    base = rng.integers(100, 140, (experts, rows, 1))
    grid = base + rng.integers(0, 2, (experts, rows, columns))
    flat = grid.reshape(experts, -1)
    flat[:, ::97] = rng.integers(0, 248, flat[:, ::97].shape)
    for expert in range(experts):
        # Dense exception rows overflow a tile's records: raw (heavy) tiles.
        for row in rng.choice(rows - 4, 3, replace=False):
            grid[expert, row : row + 4] = rng.integers(0, 248, (4, columns))
    grid[:, 0] = 250  # native preparation clamps E8M0 bytes above 247
    grid[:, 1] = 0
    grid[:, 2, ::2] = 247
    return torch.from_numpy(np.minimum(grid, 255).astype(np.uint8))


def native_plane(grid, group_rows, rotation=0):
    device = require_b12x()
    experts, rows, columns = grid.shape
    native = _e8m0_scale_to_w4a8_n64_sfb_inplace(
        grid.clone().to(device),
        weight_E=experts,
        rows=rows,
        k_dim=columns * 32,
        row_rotation=rotation,
        group_rows=group_rows,
    )
    return native.view(torch.uint8).view(experts, rows * columns).contiguous()


def staged_words(native, rows, columns, group_rows):
    """Words of every tile as the native compact stages write them, per row."""
    blocks, k_tiles = inline.inline_geometry(rows, columns, group_rows)
    offsets = torch.from_numpy(inline._slot_offsets(rows, columns, group_rows))
    offsets = offsets.to(native.device).view(blocks * k_tiles, 128, 4)
    values = native[:, offsets.clamp_min(0).view(-1)].view(-1, blocks * k_tiles, 128, 4)
    values = torch.where(offsets >= 0, values, 0)
    slot = torch.arange(128, device=native.device)
    row = (slot >> 5) * 32 + (slot & 3) * 8 + ((slot >> 2) & 7)
    words = torch.zeros_like(values)
    words[:, :, row] = values
    return words.contiguous().view(torch.int32).view(-1)


GEOMETRIES = [
    (1152, 160, 576, 576),  # DS4.1 TP4 W13 (w31), N64 group tails
    (1152, 160, 576, 0),  # DS4.1 TP4 W13 (w13)
    (5120, 18, 5120, 0),  # DS4.1 TP4 W2, K64 tail tiles
    (384, 112, 192, 192),  # Kimi TP16 W13
    (3584, 6, 3584, 0),
]


@pytest.mark.parametrize("rows,columns,group,rotation", GEOMETRIES)
def test_storage_reproduces_native_bytes(rows, columns, group, rotation):
    native = native_plane(logical_scales(rows, columns), group, rotation)
    plane = inline.build_mxfp4_csf_inline(
        native, rows=rows, columns=columns, group_rows=group
    )
    assert plane.heavy_tiles > 0
    tiles = plane.storage[: plane.tiles_bytes].view(-1, inline.TILE_BYTES)
    header = tiles.view(torch.int32)[:, inline.SELECTOR_BYTES // 4]
    assert bool(((header > 0) & (header <= inline.RECORDS)).any())
    assert torch.equal(inline.decode_mxfp4_csf_inline(plane), native)
    assert torch.equal(inline.decode_mxfp4_csf_inline(plane, 1, 3), native[1:3])


@pytest.mark.parametrize("rows,columns,group,rotation", GEOMETRIES)
def test_expansion_reproduces_native_bytes(rows, columns, group, rotation):
    """Calls above the inline limit expand the storage back to the native plane."""
    native = native_plane(logical_scales(rows, columns, seed=1), group, rotation)
    plane = inline.build_mxfp4_csf_inline(
        native, rows=rows, columns=columns, group_rows=group
    )
    assert plane.heavy_tiles > 0
    expanded = torch.full_like(native, 0xFF)
    inline.expand_mxfp4_csf_inline(plane, expanded)
    assert torch.equal(expanded, native)


def test_expansion_launches_the_program_preparation_compiles(tmp_path, monkeypatch):
    """Frozen serving finds the expansion program among fused-MoE preparation's.

    An empty Triton cache and three experts, a specialization no other test
    launches, leave the compile job's program as the only one to reuse.
    """
    from b12x.moe.fused_moe._preparation import compile_inline_scale_expansion
    from b12x.preparation._measurement import no_compilation

    require_b12x()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    rows, columns, group = 5120, 18, 5120
    native = native_plane(logical_scales(rows, columns, experts=3, seed=2), group)
    plane = inline.build_mxfp4_csf_inline(
        native, rows=rows, columns=columns, group_rows=group
    )
    compile_inline_scale_expansion(
        inline.expansion_payload(plane), torch.cuda.current_device()
    )
    expanded = torch.full_like(native, 0xFF)
    with no_compilation():
        inline.expand_mxfp4_csf_inline(plane, expanded)
    torch.cuda.synchronize()
    assert torch.equal(expanded, native)


class _WordProbe:
    """Stage each tile block like a compact kernel and store its rows' words."""

    def __init__(self, experts, blocks, k_tiles, first, tail):
        self.experts, self.blocks, self.k_tiles = experts, blocks, k_tiles
        self.first, self.tail = first, tail

    @cute.jit
    def __call__(self, storage: cute.Pointer, out: cute.Pointer, stream: cuda.CUstream):
        tiles = self.experts * self.blocks * self.k_tiles
        self.kernel(
            cute.make_tensor(storage, cute.make_layout((1,))),
            cute.make_tensor(out, cute.make_layout((tiles * 128,))),
        ).launch(grid=(tiles, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, storage: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        tid = Int32(tid)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Shared:
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, 128], 128]

        shared = shared_ptr_to_u32(smem.allocate(Shared).words.data_ptr())
        base = get_ptr_as_int64(storage, Int32(0))
        inline.stage_inline_tile(base, Int64(tile), shared, tid, self.first)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        warp, quad = tid // Int32(32), (tid & Int32(31)) >> Int32(2)
        slot = warp * Int32(32) + quad * Int32(4)
        tiles_bytes = Int64(
            self.experts * self.blocks * self.k_tiles * inline.TILE_BYTES
        )
        raw = tiles_bytes + Int64(self.experts * self.blocks * inline.BASE_BYTES)
        bases = inline.inline_row_bases(
            base, tiles_bytes, Int64(Int32(tile) // Int32(self.k_tiles)), slot
        )
        words = inline.inline_scale_words(shared, bases, slot, base + raw)
        mask = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.tail):
            mask = Uint32(
                cutlass.select_(
                    Int32(tile) % Int32(self.k_tiles) == Int32(self.k_tiles - 1),
                    Uint32(0xFFFF),
                    mask,
                )
            )
        if (tid & Int32(3)) == Int32(0):
            for nt in cutlass.range_constexpr(4):
                row = warp * Int32(32) + Int32(nt * 8) + quad
                out[Int64(tile) * Int64(128) + Int64(row)] = words[nt] & mask


@pytest.mark.parametrize("first", [0, 32])
@pytest.mark.parametrize("rows,columns,group,rotation", GEOMETRIES)
def test_device_words_match_native_stages(rows, columns, group, rotation, first):
    native = native_plane(logical_scales(rows, columns, seed=7), group, rotation)
    plane = inline.build_mxfp4_csf_inline(
        native, rows=rows, columns=columns, group_rows=group
    )
    blocks, k_tiles = plane.geometry
    probe = _WordProbe(plane.num_experts, blocks, k_tiles, first, bool(columns % 4))
    out = torch.full(
        (plane.num_experts * blocks * k_tiles * 128,),
        -1,
        dtype=torch.int32,
        device=native.device,
    )

    def ptr(tensor, dtype):
        return make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )

    args = (
        ptr(plane.storage, cutlass.Uint8),
        ptr(out, cutlass.Int32),
        current_cuda_stream(),
    )
    cute.compile(probe, *args)(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, staged_words(native, rows, columns, group))


def test_build_rejects_noncompact_geometry():
    native = torch.zeros(1, 256 * 8, dtype=torch.uint8)
    with pytest.raises(ValueError):
        inline.build_mxfp4_csf_inline(native, rows=256, columns=8, group_rows=96)


STRANDING_SCRIPT = """
import gc
import torch
from b12x._lib.quant.mxfp4_csf_inline import build_mxfp4_csf_inline

device = torch.device("cuda")
experts, hidden, inter = 384, 5120, 576
generator = torch.Generator(device=device).manual_seed(0)


def plane(rows, columns):
    base = torch.randint(100, 130, (experts, rows, 1), device=device, generator=generator)
    bump = torch.rand(experts, rows, columns, device=device, generator=generator) < 0.4
    return (base + bump).to(torch.uint8).view(experts, -1).contiguous()


def stranded():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


before = stranded()
w13 = torch.empty(experts, 2 * inter * hidden // 32, dtype=torch.uint8, device=device)
w2 = torch.empty(experts, hidden * inter // 32, dtype=torch.uint8, device=device)
planes = []
for layer in range(4):
    # The native scale preparation that precedes every build.
    staging = (torch.empty_like(w13), torch.empty_like(w2))
    w13.copy_(plane(2 * inter, hidden // 32))
    w2.copy_(plane(hidden, inter // 32))
    del staging
    planes.append(build_mxfp4_csf_inline(w13, rows=2 * inter, columns=hidden // 32, group_rows=inter))
    planes.append(build_mxfp4_csf_inline(w2, rows=hidden, columns=inter // 32, group_rows=hidden))
print(stranded() - before)
"""


def test_build_does_not_strand_allocator_pages():
    """Weight loading must keep the storage clear of freed encoding blocks.

    vLLM loads weights with expandable segments and max_split_size_mb:20. Built
    in the caller's pool, each plane's storage kept pages of its freed
    intermediates mapped: 115 MiB over these eight DeepSeek-V4.1 TP4 planes
    (15 MiB with the private pool), about 1 GiB of KV cache over the model.
    """
    require_b12x()
    environment = dict(
        os.environ,
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:20",
    )
    completed = subprocess.run(
        [sys.executable, "-c", STRANDING_SCRIPT],
        check=True,
        cwd=Path(__file__).resolve().parents[4],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert int(completed.stdout.strip().splitlines()[-1]) < 40 << 20


def test_expansion_writes_experts_past_signed_32bit_byte_offsets():
    device = require_b12x()
    experts, rows, columns = 32769, 128, 512
    blocks, k_tiles = inline.inline_geometry(rows, columns, rows)
    tiles_bytes = experts * blocks * k_tiles * inline.TILE_BYTES
    bases_bytes = experts * blocks * inline.BASE_BYTES
    storage = torch.zeros(tiles_bytes + bases_bytes, dtype=torch.uint8, device=device)
    storage[tiles_bytes:].fill_(127)
    plane = inline.Mxfp4CsfInlinePlane(storage, experts, rows, columns, rows, 0)
    native = torch.zeros(experts, rows * columns, dtype=torch.uint8, device=device)
    assert (experts - 1) * rows * columns >= 2**31
    inline.expand_mxfp4_csf_inline(plane, native)
    torch.cuda.synchronize()
    assert bool((native == 127).all())
