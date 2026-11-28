"""Synchronization of the warp split-K MXFP8 GEMM
(flashinfer/gemm/kernels/dense_mxfp8_gemm_warp_splitk.py, a runner of
``mm_mxfp8(backend="cute-dsl")``), checked in its generated PTX.

These are memory-model orderings: the kernel returns the same, correct results
with or without them, so a replay or stress test cannot tell them apart. The
test compiles the kernel on the host (no GPU needed) and inspects the PTX.
"""

import re

import pytest

# K = 5120 is 20 tiles of 256 over a 16-stage ring, so stages are reused.
_N, _K = 1024, 5120
# (output_tile, token_tile, k_tile, stages, b_loader_warps)
_TACTIC = (16, 8, 256, 16, 2)
_LOADER_THREADS = (2 + _TACTIC[4]) * 32
_COMPUTE_THREADS = 4 * 32


@pytest.fixture(scope="module", params=["sm_100a", "sm_103a"])
def ptx(request, tmp_path_factory) -> str:
    from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available

    if not is_cute_dsl_experimental_available():
        pytest.skip("nvidia-cutlass-dsl with cutlass.experimental is not available.")
    import cutlass
    from cutlass import cute
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from flashinfer.gemm.kernels.dense_mxfp8_gemm_warp_splitk import (
        CpAsyncMxFp8WarpSplitKKernel,
        MxFp8WarpSplitKTactic,
        _swizzled_scale_len,
    )

    # Match the operands _compile passes: two E4M3 bytes viewed as one Int16,
    # 128x4 swizzled uint8 scale arrays, and a dynamic 32-bit M / activation scale length.
    i16 = cutlass.Int16
    u8 = cutlass.Uint8
    bf16 = cutlass.BFloat16
    weight = make_fake_tensor(i16, (_N, _K // 2), (_K // 2, 1), assumed_align=32)
    activation = make_fake_tensor(
        i16, (cute.sym_int32(), _K // 2), (_K // 2, 1), assumed_align=32
    )
    weight_sf = make_fake_tensor(
        u8, (_swizzled_scale_len(_N, _K),), (1,), assumed_align=4
    )
    activation_sf = make_fake_tensor(u8, (cute.sym_int32(),), (1,), assumed_align=4)
    output = make_fake_tensor(bf16, (cute.sym_int32(), _N), (_N, 1), assumed_align=32)
    dump_dir = tmp_path_factory.mktemp(request.param)
    compiled = cute_ext.compile(
        CpAsyncMxFp8WarpSplitKKernel(MxFp8WarpSplitKTactic(*_TACTIC), True),
        weight,
        activation,
        weight_sf,
        activation_sf,
        output,
        make_fake_stream(),
        options=f"--gpu-arch {request.param} --keep-ptx --dump-dir {dump_dir}",
    )
    assert compiled.__ptx__ is not None
    return compiled.__ptx__


def test_output_stores_wait_for_prerequisite_grids(ptx):
    # Under PDL only the loader warps execute griddepcontrol.wait; the warp
    # that stores the output must wait too (after the split-K reduction
    # barrier, before its first store) for its stores to be ordered after the
    # prerequisite grids, whose memory `out` may reuse.
    first_store = ptx.index("st.global")
    barriers = [m.end() for m in re.finditer(r"\bbar(?:rier)?\.sync\b", ptx)]
    reduction = max(i for i in barriers if i < first_store)
    assert "griddepcontrol.wait" in ptx[reduction:first_store]


def test_every_compute_lane_releases_the_stage(ptx):
    # Every compute lane reads a stage with ldmatrix and shared scale loads
    # before the loaders may overwrite it. mbarrier.try_wait orders only the
    # arriving threads' prior accesses, so every lane arrives on the empty
    # barrier (count 128) rather than one elected lane per warp (count 4).
    counts, movs = [], {}
    for line in ptx.splitlines():
        mov = re.search(r"mov\.[bsu]32\s+(%r\d+),\s*(\d+);", line)
        if mov:
            movs[mov.group(1)] = int(mov.group(2))
        init = re.search(r"mbarrier\.init\.shared\.b64\s+\[.+\],\s*(%r\d+|\d+);", line)
        if init:
            count = init.group(1)
            counts.append(int(count) if count.isdigit() else movs[count])
    # Full barrier (every loader thread), then empty barrier (every compute lane).
    assert counts == [_LOADER_THREADS, _COMPUTE_THREADS]
    assert "elect.sync" not in ptx


def test_loaders_drain_cp_async_before_exit(ptx):
    # The weight and activation loader threads exit right after their last
    # cp.async.mbarrier.arrive; the PTX ISA does not define a thread exiting
    # with cp.async copies in flight, so each loader path drains them first.
    assert ptx.count("cp.async.wait_all") >= 2


if __name__ == "__main__":
    pytest.main([__file__])
