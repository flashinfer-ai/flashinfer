"""Synchronization of the warp split-K BF16 GEMM
(flashinfer/gemm/kernels/dense_bf16_gemm_warp_splitk.py, a runner of
``mm_bf16(backend="cute-dsl")``), checked in its generated PTX.

These are memory-model orderings: the kernel returns the same, correct results
with or without them, so a replay or stress test cannot tell them apart. The
test compiles the kernel on the host (no GPU needed) and inspects the PTX.
"""

import re

import pytest

# K = 6144 is 24 tiles of 256 over a 16-stage ring, so stages are reused.
_N, _K = 1024, 6144
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

    from flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk import (
        CpAsyncWarpSplitKKernel,
        WarpSplitKTactic,
    )

    # The operands run_warp_splitk_dense passes: row-major, with a dynamic
    # 32-bit M (the PTX then matches the runtime JIT build).
    bf16 = cutlass.BFloat16
    weight = make_fake_tensor(bf16, (_N, _K), (_K, 1), assumed_align=32)
    activation = make_fake_tensor(
        bf16, (cute.sym_int32(), _K), (_K, 1), assumed_align=32
    )
    bias = make_fake_tensor(bf16, (_N,), (1,), assumed_align=32)
    output = make_fake_tensor(bf16, (cute.sym_int32(), _N), (_N, 1), assumed_align=32)
    dump_dir = tmp_path_factory.mktemp(request.param)
    compiled = cute_ext.compile(
        CpAsyncWarpSplitKKernel(WarpSplitKTactic(*_TACTIC), True, False),
        weight,
        activation,
        bias,
        output,
        make_fake_stream(),
        options=f"--gpu-arch {request.param} --keep-ptx --dump-dir {dump_dir}",
    )
    assert compiled.__ptx__ is not None
    return compiled.__ptx__


def test_output_stores_wait_for_prerequisite_grids(ptx):
    # Under PDL only the activation loaders execute griddepcontrol.wait; the
    # warp that stores the output must wait too (after the split-K reduction
    # barrier, before its first store) for its stores to be ordered after the
    # prerequisite grids, whose memory `out` may reuse.
    first_store = ptx.index("st.global")
    barriers = [m.end() for m in re.finditer(r"\bbar(?:rier)?\.sync\b", ptx)]
    reduction = max(i for i in barriers if i < first_store)
    assert "griddepcontrol.wait" in ptx[reduction:first_store]


def test_every_compute_lane_releases_the_stage(ptx):
    # Every compute lane reads a stage with ldmatrix before the loaders may
    # overwrite it. mbarrier.try_wait orders only the arriving threads' prior
    # accesses, so every lane arrives on the empty barrier (count 128) rather
    # than one elected lane per warp (count 4).
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
