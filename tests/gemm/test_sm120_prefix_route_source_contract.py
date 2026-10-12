"""CPU-only source contracts for the SM120 fused-prefix A/B seam.

The seam spans three files: a C++ ladder header, the TVM-FFI prefix gate, and the
producer dispatcher. Python mirrors the ladder because the M537 native producer
reads a prepacked LoRA-down matrix that the cuBLASLt prefix does not, so a silent
disagreement between the two sides would corrupt results rather than slow them
down. These tests pin the agreement without a GPU.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER_HEADER = REPO_ROOT / "include/flashinfer/gemm/svdquant_sm120_prefix_route.h"
PRODUCER_HEADER = (
    REPO_ROOT / "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
)
LINEAR_SOURCE = REPO_ROOT / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu"

_LADDER_ENTRY = re.compile(
    r"\{\s*(\d+),\s*(\d+),\s*Route::k(\w+),\s*(true|false),\s*(true|false),"
    r"\s*(true|false)\s*\}"
)
_ROUTE_NAMES = {"Cublaslt": "cublaslt", "Native": "native", "NativeAlt": "native_alt"}
# The ladder lives in flashinfer::gemm; callers outside it must say so in full.
_QUALIFIED = "flashinfer::gemm::svdquant_sm120_prefix_route::"

# Mirrors KernelLaunchGeometry in the producer header.
_SF_VEC_SIZE = 16
_RANK = 32
_LARGE_M_DOWN_WARPS = _RANK // 8
# One wave on the SM count both SM120 parts in use here happen to have.
_ONE_WAVE_SM_COUNT = 110


def _large_m_geometry(m: int, k: int, block_threads: int, tile_m: int, tile_k: int):
    """Recompute the producer header's compile-time launch contract in Python."""
    quant_threads = block_threads - _LARGE_M_DOWN_WARPS * 32
    sf_cols_per_tile = tile_k // _SF_VEC_SIZE
    padded_m = (m + 127) // 128 * 128
    grid_blocks = (padded_m + tile_m - 1) // tile_m
    prefetch_lines = k * _RANK * 2 // 128
    shared_bytes = (
        2 * tile_m * (tile_k + 8) * 2
        + 2 * tile_k * 2
        + 2 * (tile_m * sf_cols_per_tile if k >= 8192 else 1)
    )
    assert k % tile_k == 0
    assert tile_k % (4 * _SF_VEC_SIZE) == 0
    assert 0 < tile_m <= 80 and tile_m % 16 == 0
    assert block_threads % 32 == 0 and quant_threads > 0
    assert (tile_m * sf_cols_per_tile) % quant_threads == 0
    assert k < 8192 or grid_blocks * block_threads >= prefetch_lines
    assert shared_bytes <= 48 * 1024
    return grid_blocks, shared_bytes


@pytest.mark.parametrize(
    "tile_m,expected_blocks",
    [(32, 64), (16, 128)],
)
def test_row57_producer_underfill_was_real_but_not_the_cause(
    tile_m: int,
    expected_blocks: int,
) -> None:
    # Given: the underfill hypothesis for row 57 -- a 32-row tile leaves 64 blocks
    # on a 110-SM device -- was arithmetically sound and still lost.
    # When: both geometries are re-derived from the producer launch contract.
    # Then: each is legal and only the 16-row tile fills the device, which pins
    # the refutation to the K14336 per-block LoRA-down re-read, not to legality.
    grid_blocks, shared_bytes = _large_m_geometry(1935, 14336, 384, tile_m, 256)
    assert grid_blocks == expected_blocks
    assert shared_bytes <= 48 * 1024
    assert (grid_blocks >= _ONE_WAVE_SM_COUNT) is (tile_m == 16)


def test_row57_is_the_only_underfilled_batch4_producer_grid() -> None:
    # Given: underfill is the one structural producer asymmetry in Batch 4.
    # When: every Batch-4 large-M producer grid is re-derived.
    # Then: only row 57's geometry leaves the device in a partial wave, so no
    # other row can be attributed to the same cause without new evidence.
    underfilled = {
        (m, k)
        for m, k, block_threads, tile_m, tile_k in (
            (1935, 14336, 384, 32, 256),
            (6913, 5120, 192, 32, 128),
            (6913, 5376, 192, 32, 128),
            (6913, 7168, 256, 32, 256),
            (6913, 14336, 384, 32, 256),
        )
        if _large_m_geometry(m, k, block_threads, tile_m, tile_k)[0]
        < _ONE_WAVE_SM_COUNT
    }
    assert underfilled == {(1935, 14336)}


def _run_ubsan(tmp_path: Path, source: str) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    program = tmp_path / "bounds.cpp"
    program.write_text(source)
    binary = tmp_path / "bounds"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(program),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("rank", (32, 64))
def test_dynamic_prefix_offsets_cross_int32_boundary(tmp_path: Path, rank: int) -> None:
    # Compile the production expressions so UBSan checks C++ promotion rules,
    # without allocating multi-gigabyte CUDA tensors to reach the boundary.
    source = PRODUCER_HEADER.read_text()
    small = source.split(
        "void nvfp4_smooth_quantize_lora_down_small_m_dyn_sm120_kernel(", 1
    )[1]
    small = small.split("// Runtime-M/K small-M launch.", 1)[0]
    large = source.split(
        "void nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel(", 1
    )[1]
    large = large.split("inline cudaError_t launch_large_m_dyn_kernel", 1)[0]

    def expression(body: str, pattern: str) -> str:
        match = re.search(pattern, body, re.DOTALL)
        assert match is not None, pattern
        return match[1]

    cases = [
        (
            expression(small, r"a_base\s*=\s*x\s*\+\s*(.*?);"),
            "int a_row = 131072;",
            2147483670,
        ),
        (
            expression(small, r"a_upper\s*=\s*x\s*\+\s*(.*?);"),
            "int a_row = 131064;",
            2147483670,
        ),
        (
            expression(small, r"b_base\s*=\s*l2t_smoothed\s*\+\s*(.*?);"),
            f"k_base = {2147483648 // rank}; pair_col = 0;",
            2147483655,
        ),
        (expression(small, r"down\[(.*?)\]\s*="), "", 2147483655),
        (expression(large, r"b_base\s*=\s*l2t_smoothed\s*\+\s*(.*?);"), "", 2147483655),
        (
            expression(small, r"kL2PrefetchLines\s*=\s*(.*?);"),
            "K = 33554432;",
            33554432 * rank * 2 // 128,
        ),
        (
            expression(large, r"kL2Bytes\s*=\s*(.*?);"),
            "K = 33554432;",
            33554432 * rank * 2,
        ),
        (
            expression(small, r"packed_index\s*=\s*(.*?);"),
            "k_base = 536870912;",
            2147483648,
        ),
    ]
    down_offsets = re.findall(r"store_down_pair<[^>]+>\(down\s*\+\s*(.*?),", large)
    assert down_offsets
    cases.extend((offset, "", 2147483655) for offset in down_offsets)
    checks = []
    for offset, setup, expected in cases:
        checks.append(
            "{\n"
            "int K = input, k_base = 16, pair_col = 6, b_col = 7, col = 7;\n"
            "int kDownNTiles = 1, kNFragments = 2, tile_n = 0, fragment = 0, lane = 0;\n"
            f"int row = {2147483648 // rank}, b_row = row;\n"
            "int row0 = row, row1 = row, row2 = row, row3 = row, row4 = row;\n"
            "int row5 = row, row6 = row, row7 = row, row8 = row, row9 = row;\n"
            f"{setup}\n"
            f"std::int64_t actual = {offset};\n"
            f"if (actual != {expected}LL) return __LINE__;\n"
            "}\n"
        )
    _run_ubsan(
        tmp_path,
        "#include <cstdint>\n"
        "namespace quant { using Type = std::uint16_t; }\n"
        f"constexpr int kRank = {rank}, kL2PrefetchBytes = 128;\n"
        "int main() { volatile int input = 16384;\n" + "".join(checks) + "}\n",
    )


def test_gemm_shape_bounds_precede_layout_and_workspace_construction(
    tmp_path: Path,
) -> None:
    source = (
        REPO_ROOT / "include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
    ).read_text()
    guard = source.split(
        "template <class Config>\ninline bool runtime_tactic_feasible", 1
    )[1]
    guard = (
        "template <class Config>\ninline bool runtime_tactic_feasible"
        + guard.split("// Keep CUTLASS's LayoutSF", 1)[0]
    )
    for entry in (
        "workspace_size_for_tactic",
        "can_implement_tactic",
        "run_tactic_impl",
    ):
        body = source.split(f"{entry}(RuntimeTacticSm120", 1)[1]
        assert body.index("runtime_tactic_feasible<Config>(rt, m, n, k)") < body.index(
            "using Gemm"
        )
    # Only the compile-time tile/config metadata is stubbed; the admission function is real.
    prelude = r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
struct StreamKTag {};
struct PersistentTag {};
struct RuntimeTacticSm120 { int splits, swizzle; };
namespace cute {
template<class A, class B> constexpr bool is_same_v = std::is_same_v<A, B>;
template<int I, class Shape> constexpr int size(Shape) { return I == 1 ? 64 : 128; }
}
template<bool Swap, class Scheduler> struct Config {
  struct MmaTileShape {};
  static constexpr bool SwapAB = Swap;
  using TileSchedulerTag = Scheduler;
  struct CollectiveMainloop {
    struct Sm1xxBlkScaledConfig { static constexpr int SFVecSize = 16; };
  };
};
"""
    checks = r"""
template<bool Swap> void check() {
  using P = Config<Swap, PersistentTag>;
  using S = Config<Swap, StreamKTag>;
  RuntimeTacticSm120 rt{1, 1};
  int const limit = std::numeric_limits<int>::max();
  assert(runtime_tactic_feasible<P>(rt, 131088, 128, 16384));
  assert(runtime_tactic_feasible<S>(rt, 131088, 128, 16384));
  assert(runtime_tactic_feasible<P>(rt, 1048448, 128, 32768));
  assert(!runtime_tactic_feasible<P>(rt, 1048576, 128, 32768));
  assert(!runtime_tactic_feasible<P>(rt, 128, 1048576, 32768));
  assert(!runtime_tactic_feasible<P>(rt, limit, 128, 128));
  assert(!runtime_tactic_feasible<P>(rt, 128, limit, 128));
  assert(!runtime_tactic_feasible<P>(rt, 128, 128, limit - 127));
  assert(!runtime_tactic_feasible<P>(rt, -1, 128, 128));
  assert(!runtime_tactic_feasible<P>(rt, 128, 128, 0));
  assert(!runtime_tactic_feasible<P>(rt, 8388608, 8388608, 64));
  assert(runtime_tactic_feasible<P>(rt, 1048576, 1048576, 4096));
  assert(!runtime_tactic_feasible<S>(rt, 1048576, 1048576, 4096));
  rt.splits = 32;
  assert(runtime_tactic_feasible<S>(rt, 128, 128, 4096));
  rt.splits = 33;
  assert(!runtime_tactic_feasible<S>(rt, 128, 128, 4096));
}
int main() { check<false>(); check<true>(); }
"""
    _run_ubsan(tmp_path, prelude + guard + checks)
