"""CPU UBSan coverage of SM120 FFI admission and quantizer size arithmetic.

The backend stub records forwarded arguments; CUTLASS feasibility and CUDA
execution are deliberately outside this host-only boundary test.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest


REPO_ROOT: Final = Path(__file__).resolve().parents[2]
GEMM_SOURCE: Final = REPO_ROOT / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu"
QUANT_SOURCE: Final = REPO_ROOT / "csrc/nvfp4_smooth_quantize_sm100.cu"
QUANT_HEADER: Final = (
    REPO_ROOT / "include/flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"
)


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (source[closing] == "{") - (source[closing] == "}")
        closing += 1
    return source[start:closing]


@pytest.fixture(scope="module")
def bounds_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    source = GEMM_SOURCE.read_text()
    signatures = [
        "inline bool fits_nonnegative_int(",
        "inline void check_sm120_tactic(",
        "bool nvfp4_svdquant_gemm_can_implement(int64_t",
        "int64_t nvfp4_svdquant_gemm_fallback_tactic(int64_t",
        "int64_t nvfp4_svdquant_gemm_workspace_size(int64_t",
        "int64_t nvfp4_svdquant_gemm_shared_storage_size(int64_t",
        "int64_t nvfp4_svdquant_gemm_tactic_row(int64_t",
    ]
    # Optional helpers let the same behavioral harness run against the pre-fix source.
    functions = "\n".join(
        _function(source, signature) for signature in signatures if signature in source
    )
    quant_source = QUANT_SOURCE.read_text()
    sf_functions = []
    for name in ("nvfp4_quantize_smooth", "check_sm120_prefix_inputs"):
        body = _function(quant_source, f"void {name}(")
        initializer = re.search(r"int64_t const sfSize\s*=\s*(.*?);", body, re.DOTALL)
        assert initializer is not None
        sf_functions.append(
            f"int64_t {name}_sf(int m, int n, int k) {{ return {initializer[1]}; }}"
        )
    quant_header = QUANT_HEADER.read_text()
    padding = re.search(r"__host__\s+__device__\s+inline\s+\w+\s+padUp\(", quant_header)
    assert padding is not None
    program = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#define __host__
#define __device__
struct Check {
  explicit Check(bool condition) { if (!condition) throw std::out_of_range("FFI bounds"); }
  template <class T> Check& operator<<(T const&) { return *this; }
};
#define TVM_FFI_ICHECK(condition) Check(condition)
std::array<int64_t, 5> expected{128, 128, 128, 32, 0};
int backend_calls = 0;
namespace flashinfer::gemm {
namespace svdquant_sm120 {
constexpr int kNvfp4SvdquantGemmSm120NumTactics = 83;
struct RuntimeTactic { int kernel_id, splits, raster, swizzle; };
RuntimeTactic decode_runtime_tactic(int tactic) {
  ++backend_calls;
  if (tactic < 0 || tactic >= kNvfp4SvdquantGemmSm120NumTactics)
    throw std::out_of_range("backend tactic");
  return {tactic, 1, 0, 1};
}
bool kernel_is_streamk_sm120(int) { return false; }
}
void check_dimensions(int m, int n, int k) {
  ++backend_calls;
  assert(m == expected[0] && n == expected[1] && k == expected[2]);
}
bool nvfp4_svdquant_gemm_can_implement(int m, int n, int k, int rank, int tactic) {
  check_dimensions(m, n, k);
  assert(rank == expected[3] && tactic == expected[4]);
  return true;
}
int nvfp4_svdquant_gemm_fallback_tactic(int m, int n, int k, int rank) {
  check_dimensions(m, n, k);
  assert(rank == expected[3]);
  return 0;
}
size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k, int tactic) {
  check_dimensions(m, n, k);
  assert(tactic == expected[4]);
  return 64;
}
size_t nvfp4_svdquant_gemm_shared_storage_size(int tactic) {
  ++backend_calls;
  assert(tactic == expected[4]);
  return 32;
}
}
namespace boundary {
"""
    program += (
        functions
        + "\n"
        + _function(quant_header, padding[0])
        + "\n"
        + "\n".join(sf_functions)
    )
    program += r"""
}
template <class F> void require_rejection(F call) {
  bool rejected = false;
  try { call(); } catch (std::out_of_range const&) { rejected = true; }
  assert(rejected);
}
int main(int argc, char** argv) {
  assert(argc == 4);
  int const scenario = std::stoi(argv[1]), axis = std::stoi(argv[2]);
  int64_t const value = std::stoll(argv[3]);
  expected[axis] = value;
  auto const [m, n, k, rank, tactic] = expected;
  using namespace boundary;
  auto can_implement = [&] { return nvfp4_svdquant_gemm_can_implement(m, n, k, rank, tactic); };
  auto workspace = [&] { return nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic); };
  switch (scenario) {
    case 0:
      assert(!can_implement());
      assert(nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank) == -1);
      if (axis != 3) require_rejection(workspace);
      assert(backend_calls == 0);
      break;
    case 1:
      require_rejection(can_implement);
      require_rejection(workspace);
      require_rejection([&] { nvfp4_svdquant_gemm_shared_storage_size(tactic); });
      require_rejection([&] { nvfp4_svdquant_gemm_tactic_row(tactic); });
      assert(backend_calls == 0);
      break;
    case 2:
      assert(can_implement());
      assert(nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank) == 0);
      assert(workspace() == 64);
      assert(nvfp4_svdquant_gemm_shared_storage_size(tactic) == 32);
      assert(nvfp4_svdquant_gemm_tactic_row(tactic) == (tactic | (1 << 8) | (1 << 24)));
      assert(backend_calls == 5);
      break;
    case 3: {
      int const rows = static_cast<int>(value);
      int64_t const padded = (value / 128 + (value % 128 != 0)) * 128;
      assert(padUp(rows, 128) == padded);
      assert(padUp(rows, 64) == (value / 64 + (value % 64 != 0)) * 64);
      assert(nvfp4_quantize_smooth_sf(rows, 128, 128) == padded * 8);
      assert(check_sm120_prefix_inputs_sf(rows, 128, 128) == padded * 8);
      break;
    }
    default: assert(false);
  }
}
"""
    directory = tmp_path_factory.mktemp("sm120_integer_bounds")
    cpp = directory / "bounds.cpp"
    binary = directory / "bounds"
    cpp.write_text(program)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _run_case(binary: Path, arguments: tuple[int, int, int]) -> None:
    result = subprocess.run(
        [str(binary), *map(str, arguments)],
        cwd=binary.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("axis", range(4), ids=("m", "n", "k", "rank"))
@pytest.mark.parametrize("value", (-1, 2**31, 2**32 + 128, 2**32 + 32, 2**63 - 1))
def test_ffi_queries_reject_dimensions_before_narrowing(
    bounds_binary: Path, axis: int, value: int
) -> None:
    _run_case(bounds_binary, (0, axis, value))


@pytest.mark.parametrize("tactic", (-(2**32), -1, 83, 2**32, 2**32 + 1, 2**63 - 1))
def test_ffi_queries_reject_tactics_before_narrowing(
    bounds_binary: Path, tactic: int
) -> None:
    _run_case(bounds_binary, (1, 4, tactic))


@pytest.mark.parametrize("axis", range(4), ids=("m", "n", "k", "rank"))
@pytest.mark.parametrize("value", (0, 128, 2**31 - 1))
def test_ffi_queries_preserve_representable_arguments(
    bounds_binary: Path, axis: int, value: int
) -> None:
    _run_case(bounds_binary, (2, axis, value))


@pytest.mark.parametrize("rows", (0, 1, 128, 2**31 - 128, 2**31 - 127, 2**31 - 1))
def test_scale_padding_preserves_int32_boundary(bounds_binary: Path, rows: int) -> None:
    _run_case(bounds_binary, (3, 0, rows))


@pytest.mark.parametrize("entry", ("can_implement_tactic", "run_tactic_impl"))
def test_cutlass_scale_layout_uses_wide_intermediates(
    tmp_path: Path, entry: str
) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    nvcc = shutil.which("nvcc")
    cuda_root = Path(
        os.environ.get(
            "CUDA_HOME", str(Path(nvcc).parent.parent) if nvcc else "/usr/local/cuda"
        )
    )
    includes = [f"-I{cuda_root}/include", f"-I{cuda_root}/include/cccl"]
    probe = subprocess.run(
        [compiler, "-std=c++17", *includes, "-x", "c++", "-fsyntax-only", "-"],
        input="#include <cuda_runtime_api.h>\n#include <cuda/std/type_traits>\n",
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.skip("requires CUDA and CCCL development headers (no GPU needed)")
    source = (
        REPO_ROOT / "include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
    ).read_text()
    helper = _function(
        source, "template <class Layout>\nconstexpr auto narrow_sf_layout("
    )
    body = _function(source, f"{entry}(RuntimeTacticSm120")
    layout_setup = body[
        body.index("auto const sf_shape") : body.index("args.mainloop.lora_rank")
    ]
    program = r"""
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cassert>
#include <cstdint>
"""
    program += (
        helper
        + r"""
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
using LayoutSF = Sm1xxBlkScaledConfig::LayoutSF;
template<bool Swap> struct Configuration { static constexpr bool SwapAB = Swap; };
template<bool Swap> void check(int m, int n, int k) {
  using Config = Configuration<Swap>;
  struct { struct { LayoutSF layout_SFA, layout_SFB; } mainloop; } args;
"""
    )
    program += (
        layout_setup
        + r"""
  auto expected_size = [=](int rows) { return (rows + 127LL) / 128 * 128 * ((k + 63LL) / 64 * 4); };
  assert(cute::cosize(args.mainloop.layout_SFA) == expected_size(Swap ? n : m));
  assert(cute::cosize(args.mainloop.layout_SFB) == expected_size(Swap ? m : n));
  if (m == 131073) {
    auto result = Swap ? args.mainloop.layout_SFB : args.mainloop.layout_SFA;
    assert(result(cute::make_coord(131072, 16368, 0)) == 134348291);
  }
  if (m == 537) {
    auto original = cute::make_shape(Swap ? n : m, Swap ? m : n, k, 1);
    assert(args.mainloop.layout_SFA == Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(original));
    assert(args.mainloop.layout_SFB == Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(original));
  }
}
int main() {
  volatile int m = 131073;
  check<false>(m, 128, 16384); check<true>(m, 128, 16384);
  check<false>(537, 5376, 5120); check<true>(537, 5376, 5120);
  check<false>(1048575, 128, 32704); check<true>(1048575, 128, 32704);
}
"""
    )
    cpp, binary = tmp_path / "layout.cpp", tmp_path / "layout"
    cpp.write_text(program)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            *includes,
            f"-I{REPO_ROOT}/3rdparty/cutlass/include",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
