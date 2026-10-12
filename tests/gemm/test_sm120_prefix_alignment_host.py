"""Host execution of native FFI guards; CUDA launches and tensor metadata are stubbed."""

import shutil
import subprocess
from pathlib import Path

import pytest


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth, closing = 1, opening + 1
    while depth:
        depth += (source[closing] == "{") - (source[closing] == "}")
        closing += 1
    return source[start:closing]


@pytest.fixture(scope="module", params=(32, 64))
def alignment_binary(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    root = Path(__file__).resolve().parents[2]
    quantizer = (root / "csrc/nvfp4_smooth_quantize_sm100.cu").read_text()
    linear = (root / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu").read_text()
    functions = "\n".join(
        _function(quantizer, f"void {name}(")
        for name in (
            "nvfp4_quantize_smooth",
            "check_sm120_prefix_inputs",
            "nvfp4_quantize_smooth_lora_down_sm120_impl",
            "nvfp4_quantize_smooth_lora_down_dyn_sm120_impl",
        )
    )
    functions += _function(
        linear, "void nvfp4_quantize_smooth_lora_down_cublaslt_sm120("
    )
    program = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Check {
  bool passed;
  std::ostringstream message;
  explicit Check(bool condition) : passed(condition) {}
  template<class T> Check& operator<<(T const& value) {
    if (!passed) message << value;
    return *this;
  }
  ~Check() noexcept(false) { if (!passed) throw std::runtime_error(message.str()); }
};
#define TVM_FFI_ICHECK(condition) Check(condition)
#define TVM_FFI_ICHECK_EQ(a, b) Check((a) == (b))
#define TVM_FFI_ICHECK_GE(a, b) Check((a) >= (b))
#define CHECK_INPUT_AND_TYPE(...)
#define CHECK_DEVICE(...)
struct Device { int device_id = 0; };
struct TensorView {
  std::vector<int64_t> shape;
  unsigned char* pointer;
  int ndim() const { return shape.size(); }
  int64_t size(int index) const { return shape.at(index); }
  int64_t numel() const {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>{});
  }
  void* data_ptr() const { return pointer; }
  Device device() const { return {}; }
};
using cudaStream_t = int;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int kSm120PrefixRank = TEST_RANK;
int backend_calls = 0;
int getMultiProcessorCount(int) { return 1; }
int get_stream(Device) { return 0; }
const char* cudaGetErrorString(int) { return "CUDA stub"; }
template<class... Args> void run_lora_down_cublaslt(Args...) { ++backend_calls; }
namespace flashinfer::gemm {
template<class... Args> void nvfp4_smooth_quantize(Args...) { ++backend_calls; }
template<int Rank, class... Args> int nvfp4_smooth_quantize_lora_down_sm120(Args...) {
  ++backend_calls;
  return cudaSuccess;
}
template<int Rank, class... Args> int nvfp4_smooth_quantize_lora_down_family_sm120(Args...) {
  ++backend_calls;
  return cudaSuccess;
}
}
"""
    program += functions
    program += r"""
int main(int argc, char** argv) {
  assert(argc == 4);
  int const entry = std::stoi(argv[1]), index = std::stoi(argv[2]), offset = std::stoi(argv[3]);
  alignas(16) std::array<unsigned char, 512> storage{};
  std::array<TensorView, 8> inputs{{
    {{64, 3072}, storage.data()}, {{3072}, storage.data() + 64},
    {{1}, storage.data() + 128}, {{3072, TEST_RANK}, storage.data() + 192},
    {{64, 1536}, storage.data() + 256}, {{128 * 192}, storage.data() + 320},
    {{64, TEST_RANK}, storage.data() + 384}, {{16}, storage.data() + 448}
  }};
  inputs.at(index).pointer += offset;
  auto const& [x, pqs, scale, l2t, xq, sf, down, workspace] = inputs;
  bool rejected = false;
  try {
    switch (entry) {
      case 0: nvfp4_quantize_smooth(x, pqs, scale, xq, sf, false); break;
      case 1: nvfp4_quantize_smooth_lora_down_sm120_impl(x, pqs, scale, l2t, xq, sf, down, false); break;
      case 2: nvfp4_quantize_smooth_lora_down_dyn_sm120_impl(x, pqs, scale, l2t, xq, sf, down, 0, 1, 1, 1, 0, false); break;
      case 3: nvfp4_quantize_smooth_lora_down_cublaslt_sm120(x, pqs, scale, l2t, xq, sf, down, workspace); break;
      default: assert(false);
    }
  } catch (std::runtime_error const& error) {
    rejected = true;
    std::string expected = std::string(index == 0 ? "x" : "pqs") + " must be 16-byte aligned";
    assert(std::string(error.what()).find(expected) != std::string::npos);
  }
  assert(rejected == (offset % 16 != 0));
  assert(backend_calls == (rejected ? 0 : (entry == 3 ? 2 : 1)));
}
"""
    directory = tmp_path_factory.mktemp(f"sm120_alignment_rank{request.param}")
    cpp, binary = directory / "alignment.cpp", directory / "alignment"
    cpp.write_text(program)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            f"-DTEST_RANK={request.param}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "entry", range(4), ids=("quantize", "fused", "dynamic", "cublaslt")
)
@pytest.mark.parametrize("index", (0, 1), ids=("x", "pqs"))
@pytest.mark.parametrize("offset", range(0, 18, 2))
def test_native_alignment_guard_before_launch(
    alignment_binary: Path, entry: int, index: int, offset: int
) -> None:
    result = subprocess.run(
        [str(alignment_binary), str(entry), str(index), str(offset)],
        cwd=alignment_binary.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
