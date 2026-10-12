"""SM120 cuBLASLt plan lifetime and CUDA Graph capture guards."""

import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

_BINDING_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "nvfp4_svdquant_gemm_cutlass_sm120.cu"
)


def test_persisting_l2_hints_are_skipped_during_stream_capture() -> None:
    source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "cudaStreamIsCapturing(stream, &capture_status)" in source
    assert "state.persisting_l2t_stream != stream" in source
    assert "if (!is_capturing)" in source

    linear = source[source.index("void nvfp4_svdquant_linear_sm120(") :]
    linear = linear[: linear.index("\n#endif")]
    assert "configure_case7_persisting_l2 = !is_capturing" in linear
    assert linear.count("if (configure_case7_persisting_l2)") == 3


@pytest.fixture(scope="module")
def plan_cache_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = _BINDING_PATH.read_text(encoding="utf-8")
    # Compile the actual ownership and cache code; only the vendor API is faked.
    cache = source[
        source.index("struct LoraDownCublasLtState {") : source.index(
            "void run_lora_down_cublaslt("
        )
    ]
    program = r"""
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
using cudaStream_t = void*;
using cudaError_t = int;
using cublasStatus_t = int;
using cublasLtHandle_t = int*;
using cublasLtMatmulDesc_t = int*;
using cublasLtMatrixLayout_t = int*;
using cublasLtMatmulPreference_t = int*;
constexpr int cudaSuccess = 0, CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_COMPUTE_32F = 0, CUDA_R_32F = 0, CUDA_R_16BF = 0;
constexpr int CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 0;
struct cublasLtMatmulHeuristicResult_t { int state = 0; int algo = 0; };
std::atomic<int> live_resources{0};
std::atomic<int> resource_ids{0};
thread_local int current_device = 0, api_calls = 0, fail_on_call = -1;
int cudaGetDevice(int* device) { *device = current_device; return cudaSuccess; }
int next_status() { return ++api_calls == fail_on_call ? 1 : 0; }
int create(int** resource) {
  if (int status = next_status()) return status;
  *resource = new int(++resource_ids);
  ++live_resources;
  return 0;
}
int destroy(int* resource) { delete resource; --live_resources; return 0; }
int cublasLtCreate(int** resource) { return create(resource); }
int cublasLtDestroy(int* resource) { return destroy(resource); }
template <class... Args>
int cublasLtMatmulDescCreate(int** resource, Args...) { return create(resource); }
int cublasLtMatmulDescDestroy(int* resource) { return destroy(resource); }
template <class... Args>
int cublasLtMatrixLayoutCreate(int** resource, Args...) { return create(resource); }
int cublasLtMatrixLayoutDestroy(int* resource) { return destroy(resource); }
int cublasLtMatmulPreferenceCreate(int** resource) { return create(resource); }
int cublasLtMatmulPreferenceDestroy(int* resource) { return destroy(resource); }
template <class... Args>
int cublasLtMatmulPreferenceSetAttribute(Args...) { return next_status(); }
int cublasLtMatmulAlgoGetHeuristic(int*, int*, int*, int*, int*, int*, int*,
                                 int count, cublasLtMatmulHeuristicResult_t* results,
                                 int* returned) {
  if (int status = next_status()) return status;
  *returned = count;
  for (int i = 0; i < count; ++i) results[i].algo = i;
  return 0;
}
struct Check {
  explicit Check(bool condition) { if (!condition) throw std::runtime_error("vendor failure"); }
  template <class T> Check& operator<<(T const&) { return *this; }
};
#define TVM_FFI_ICHECK(condition) Check(condition)
#define TVM_FFI_ICHECK_EQ(a, b) Check((a) == (b))
#define TVM_FFI_ICHECK_GT(a, b) Check((a) > (b))
"""
    program += cache
    program += r"""
void growth() {
  std::vector<LoraDownCublasLtState*> original;
  for (int i = 0; i < 24; ++i) {
    current_device = i % 2;
    auto& state = lora_down_cublaslt_state(current_device, 64 * (i / 2 + 1), 3072);
    assert(state.initialized && state.device_id == current_device);
    original.push_back(&state);
  }
  int const warmed_calls = api_calls;
  for (int i = 23; i >= 0; --i) {
    current_device = i % 2;
    assert(original[i] == &lora_down_cublaslt_state(current_device, 64 * (i / 2 + 1), 3072));
  }
  assert(api_calls == warmed_calls);
  auto& different_k = lora_down_cublaslt_state(current_device, 64, 6144);
  assert(&different_k != original[0]);
  assert(live_resources == 25 * 6);
}
void retry(int stage) {
  auto& warmed = lora_down_cublaslt_state(0, 64, 3072);
  int const warmed_handle = *warmed.handle;
  int const before = live_resources;
  fail_on_call = api_calls + stage;
  bool failed = false;
  try { lora_down_cublaslt_state(0, 128, 3072); }
  catch (std::runtime_error const&) { failed = true; }
  assert(failed && live_resources == before);
  assert(&warmed == &lora_down_cublaslt_state(0, 64, 3072));
  assert(*warmed.handle == warmed_handle);
  fail_on_call = -1;
  auto& recovered = lora_down_cublaslt_state(0, 128, 3072);
  assert(recovered.initialized && live_resources == before + 6);
  int const warmed_calls = api_calls;
  assert(&recovered == &lora_down_cublaslt_state(0, 128, 3072));
  assert(api_calls == warmed_calls);
}
int main(int argc, char** argv) {
  assert(argc == 2);
  int const stage = std::stoi(argv[1]);
  std::string failure;
  // Thread exit must release all plans; another thread starts with an empty cache.
  for (int repetition = 0; repetition < 2; ++repetition) {
    std::thread worker([&] {
      try { if (stage == 0) growth(); else retry(stage); }
      catch (std::runtime_error const& error) { failure = error.what(); }
    });
    worker.join();
    assert(live_resources == 0);
    if (!failure.empty()) { std::cerr << failure << '\n'; return 1; }
  }
}
"""
    directory = tmp_path_factory.mktemp("cublaslt-plan-cache")
    cpp = directory / "cache.cpp"
    binary = directory / "cache"
    cpp.write_text(program, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
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


@pytest.mark.parametrize(
    "failure_stage", range(9), ids=("growth", *(f"retry-{i}" for i in range(1, 9)))
)
def test_cublaslt_plan_cache_growth_and_failure_cleanup(
    plan_cache_binary: Path, failure_stage: int
) -> None:
    result = subprocess.run(
        [str(plan_cache_binary), str(failure_stage)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_cublaslt_graph_replays_after_more_than_eight_shapes() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    from flashinfer.gemm.svdquant_sm120_cutlass import get_nvfp4_svdquant_sm120_module

    from .test_nvfp4_svdquant_gemm import _sqnr_db

    torch.manual_seed(0)
    module = get_nvfp4_svdquant_sm120_module(32)
    k, rank = 256, 32
    workspace = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    pqs = torch.ones(k, dtype=torch.bfloat16, device="cuda")
    scale = torch.ones(1, device="cuda")
    l2t = torch.randn((k, rank), dtype=torch.bfloat16, device="cuda")
    inputs = [
        (
            torch.randn((m, k), dtype=torch.bfloat16, device="cuda"),
            torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
            torch.zeros(
                ((m + 127) // 128 * 128 * (k // 16),), dtype=torch.uint8, device="cuda"
            ),
            torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        )
        for m in range(64, 64 * 13, 64)
    ]

    def launch(index: int) -> None:
        x, xq, sf, down = inputs[index]
        module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
            x, pqs, scale, l2t, xq, sf, down, workspace
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        launch(0)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(0)

    for index in range(1, len(inputs)):
        launch(index)
    torch.cuda.synchronize()
    for x, _, _, down in inputs[1:]:
        assert _sqnr_db(x.float() @ l2t.float(), down.float()) > 40

    x, _, _, down = inputs[0]
    x.copy_(torch.randn_like(x))
    l2t.copy_(torch.randn_like(l2t))
    down.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert _sqnr_db(x.float() @ l2t.float(), down.float()) > 40
