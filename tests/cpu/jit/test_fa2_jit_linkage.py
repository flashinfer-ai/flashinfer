"""FA2 JIT link/load regressions; requires Linux, nvcc and a CUDA driver or stub.

No kernels are launched. On a host without GPUs, set FLASHINFER_CUDA_ARCH_LIST
and put a CUDA stub libcuda.so.1 on LD_LIBRARY_PATH before running pytest.
"""

import shutil
import subprocess
import sys

import pytest
import torch

from flashinfer.jit import env as jit_env
from flashinfer.jit.attention import modules

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def jit_workspace(tmp_path, monkeypatch):
    if sys.platform != "linux" or shutil.which("nvcc") is None:
        pytest.skip("requires the Linux CUDA toolkit")
    monkeypatch.delenv("FLASHINFER_EXTRA_LDFLAGS", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "compiled")
    return tmp_path


def _check_link_and_load(spec):
    ldflags = list(spec.extra_ldflags or [])
    # Relink the same objects in both modes, loading in separate processes so a
    # previously loaded DSO cannot hide a missing symbol after relinking.
    for flags in (["-Wl,-z,now"], []):
        spec.extra_ldflags = ldflags + flags
        spec.build()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, tvm_ffi; tvm_ffi.load_module(sys.argv[1])",
                str(spec.jit_library_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        undefined = subprocess.check_output(
            ["nm", "-D", "--undefined-only", "--demangle", str(spec.jit_library_path)],
            text=True,
        )
        assert "flashinfer::" not in undefined, undefined


@pytest.mark.parametrize(
    "factory",
    [
        modules.gen_batch_prefill_module,
        modules._gen_batch_prefill_primary_module,
        modules._gen_batch_prefill_independent_paged_module,
    ],
    ids=["runtime", "equal", "independent"],
)
@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo", [(64, 64), (256, 256), (512, 512), (512, 256)]
)
def test_batch_prefill_linkage(jit_workspace, factory, head_dim_qk, head_dim_vo):
    spec = factory(
        "fa2",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        head_dim_qk,
        head_dim_vo,
        0,
        False,
        False,
        False,
    )
    _check_link_and_load(spec)


def test_batch_prefill_fp16_linkage(jit_workspace):
    spec = modules.gen_batch_prefill_module(
        "fa2",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        0,
        False,
        False,
        False,
    )
    _check_link_and_load(spec)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_pod_linkage(jit_workspace, dtype):
    spec = modules.gen_batch_pod_module(
        dtype,
        dtype,
        dtype,
        128,
        0,
        False,
        False,
        False,
        torch.int32,
        0,
        False,
        False,
    )
    _check_link_and_load(spec)


def test_fa2_cta_tile_dispatch(jit_workspace):
    source = jit_workspace / "dispatch.cu"
    source.write_text(
        r"""
#include <flashinfer/utils.cuh>
#include <cassert>
#include <initializer_list>

using namespace flashinfer;

template <uint32_t QK, uint32_t VO>
void CheckDispatch(std::initializer_list<uint32_t> supported) {
  for (uint32_t tile : {0, 16, 32, 64, 128, 256}) {
    bool expected = false;
    for (auto allowed : supported) expected |= tile == allowed;
    uint32_t observed = 0;
    bool rejected = false;
    try {
      DISPATCH_FA2_CTA_TILE_Q(tile, QK, VO, TILE, { observed = TILE; });
    } catch (const flashinfer::Error&) {
      rejected = true;
    }
    assert(rejected != expected);
    if (expected) assert(observed == tile);
  }
}

int main() {
  CheckDispatch<64, 64>({16, 64, 128});
  CheckDispatch<256, 256>({16, 64, 128});
  CheckDispatch<496, 256>({16, 64, 128});
  CheckDispatch<512, 256>({16});
  CheckDispatch<512, 512>({16, 32});
  // These planner paths return before any CUDA device query.
  for (int64_t n : {1, 16, 17, 32, 33, 64, 65, 128}) {
    assert(FA2DetermineCtaTileQ(n, 512, 512) == (n <= 32 ? 16 : 32));
    assert(FA2DetermineCtaTileQ(n, 256, 512) == 16);
  }
  assert(FA2DetermineCtaTileQ(65, 64, 64) == 128);
}
"""
    )
    executable = jit_workspace / "dispatch"
    subprocess.run(
        [
            "nvcc",
            "-std=c++17",
            f"-I{jit_env.FLASHINFER_INCLUDE_DIR}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
