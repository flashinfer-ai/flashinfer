"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Check the shared-memory bounds and padding of the FA2 NVFP4 SF loaders."""

import pytest
import torch

from flashinfer.jit.core import gen_jit_spec


CUDA_SOURCE = r"""
#include <flashinfer/attention/prefill.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

template <int TileRows>
struct Traits {
  using DTypeKV = __nv_fp4x2_e2m1;
  static constexpr uint32_t HEAD_DIM_QK = 128;
  static constexpr uint32_t HEAD_DIM_VO = 128;
  static constexpr uint32_t NUM_WARPS = 4;
  static constexpr uint32_t CTA_TILE_KV = TileRows;
};

template <int TileRows>
struct Storage {
  // Reserve a sentinel after the logical SF buffer. A predicated zero-fill must
  // never touch it, even when there are more participating lanes than SF words.
  alignas(16) uint8_t bytes[TileRows * 8 + 512];
  __device__ uint8_t* k_sf_smem_ptr() { return bytes; }
  __device__ uint8_t* v_sf_smem_ptr() { return bytes; }
};

template <int TileRows, bool Paged, bool ProduceV>
__global__ void stage_sf(uint8_t* input, int32_t* indices, uint8_t* output, int kv_len,
                         uint_fastdiv page_size) {
  __shared__ Storage<TileRows> storage;
  for (int i = threadIdx.x; i < sizeof(storage.bytes); i += blockDim.x) {
    storage.bytes[i] = 0xa5;
  }
  __syncthreads();
  if constexpr (Paged) {
    page_produce_kv_sf<ProduceV, Traits<TileRows>>(&storage, input, 0, kv_len, 1, 32 * 2 * 8, 8,
                                                   2 * 8, page_size, indices, 0, kv_len,
                                                   threadIdx.x / 32, threadIdx.x % 32);
  } else {
    produce_kv_sf<ProduceV, Traits<TileRows>>(&storage, input, 3, 1, 2 * 64, 64, 0, kv_len,
                                              threadIdx.x / 32, threadIdx.x % 32);
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  for (int i = threadIdx.x; i < sizeof(storage.bytes); i += blockDim.x) {
    output[i] = storage.bytes[i];
  }
}

template <int TileRows>
void launch_sf(TensorView input, TensorView indices, TensorView output, int kv_len, bool paged,
               bool produce_v) {
  auto stream = get_stream(input.device());
  auto* src = static_cast<uint8_t*>(input.data_ptr());
  auto* idx = static_cast<int32_t*>(indices.data_ptr());
  auto* dst = static_cast<uint8_t*>(output.data_ptr());
  if (paged) {
    if (produce_v)
      stage_sf<TileRows, true, true>
          <<<1, 128, 0, stream>>>(src, idx, dst, kv_len, uint_fastdiv(32));
    else
      stage_sf<TileRows, true, false>
          <<<1, 128, 0, stream>>>(src, idx, dst, kv_len, uint_fastdiv(32));
  } else {
    if (produce_v)
      stage_sf<TileRows, false, true>
          <<<1, 128, 0, stream>>>(src, idx, dst, kv_len, uint_fastdiv(32));
    else
      stage_sf<TileRows, false, false>
          <<<1, 128, 0, stream>>>(src, idx, dst, kv_len, uint_fastdiv(32));
  }
}

void run(TensorView input, TensorView indices, TensorView output, int64_t tile_rows, int64_t kv_len,
         bool paged, bool produce_v) {
  ffi::CUDADeviceGuard guard(input.device().device_id);
  switch (tile_rows) {
    case 16:
      launch_sf<16>(input, indices, output, kv_len, paged, produce_v);
      break;
    case 32:
      launch_sf<32>(input, indices, output, kv_len, paged, produce_v);
      break;
    case 64:
      launch_sf<64>(input, indices, output, kv_len, paged, produce_v);
      break;
    case 128:
      launch_sf<128>(input, indices, output, kv_len, paged, produce_v);
      break;
    default:
      TVM_FFI_ICHECK(false) << "Unsupported test tile";
  }
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""


@pytest.fixture(scope="module")
def sf_staging_module(tmp_path_factory):
    if tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8):
        pytest.skip("NVFP4 requires CUDA 12.8 or newer")
    source = tmp_path_factory.mktemp("nvfp4_sf_staging") / "staging.cu"
    source.write_text(CUDA_SOURCE)
    return gen_jit_spec("test_nvfp4_sf_staging", [source]).build_and_load()


@pytest.mark.parametrize("tile_rows", [16, 32, 64, 128])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("produce_v", [False, True])
@pytest.mark.parametrize("tail", [0, 3])
def test_nvfp4_sf_staging_bounds(sf_staging_module, tile_rows, paged, produce_v, tail):
    """Unused lanes must preserve the sentinel and still zero-fill V padding."""
    kv_len = tile_rows - tail
    sf_bytes = tile_rows * 8
    pages = (tile_rows + 31) // 32
    indices = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device="cuda")
    if paged:
        src = torch.randint(0, 128, (pages, 32, 2, 8), dtype=torch.uint8, device="cuda")
        expected_data = src[indices.long(), :, 1].reshape(-1, 8)[:kv_len]
    else:
        src = torch.randint(
            0, 128, (tile_rows + 3, 2, 8), dtype=torch.uint8, device="cuda"
        )
        expected_data = src[3 : 3 + kv_len, 1]
    output = torch.empty(sf_bytes + 512, dtype=torch.uint8, device="cuda")
    expected = torch.full_like(output, 0xA5)
    expected[: kv_len * 8] = expected_data.flatten()
    if produce_v:
        expected[kv_len * 8 : sf_bytes] = 0
    sf_staging_module.run(src, indices, output, tile_rows, kv_len, paged, produce_v)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
