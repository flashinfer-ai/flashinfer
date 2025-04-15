/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_PERSISTENT_TEMPLATE_CUH
#define FLASHINFER_ATTENTION_PERSISTENT_TEMPLATE_CUH

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {

// Helper metafunction to find maximum threads among multiple BlockPersistentRunners
template <typename... Runners>
struct max_threads;

template <typename Runner>
struct max_threads<Runner> {
  static constexpr size_t value = Runner::KTraits::NUM_THREADS;
};

template <typename Runner1, typename Runner2, typename... RestRunners>
struct max_threads<Runner1, Runner2, RestRunners...> {
  static constexpr size_t value = Runner1::KTraits::NUM_THREADS > Runner2::KTraits::NUM_THREADS
                                      ? max_threads<Runner1, RestRunners...>::value
                                      : max_threads<Runner2, RestRunners...>::value;
};

// Single runner version
template <class BlockPersistentRunner>
__global__
__launch_bounds__(BlockPersistentRunner::KTraits::NUM_THREADS) void PersistentKernelTemplate(
    const __grid_constant__ typename BlockPersistentRunner::Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage =
      reinterpret_cast<typename BlockPersistentRunner::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner::Run(params, &smem_storage);
}

// Two runners version
template <class BlockPersistentRunner1, class BlockPersistentRunner2>
__global__ __launch_bounds__(
    max_threads<BlockPersistentRunner1, BlockPersistentRunner2>::
        value) void PersistentKernelTemplate(const __grid_constant__
                                             typename BlockPersistentRunner1::Params params_1,
                                             const __grid_constant__
                                             typename BlockPersistentRunner2::Params params_2) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage_1 =
      reinterpret_cast<typename BlockPersistentRunner1::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner1::Run(params_1, &smem_storage_1);
  __syncthreads();
  auto& smem_storage_2 =
      reinterpret_cast<typename BlockPersistentRunner2::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner2::Run(params_2, &smem_storage_2);
}

// Three runners version
template <class BlockPersistentRunner1, class BlockPersistentRunner2, class BlockPersistentRunner3>
__global__ __launch_bounds__(
    max_threads<BlockPersistentRunner1, BlockPersistentRunner2, BlockPersistentRunner3>::
        value) void PersistentKernelTemplate(const __grid_constant__
                                             typename BlockPersistentRunner1::Params params_1,
                                             const __grid_constant__
                                             typename BlockPersistentRunner2::Params params_2,
                                             const __grid_constant__
                                             typename BlockPersistentRunner3::Params params_3) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage_1 =
      reinterpret_cast<typename BlockPersistentRunner1::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner1::Run(params_1, &smem_storage_1);
  __syncthreads();
  auto& smem_storage_2 =
      reinterpret_cast<typename BlockPersistentRunner2::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner2::Run(params_2, &smem_storage_2);
  __syncthreads();
  auto& smem_storage_3 =
      reinterpret_cast<typename BlockPersistentRunner3::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner3::Run(params_3, &smem_storage_3);
}

// Four runners version
template <class BlockPersistentRunner1, class BlockPersistentRunner2, class BlockPersistentRunner3,
          class BlockPersistentRunner4>
__global__ __launch_bounds__(
    max_threads<BlockPersistentRunner1, BlockPersistentRunner2, BlockPersistentRunner3,
                BlockPersistentRunner4>::
        value) void PersistentKernelTemplate(const __grid_constant__
                                             typename BlockPersistentRunner1::Params params_1,
                                             const __grid_constant__
                                             typename BlockPersistentRunner2::Params params_2,
                                             const __grid_constant__
                                             typename BlockPersistentRunner3::Params params_3,
                                             const __grid_constant__
                                             typename BlockPersistentRunner4::Params params_4) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage_1 =
      reinterpret_cast<typename BlockPersistentRunner1::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner1::Run(params_1, &smem_storage_1);
  __syncthreads();
  auto& smem_storage_2 =
      reinterpret_cast<typename BlockPersistentRunner2::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner2::Run(params_2, &smem_storage_2);
  __syncthreads();
  auto& smem_storage_3 =
      reinterpret_cast<typename BlockPersistentRunner3::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner3::Run(params_3, &smem_storage_3);
  __syncthreads();
  auto& smem_storage_4 =
      reinterpret_cast<typename BlockPersistentRunner4::KTraits::SharedStorage&>(smem);
  BlockPersistentRunner4::Run(params_4, &smem_storage_4);
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PERSISTENT_TEMPLATE_CUH
