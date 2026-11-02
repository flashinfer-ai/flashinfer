/*
 * Copyright (c) 2026 by FlashInfer team.
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
#pragma once

#include <cuda_runtime.h>

// The three original generated payloads came from Cake checkpoint
// 924403cd62ea6dc160ffcea63c944d1e828d393a and MR474 cleanup
// d77b50ee1e56fb3701a512f1dd2765e6067b1be2. Their pre-integration generated
// sm_100a and sm_103a output was byte-identical:
//   bootstrap: sha256:46049370dbd905ff7234d414745d197f883157c8edc01d83230562b4dff5f862
//   rowwise:   sha256:e47f82923112e143173f32dbb145986f8c9a1503fd2bb71f22f5c065769162e3
//   warp:      sha256:8bc9df2b57d4e6021d0add03110a11f0cbb3bab1218e94d00d92f4bbf597ee9e
// Integration adds standard integer types, sm_103a-only PDL device control,
// and a CUDA <12.9 2x128-bit fallback for bootstrap's 256-bit inline PTX.

extern "C" {

__global__ void kernel_flashinfer_blackwell_softmax_bootstrap_seed(
    float* logits, float* parameter, float* output, float* partial_max, float* partial_sum,
    int rows, int vocab_size, int splits, int parameter_kind, float scalar_temperature);

__global__ void kernel_flashinfer_blackwell_softmax_followup_rowwise(
    float* logits, float* parameter, float* output, int rows, int vocab_size, int parameter_kind,
    float scalar_temperature);

__global__ void kernel_flashinfer_blackwell_softmax_followup_warp(float* logits, float* parameter,
                                                                  float* output, int rows,
                                                                  int vocab_size,
                                                                  int parameter_kind,
                                                                  float scalar_temperature);

// Row-caching cluster family: a row is split across a cluster of C CTAs (256
// threads) that each keep T tiles of 8 fp32 per thread in registers. Every
// warp's (max, sum) pair goes to every CTA of the cluster through st.async
// pushes counted on an mbarrier, and each CTA writes its normalized chunk.
// Clusters of 16 CTAs need cudaFuncAttributeNonPortableClusterSizeAllowed.
// Each row-caching cluster kernel NAME (including the shared-tile kernels
// below) has a twin NAME_pdl for PDL launches, which waits on the
// prerequisite grid at entry and triggers its dependents at exit; both take
// the plain 0/1/2 parameter_kind.
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c1_t4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c1_t4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c2_t4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c2_t4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c4_t4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c4_t4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t8(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t8_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t8(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t8_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);

// Shared-tile row-caching cluster kernels of 8 and 16 CTAs: each CTA keeps 4
// tiles of its chunk in registers and up to 4 more in shared memory, copied
// in at entry by 1D bulk copies, so 8-tile chunks run four CTAs per SM.
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t4_s4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c8_t4_s4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t4_s4(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c16_t4_s4_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);

// Wide row-caching cluster kernels of 10 and 12 CTAs: each CTA keeps up to 13
// (10 CTAs) or 11 (12 CTAs) tiles of its chunk in registers at one resident
// CTA per SM, so that a batch with one row more than 16-CTA clusters fit one
// CTA per SM keeps every CTA on its own SM.
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c10_t8(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c10_t8_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c10_t10(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c10_t10_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c10_t13(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c10_t13_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c12_t8(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c12_t8_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c12_t10(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c12_t10_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c12_t11(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_c12_t11_pdl(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);

// Persistent row-caching cluster kernels of the 8-tile variants: a grid of at
// most one cluster per row, each cluster walking rows k, k + n, ... with the
// next row's chunk copied into shared memory by 1D bulk copies. Dynamic shared
// memory exceeds 48 KiB and needs cudaFuncAttributeMaxDynamicSharedMemorySize.
// They have no _pdl twins: the sm_103a PDL flag rides in bit 2 of
// parameter_kind, as for the warp and rowwise kernels.
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_persistent_c8_t8(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);
__global__ void kernel_flashinfer_blackwell_softmax_cached_cluster_persistent_c16_t8(
    float* x, float* parameter, float* output, int rows, int vocab_size, int chunk_elements,
    int parameter_kind, float scalar_temperature);

}  // extern "C"
