// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// This translation unit is intentionally compiled against the NVSHMEM
// headers installed next to the device bitcode used by CuTeDSL.  Access to
// nvshmemi_device_state_d is an internal-ABI compatibility shim, not a public
// NVSHMEM API.  The static assertions turn an ABI change into a build failure
// instead of an incorrect device-state write.

#include "device_host/nvshmem_types.h"
#include "non_abi/nvshmem_version.h"

// Match the installed bitcode library's constant-memory declaration without
// pulling cuda_runtime.h into the standalone Clang-to-NVPTX compilation.
extern __attribute__((address_space(4))) nvshmemi_device_host_state_t
    nvshmemi_device_state_d;

static_assert(NVSHMEM_VENDOR_MAJOR_VERSION == 3);
static_assert(NVSHMEM_VENDOR_MINOR_VERSION == 7);
static_assert(NVSHMEM_VENDOR_PATCH_VERSION == 0);
static_assert(sizeof(nvshmemi_device_host_state_v1) == 816);
static_assert(offsetof(nvshmemi_device_host_state_v1, peer_heap_base_p2p) == 56);

extern "C" __attribute__((noinline))
void megamoe_nvshmem_disable_p2p_peer(int peer) {
  nvshmemi_device_state_d.peer_heap_base_p2p[peer] = nullptr;
  asm volatile("membar.sys;" ::: "memory");
}
