"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared Python constants for PCIe IPC collectives; importing them needs no CUDA.
"""

# All three collectives address complete uint4 packs across the FFI boundary.
PACK_BYTES = 16

# Hard AG/RS launch limits, mirrored by kMaxBlocks/kMaxThreads in
# include/flashinfer/comm/pcie_ipc_common.cuh. A workspace may choose fewer
# blocks; policy seeds and tuning candidates must fit that capacity.
AG_RS_MAX_BLOCKS = 64
AG_RS_MAX_THREADS = 512

# Existing AllReduce workspace/policy default. This is a configurable capacity,
# not the AG/RS kernel limit or a universal CUDA grid limit.
AR_MAX_BLOCKS = 128
