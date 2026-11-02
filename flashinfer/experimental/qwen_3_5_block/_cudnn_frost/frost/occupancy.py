# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""How many thread-block clusters the device can actually run at once."""

from __future__ import annotations

from .device import compute_capability, device_context, device_name, is_available, multiprocessor_count, resolve_device, shared_memory_per_block_optin

MAX_CLUSTER_SIZE = 16

MIN_CLUSTER_CAPABILITY = (9, 0)

# {device index: table}. Per device — the count depends on the GPU's SM/GPC
# geometry, so a process driving two different models needs two tables.
_OCCUPANCY_MAPS: dict[int, list[int]] = {}


_PROBE_PTX = b"""
.version 7.8
.target sm_90
.address_size 64

.extern .shared .align 16 .b8 frost_probe_smem[];

.visible .entry frost_occupancy_probe(
    .param .u64 frost_occupancy_probe_param_0
)
{
    .reg .b32 %r<4>;

    mov.u32      %r1, %tid.x;
    mov.u32      %r2, frost_probe_smem;
    add.s32      %r3, %r2, %r1;
    st.shared.u8 [%r3], %r1;
    ret;
}
"""


def _build_occupancy_map(device: int) -> list[int]:
    """Query sizes 1..:data:`MAX_CLUSTER_SIZE` against a proxy kernel pinned to
    1 CTA/SM (dynamic SMEM = the device's full opt-in budget)."""
    import cuda.bindings.driver as drv

    def _ck(err, *rest):
        if int(err) != 0:
            raise RuntimeError(f"cudnn.frost.occupancy: {err}")
        return rest[0] if len(rest) == 1 else None

    if not is_available():
        raise RuntimeError("cudnn.frost.occupancy: no CUDA device visible")
    if compute_capability(device) < MIN_CLUSTER_CAPABILITY:
        # No clusters below sm_90: a size-1 "cluster" is a CTA, one per SM under
        # the same full-opt-in-SMEM pinning the probe uses, and nothing wider is
        # launchable — the zeros are what max_active_clusters rejects on.
        return [multiprocessor_count(device)] + [0] * (MAX_CLUSTER_SIZE - 1)
    smem_bytes = shared_memory_per_block_optin(device)

    # The driver calls below answer for the CURRENT context, so the query has to
    # run on `device` — importing a FROST engine deliberately creates none.
    with device_context(device):
        mod = _ck(*drv.cuModuleLoadData(_PROBE_PTX))
        try:
            fn = _ck(*drv.cuModuleGetFunction(mod, b"frost_occupancy_probe"))
            _ck(*drv.cuFuncSetAttribute(fn, drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes), None)
            _ck(*drv.cuFuncSetAttribute(fn, drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1), None)
            table = []
            for size in range(1, MAX_CLUSTER_SIZE + 1):
                cfg = drv.CUlaunchConfig()
                cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = size, 1, 1
                cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 256, 1, 1
                cfg.sharedMemBytes = smem_bytes
                attr = drv.CUlaunchAttribute()
                attr.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = size, 1, 1
                cfg.attrs, cfg.numAttrs = [attr], 1
                table.append(int(_ck(*drv.cuOccupancyMaxActiveClusters(fn, cfg))))
            return table
        finally:
            drv.cuModuleUnload(mod)


def occupancy_map(device=None) -> list[int]:
    """The whole table for ``device`` (default: the current one): entry ``i`` is
    the max co-resident cluster count for ``cluster_size == i + 1``. Built once
    per device per process."""
    dev = resolve_device(device)
    if dev not in _OCCUPANCY_MAPS:
        _OCCUPANCY_MAPS[dev] = _build_occupancy_map(dev)
    return _OCCUPANCY_MAPS[dev]


def max_active_clusters(cluster_size: int, device=None) -> int:
    """Max clusters of ``cluster_size`` CTAs ``device`` can run at once."""
    if not 1 <= cluster_size <= MAX_CLUSTER_SIZE:
        raise ValueError(f"cudnn.frost: cluster_size={cluster_size} is outside the supported " f"range 1..{MAX_CLUSTER_SIZE} (architecture CGA limit)")
    dev = resolve_device(device)
    count = occupancy_map(dev)[cluster_size - 1]
    if count == 0:
        raise NotImplementedError(
            f"cudnn.frost: a {cluster_size}-CTA cluster does not fit on cuda:{dev} " f"({device_name(dev)}) — no grid of that cluster size is launchable"
        )
    return count
