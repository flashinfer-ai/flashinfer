# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cache device facts and derive Frost launch/storage constants at preparation.

Resource accounting follows cuDNN Frontend's Frost renderer (NVIDIA, MIT).
No cuDNN import, tensor readback or device query occurs on the execution path.
"""

from __future__ import annotations

import ast
import functools
from contextlib import suppress
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceFacts:
    arch: str
    smem_bytes: int
    l2_bytes: int
    clusters: tuple[int, ...]


def _checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"Frost device query failed: {error}")
    return values[0] if values else None


@functools.lru_cache(maxsize=1024)
def device_facts(index: int) -> DeviceFacts:
    from cuda.bindings import driver as drv

    with torch.cuda.device(index):
        torch.cuda.init()
        props = torch.cuda.get_device_properties(index)
        device = _checked(drv.cuDeviceGet(index))
        optin = props.shared_memory_per_block_optin
        oversized = 0
        if _checked(drv.cuDriverGetVersion()) >= 13040:
            attribute = getattr(
                drv.CUdevice_attribute,
                "CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK",
                150,
            )
            with suppress(AttributeError):
                oversized = int(_checked(drv.cuDeviceGetAttribute(attribute, device)))
        # A full-opt-in-SMEM proxy measures one CTA/SM, including GPC packing.
        ptx = b""".version 7.8
.target sm_90
.address_size 64
.extern .shared .align 16 .b8 frost_probe_smem[];
.visible .entry frost_occupancy_probe() {
    .reg .b32 %r<3>;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, frost_probe_smem;
    add.s32 %r2, %r2, %r1;
    st.shared.u8 [%r2], %r1;
    ret;
}
"""
        module = None
        previous = _checked(drv.cuCtxGetCurrent())
        primary = _checked(drv.cuDevicePrimaryCtxRetain(device))
        _checked(drv.cuCtxSetCurrent(primary))
        try:
            module = _checked(drv.cuModuleLoadData(ptx))
            function = _checked(
                drv.cuModuleGetFunction(module, b"frost_occupancy_probe")
            )
            _checked(
                drv.cuFuncSetAttribute(
                    function,
                    drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    optin,
                )
            )
            _checked(
                drv.cuFuncSetAttribute(
                    function,
                    drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                    1,
                )
            )
            clusters = []
            for size in range(1, 17):
                launch = drv.CUlaunchConfig()
                launch.gridDimX, launch.gridDimY, launch.gridDimZ = size, 1, 1
                launch.blockDimX, launch.blockDimY, launch.blockDimZ = 256, 1, 1
                launch.sharedMemBytes = optin
                attribute = drv.CUlaunchAttribute()
                attribute.id = (
                    drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
                (
                    attribute.value.clusterDim.x,
                    attribute.value.clusterDim.y,
                    attribute.value.clusterDim.z,
                ) = size, 1, 1
                launch.attrs, launch.numAttrs = [attribute], 1
                result = drv.cuOccupancyMaxActiveClusters(function, launch)
                if result[0] == drv.CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE:
                    clusters.append(0)
                else:
                    clusters.append(int(_checked(result)))
        finally:
            if module is not None:
                _checked(drv.cuModuleUnload(module))
            _checked(drv.cuCtxSetCurrent(previous))
            _checked(drv.cuDevicePrimaryCtxRelease(device))
        return DeviceFacts(
            f"sm_{props.major}{props.minor}a",
            max(optin, oversized),
            props.L2_cache_size,
            tuple(clusters),
        )


def _sm120_block_scale_stages(value, smem_bytes, swap_ab):
    """AB ring depth of cuDNN Frost's SM120 block-scale MoE renderer.

    A stage holds every operand's packed K-tile and scale-factor box (+16 B).
    The transposed-STG epilogue staging (4 B x 528 per compute warp) and the
    template's 2 KB reserve come off the budget first; GATHER adds 32 B of
    linear-scale scratch per routed token row (the token is B after swap-AB).
    Multi-GEMM (gated) kernels then drop stages until the aligned buffer
    layout fits.
    """
    m, n, _ = value("cta_tile_mnk")
    na, nb = value("num_a_operands"), value("num_b_operands")
    sfa, sfb = value("sfa_smem_bytes"), value("sfb_smem_bytes")
    gemms = value("num_gemms", 1)
    if gemms > 1:
        warps = 8
        per_stage = (
            na * (value("sA_packed_elems") + (0 if value("fake_dequant_a") else sfa))
            + nb * (value("sB_packed_elems") + (0 if value("fake_dequant_b") else sfb))
            + 16
        )
    else:
        warps = (m // (16 * value("mma_size_m"))) * (n // (16 * value("mma_size_n")))
        per_stage = (
            value("sA_packed_elems")
            + value("sB_packed_elems")
            + value("num_sfa_operands") * sfa
            + value("num_sfb_operands") * sfb
            + 16
        )
    staging = 4 * 528 * warps
    rows, tokens = (n, nb) if swap_ab else (m, na)
    gather = value("moe_gather", False)
    if gather and gemms > 1:
        if not value("fake_dequant_b" if swap_ab else "fake_dequant_a"):
            staging += rows * 32 * tokens
    elif gather:
        scaled = value("num_sfb_operands" if swap_ab else "num_sfa_operands")
        staging += rows * 32 * scaled
    available = smem_bytes - 2048 - staging
    stages = min(available // per_stage, 16) if available >= per_stage else 0
    if gemms > 1:
        kb = max(value("a_packed_per_row"), value("b_packed_per_row"))

        def up(offset, align):
            return -(-offset // align) * align

        def footprint(count):
            offset = 1024
            for operands, tile in ((na, m * kb), (nb, n * kb)):
                for _ in range(operands):
                    offset = up(offset, 1024) + count * tile
            for operands, tile in ((na, sfa), (nb, sfb)):
                for _ in range(operands):
                    offset = up(offset, 512) + count * tile
            offset = up(offset, 1024) + 4 * 528 * warps
            if gather:
                for _ in range(tokens):
                    offset = up(offset, 128) + rows * 32
                for _ in range(tokens):
                    offset = up(offset, 8) + 8
            return offset

        while stages > 0 and footprint(stages) > smem_bytes:
            stages -= 1
    return stages


def specialize_parameters(parameters, facts: DeviceFacts):
    """Resolve device-derived constants before source generation and JIT."""
    from copy import deepcopy

    result = deepcopy(parameters)
    constants = dict(result["constants"])

    def value(name, default=None):
        return ast.literal_eval(constants[name]) if name in constants else default

    cm, cn, _ = value("cluster_shape_mnk")
    grid = facts.clusters[cm * cn - 1]
    if not grid:
        raise NotImplementedError("Frost cluster geometry does not fit this device")
    m, n, k = value("cta_tile_mnk")
    na, nb = value("num_a_operands"), value("num_b_operands")
    reserve = 4096
    tma_bytes = 0
    if value("n_tma_outputs", 0):
        element_bytes = 2 if constants["epi_store_dtype"] == "cutlass.BFloat16" else 1
        tma_bytes = (
            2
            * value("epi_stage_rows")
            * value("epi_row_elems")
            * element_bytes
            * value("epi_slot_widen", 1)
            + 16
        )
    if facts.arch == "sm_120a" and "sfa_smem_bytes" in constants:
        stages = _sm120_block_scale_stages(value, facts.smem_bytes, result["swap_ab"])
    elif facts.arch == "sm_120a":
        reserve = 2048
        per_stage = (na * m + nb * n) * k * 2 + 16
        if value("num_gemms") > 1:
            reserve += 4 * 528 * 8  # Eight compute warps' transposed-STG staging.
        stages = min(16, (facts.smem_bytes - reserve) // per_stage)
    elif "sfa_smem_bytes" in constants:
        rings = (
            [value("sfa_smem_bytes")] * value("num_sfa_operands")
            + [value("sfb_smem_bytes")] * value("num_sfb_operands")
            + [value("sA_packed_elems")] * na
            + [value("sB_packed_elems")] * nb
        )
        stages = min(16, (facts.smem_bytes - reserve - tma_bytes) // sum(rings))
        while stages > 0:
            size = 0
            root = reserve + (tma_bytes if not result["swap_ab"] else 0)
            safe = True
            for ring in rings:
                size = (size + 1023) // 1024 * 1024 + ring * stages
                root = (root + 1023) // 1024 * 1024
                safe &= root < 262144  # Tcgen05SmemDesc.build's 14-bit root.
                root += ring * stages
            if safe and size + reserve + tma_bytes <= facts.smem_bytes:
                break
            stages -= 1
    else:
        per_stage = (na * m + nb * n) * k * 2
        stages = min(16, (facts.smem_bytes - reserve - tma_bytes) // per_stage)
    if stages < 1:
        raise NotImplementedError("Frost tile does not fit the device SMEM budget")
    derived = {
        "grid_num_clusters": str(grid),
        "swizzle_l2_budget_bytes": str(facts.l2_bytes // 3),
        "ab_stages": str(stages),
    }
    result["constants"] = [
        [name, derived.get(name, expression)]
        for name, expression in result["constants"]
    ]
    result.pop("device_dependent", None)
    slots = (
        int(value("moe_gather", False)) + int(value("moe_scatter", False))
        if facts.arch == "sm_120a"
        else (nb if result["swap_ab"] else na)
        * (2 if "sfa_smem_bytes" in constants else 1)
        + value("n_tma_outputs", 0)
    )
    workspace_bytes = (grid * cm * cn * slots + 1) * 128
    return result, workspace_bytes
