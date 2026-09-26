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

Generated-program backend: Kimi-K3 Stable LatentMoE front / tail projections
(SM100 / SM103; flashinfer-ai/flashinfer#4568, tracker #4254).

Per rank and layer, around the routed experts of ``nvidia/Kimi-K3-NVFP4``::

    front:  logits     = FP32(x @ gate_weight.T)                     [T, 896]  router logits
            latent     = BF16(x @ down_weight.T)                     [T, 3584] routed-expert input
            shared_act = SiTU(x @ shared_gate.T, x @ shared_up.T)    [T, 6144 / TP]
    tail:   y   = KimiRMSNorm(sum of the P routed partials)          [T, 3584] (caller-owned workspace)
            out = BF16(y[:, cols] @ up_weight[:, cols].T + shared_act @ shared_down.T)   [T, 7168]
                  (cols = this rank's 3584 / TP latent slice; TP > 1 all-reduces ``out`` outside)

The host route mirrors the Cake production launcher exactly: for ``T <= 128``
tokens one weight-streaming swapped-AB tcgen05 kernel per stage (the tail's
KimiRMSNorm is fused into that launch), for ``T > 128`` a persistent 2-CTA
tcgen05 GEMM per stage (the tail additionally runs a one-pass RMSNorm kernel
first and launches its GEMM programmatic-dependent).  The decode planner
(N padding, ring depth, cluster pairs, resident B operand, staged rows, issue
gate) and the prefill planner (persistent pair tiles, trailing-wave stream-K,
norm PDL trigger) are re-implemented here byte-for-byte from the Cake modules
``kimi_k3_latent_moe_decode`` / ``kimi_k3_latent_moe_front`` /
``kimi_k3_latent_moe_tail``; every plan names the physical generated kernel
through a logical key resolved in ``cake_jit.KERNELS``.  Weights are read in
the model layout (``nn.Linear`` ``[out, in]`` BF16); nothing is copied or
packed and nothing is allocated at launch, so a prepared runner is CUDA Graph
safe.  See ``README.md`` in this package.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    KERNELS,
    MODULES,
    kernel_module_name,
    load_cake_kimi_k3_latent_moe_module,
)

HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
SHARED_INTERMEDIATE = 6144
RMS_EPS = 1.0e-5
SUPPORTED_TP = (1, 8)
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
#: SM count the plan rules were frozen with (B200 and B300 both expose 148 SMs);
#: ``prepare_*`` refuses a device with another count.
SM_COUNT = 148
#: Largest token count served by the decode (weight-streaming) kernels.
DECODE_MAX_T = 128
#: Token counts of the validated route set (both stages, TP 1 and 8).
ROW_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

# ---------------------------------------------------------------------------
# Decode planner (port of loom/examples/weave/kimi_k3_latent_moe_decode.py)
# ---------------------------------------------------------------------------

BLOCK_ROWS = 128
HALF_ROWS = 64
CHUNK_K = 64
EPI_WARPS = 4
MAX_DYN_SMEM = 232448
SMEM_BARRIER_RESERVE = 1024
FLAG_BYTES = 1024
SUPPORTED_N_PAD = (8, 16, 32, 64, 128)
MAX_STAGES = 12
GATE_MAX_DOWN_UNITS = 16
GATE_PRO = 3
SMEM_B1_MAX_TOKENS = 2 * EPI_WARPS
SMEM_B1_MIN_STAGES = 6
SMEM_B1_MAX_TOKENS_RS = 4 * EPI_WARPS
SMEM_B1_MIN_STAGES_RS = 5
TL_SLOTS = 16
TL_MAX_GRID = 512
DECODE_THREADS = 192

# Exact keyword set of the decode program's ``run`` entry (argument plan of the
# generated binding); ``grid`` is expanded to ``grid_x/y/z``.
DECODE_KWARGS = (
    "A_R",
    "A_L",
    "A_S",
    "A_2",
    "B_1",
    "B_2",
    "out_r",
    "out_l",
    "out_s",
    "counters",
    "routed",
    "norm_w",
    "y_out",
    "tl",
    "num_tokens",
    "k1_off",
    "num_partials",
    "eps",
    "grid",
)


def n_pad_for(num_tokens: int) -> int:
    for n in SUPPORTED_N_PAD:
        if num_tokens <= n:
            return n
    raise ValueError(
        f"the decode kernels support at most {SUPPORTED_N_PAD[-1]} tokens, got {num_tokens}"
    )


def plan_stages(
    *,
    n_pad: int,
    kdepth: int = 1,
    max_stages: int = MAX_STAGES,
    cluster: int = 1,
    extra_bytes: int = 0,
) -> int:
    stage_bytes = (BLOCK_ROWS + n_pad) * CHUNK_K * 2 * kdepth
    red_alloc = n_pad * BLOCK_ROWS * 4 if cluster == 2 else 1024
    return max(
        1,
        min(
            max_stages,
            (MAX_DYN_SMEM - SMEM_BARRIER_RESERVE - FLAG_BYTES - red_alloc - extra_bytes)
            // stage_bytes,
        ),
    )


def smem_b1_max_tokens(k2_units_per_cta: int) -> int:
    return SMEM_B1_MAX_TOKENS if k2_units_per_cta > GATE_MAX_DOWN_UNITS else EPI_WARPS


def plan_partition(
    *,
    tiles: int,
    k1_chunks: int,
    k2_chunks: int,
    kdepth: int,
    sm_count: int,
    n_pad: int = 8,
) -> dict[str, int]:
    """Aligned cluster pairs when ``2 * tiles`` fits one wave, else one CTA per tile."""
    mpt = (k1_chunks + k2_chunks) // kdepth
    cluster = 2 if (2 * tiles <= sm_count and mpt >= 2 and n_pad % 4 == 0) else 1
    g = 2 * tiles if cluster == 2 else tiles
    if g > sm_count:
        raise ValueError(f"grid {g} exceeds the SM count {sm_count} (co-residency)")
    return {"grid": g, "cluster": cluster, "mpt": mpt, "units": tiles * mpt}


def kdepth_for(*k_chunks: int, requested: Optional[int] = None) -> int:
    for d in (requested,) if requested else (4, 2, 1):
        if d and all(k % d == 0 for k in k_chunks):
            return d
    raise ValueError(f"kdepth {requested} does not divide K segments {k_chunks}")


def choose_config(
    *, tiles: int, n_pad: int, k_chunks: tuple[int, ...], sm_count: int
) -> tuple[int, int, int, int]:
    """``(kdepth, grid, stages, cluster)`` of the production rule (depth 1, ring takes all smem)."""
    kd = kdepth_for(*k_chunks, requested=1)
    part = plan_partition(
        tiles=tiles,
        k1_chunks=k_chunks[0],
        k2_chunks=k_chunks[1] if len(k_chunks) > 1 else 0,
        kdepth=kd,
        sm_count=sm_count,
        n_pad=n_pad,
    )
    return (
        kd,
        part["grid"],
        plan_stages(n_pad=n_pad, kdepth=kd, cluster=part["cluster"]),
        part["cluster"],
    )


def decode_front_plan(
    num_tokens: int, i_local: int, sm_count: int = SM_COUNT
) -> dict[str, Any]:
    """Instance configuration of ``front_decode`` (router + latent + shared SiTU tiles, one launch)."""
    if i_local % HALF_ROWS:
        raise ValueError("I_LOCAL must be a multiple of 64")
    n_pad = n_pad_for(num_tokens)
    r_tiles, l_tiles, s_tiles = (
        NUM_EXPERTS // BLOCK_ROWS,
        LATENT // BLOCK_ROWS,
        i_local // HALF_ROWS,
    )
    tiles = r_tiles + l_tiles + s_tiles
    k_chunks = HIDDEN // CHUNK_K
    kdepth, grid, stages, cluster = choose_config(
        tiles=tiles, n_pad=n_pad, k_chunks=(k_chunks,), sm_count=sm_count
    )
    return dict(
        grid=grid,
        n_pad=n_pad,
        stages=stages,
        r_tiles=r_tiles,
        l_tiles=l_tiles,
        s_tiles=s_tiles,
        i_local=i_local,
        k1_chunks=k_chunks,
        k2_chunks=0,
        out_r_ld=NUM_EXPERTS,
        out_l_ld=LATENT,
        out_s_ld=i_local,
        kdepth=kdepth,
        pdl=False,
        packed=False,
        pack_k1=k_chunks,
        stream_only=False,
        cluster=int(cluster),
        fused=False,
        fused_probe=0,
        issue_gate=False,
        smem_b1=False,
        rows_smem=False,
        timeline=False,
        gate_pro=GATE_PRO,
        tmap_prefetch=True,
        tiles=tiles,
    )


def decode_tail_plan(
    num_tokens: int,
    i_local: int,
    tp: int,
    num_partials: int = 1,
    sm_count: int = SM_COUNT,
) -> dict[str, Any]:
    """Instance configuration of ``tail_decode_fused`` (port of ``_tail_plan`` with the production defaults)."""
    n_pad = n_pad_for(num_tokens)
    tiles = HIDDEN // BLOCK_ROWS
    k_up = LATENT // tp
    if k_up % CHUNK_K or i_local % CHUNK_K:
        raise ValueError("K slices must be multiples of 64")
    k1, k2 = k_up // CHUNK_K, i_local // CHUNK_K
    kdepth, grid, stages, cluster = choose_config(
        tiles=tiles, n_pad=n_pad, k_chunks=(k1, k2), sm_count=sm_count
    )
    k1_macros = k1 // kdepth
    cl_u0 = (k1_macros + 1) // 2 if cluster == 2 else k1_macros
    bn_units = max(cl_u0, k1_macros - cl_u0) if cluster == 2 else k1_macros
    bn_bytes = bn_units * n_pad * CHUNK_K * 2 * kdepth
    k2_units = -(-k2 // (2 if cluster == 2 else 1))
    rows_ok = kdepth == 1 and int(num_partials) == 1 and k2_units <= GATE_MAX_DOWN_UNITS
    rows_bytes = (n_pad * LATENT + LATENT) * 2
    use_bn = kdepth == 1 and num_tokens <= (
        SMEM_B1_MAX_TOKENS_RS if rows_ok else smem_b1_max_tokens(k2_units)
    )
    use_rows = use_bn and rows_ok
    gate_default = k2_units <= GATE_MAX_DOWN_UNITS and not use_rows
    if use_bn:
        bn_stages = plan_stages(
            n_pad=n_pad,
            kdepth=kdepth,
            cluster=cluster,
            extra_bytes=bn_bytes + (rows_bytes if use_rows else 0),
        )
        if bn_stages < SMEM_B1_MIN_STAGES_RS and use_rows:
            use_rows = False
            use_bn = num_tokens <= smem_b1_max_tokens(k2_units)
            gate_default = k2_units <= GATE_MAX_DOWN_UNITS
            bn_stages = plan_stages(
                n_pad=n_pad, kdepth=kdepth, cluster=cluster, extra_bytes=bn_bytes
            )
        if use_bn and bn_stages < SMEM_B1_MIN_STAGES:
            use_bn = False
            use_rows = False
        elif use_bn:
            stages = bn_stages
    return dict(
        grid=grid,
        n_pad=n_pad,
        stages=stages,
        r_tiles=0,
        l_tiles=tiles,
        s_tiles=0,
        i_local=i_local,
        k1_chunks=k1,
        k2_chunks=k2,
        out_r_ld=NUM_EXPERTS,
        out_l_ld=HIDDEN,
        out_s_ld=i_local,
        kdepth=kdepth,
        pdl=False,
        packed=False,
        pack_k1=LATENT // CHUNK_K,
        stream_only=False,
        cluster=int(cluster),
        fused=True,
        fused_probe=0,
        issue_gate=bool(gate_default),
        smem_b1=bool(use_bn),
        rows_smem=bool(use_rows),
        timeline=False,
        gate_pro=GATE_PRO,
        tmap_prefetch=True,
        tiles=tiles,
        k1=k1,
        k2=k2,
    )


def decode_symbol(plan: dict[str, Any]) -> str:
    """Kernel symbol of one decode instance (``build_decode_gemm_ir`` naming rule)."""
    fused = bool(plan["fused"])
    probe = f"_fp{plan['fused_probe']}" if (fused and plan["fused_probe"]) else ""
    gate = f"_g{plan['gate_pro']}" if (fused and plan["issue_gate"]) else ""
    smem_b1 = "_sb" if plan["smem_b1"] else ""
    rows = "_rs" if (plan["smem_b1"] and plan["rows_smem"]) else ""
    return (
        f"kimi_k3_latent_moe_decode_g{plan['grid']}_n{plan['n_pad']}_r{plan['stages']}_d{plan['kdepth']}"
        f"_p{1 if plan['pdl'] else 0}_w{1 if plan['packed'] else 0}"
        f"_t{plan['r_tiles']}_{plan['l_tiles']}_{plan['s_tiles']}_k{plan['k1_chunks']}_{plan['k2_chunks']}"
        f"_o{plan['out_l_ld']}{'_so' if plan['stream_only'] else ''}_c{plan['cluster']}{'_f' if fused else ''}"
        f"{probe}{gate}{smem_b1}{rows}{'_tl' if plan['timeline'] else ''}{'' if plan['tmap_prefetch'] else '_np'}"
    )


def decode_kernel_key(plan: dict[str, Any]) -> str:
    return f"decode:{decode_symbol(plan)}"


# ---------------------------------------------------------------------------
# Prefill planner (ports of kimi_k3_latent_moe_front / kimi_k3_latent_moe_tail)
# ---------------------------------------------------------------------------

BLOCK_M = 128
BLOCK_N = 256
B_HALF_N = 128
CTA_GROUP = 2
BLOCK_K = 64
FRONT_R_TILES = (NUM_EXPERTS + BLOCK_N - 1) // BLOCK_N  # 4
FRONT_L_TILES = LATENT // BLOCK_N  # 14
TAIL_N_TILES = HIDDEN // BLOCK_N  # 28
GROUP_M = 16
NORM_THREADS = 128
NORM_ROWS_PER_CTA = NORM_THREADS // 32
MAX_SEG = 4
SK_MIN_NUM_K = 64
SK_MIN_ITERS = 32
SK_FIXUP_ITERS = 12
SK_MIN_REUSE_ROWS = 8

FRONT_KWARGS = (
    "A",
    "WG",
    "WD",
    "WS",
    "logits",
    "latent",
    "shared_act",
    "M",
    "m_tiles",
    "grid",
)
NORM_KWARGS = ("routed", "norm_weight", "y_out", "M", "num_partials", "eps", "grid")
TAIL_GEMM_KWARGS = (
    "A1",
    "B1",
    "A2",
    "B2",
    "out",
    "ws",
    "counters",
    "M",
    "m_tiles",
    "k0_blocks",
    "num_items",
    "full_items",
    "sk_ipc",
    "sk_max_seg",
    "sk_total",
    "grid",
)


def i_local_for_tp(tp: int) -> int:
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    return SHARED_INTERMEDIATE // tp


def k_up_for_tp(tp: int) -> int:
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    return LATENT // tp


def front_s_tiles(i_local: int) -> int:
    if i_local % B_HALF_N:
        raise ValueError(f"I_LOCAL must be a multiple of {B_HALF_N}, got {i_local}")
    return i_local // B_HALF_N


def front_n_tiles(i_local: int) -> int:
    return FRONT_R_TILES + FRONT_L_TILES + front_s_tiles(i_local)


def m_tiles_for(M: int) -> int:
    tiles = (M + BLOCK_M - 1) // BLOCK_M
    return tiles + (tiles % CTA_GROUP)


def front_grid(m_tiles: int, i_local: int) -> int:
    """One cluster (CTA pair) per output tile pair; CLC hands out the rest in raster order."""
    return (m_tiles // CTA_GROUP) * front_n_tiles(i_local) * CTA_GROUP


def front_kernel_key(i_local: int) -> str:
    return f"front:i{int(i_local)}"


def norm_kernel_key(early_trigger: bool) -> str:
    return f"tail_norm:e{1 if early_trigger else 0}"


def tail_gemm_kernel_key(tp: int) -> str:
    return f"tail_gemm:tp{int(tp)}"


def _sk_max_seg(sk_tiles: int, num_k: int, ipc: int) -> int:
    return max(
        ((t * num_k + num_k - 1) // ipc) - ((t * num_k) // ipc) + 1
        for t in range(sk_tiles)
    )


@functools.lru_cache(maxsize=None)
def split_plan(
    cluster_tiles: int, num_k: int, sm_count: int, reuse_rows: int = 1
) -> dict[str, int]:
    """Persistent plan: whole pair tiles for the full waves, a stream-K trailing wave when it wins."""
    resident = max(1, sm_count // CTA_GROUP)
    full = (cluster_tiles // resident) * resident
    rem = cluster_tiles - full
    waves = (cluster_tiles + resident - 1) // resident
    best_cost = float(waves * num_k)
    plan = {
        "num_items": cluster_tiles,
        "full_items": cluster_tiles,
        "sk_ipc": num_k,
        "sk_max_seg": 1,
        "sk_total": 0,
        "sk_tiles": 0,
    }
    if rem == 0 or num_k < SK_MIN_NUM_K or 1 < reuse_rows < SK_MIN_REUSE_ROWS:
        return plan
    for sk_tiles in (rem, rem + resident):
        if sk_tiles > cluster_tiles:
            continue
        full_i = cluster_tiles - sk_tiles
        total = sk_tiles * num_k
        for g in range(min(resident, total // SK_MIN_ITERS), 0, -1):
            ipc = (total + g - 1) // g
            g2 = (total + ipc - 1) // ipc
            max_seg = _sk_max_seg(sk_tiles, num_k, ipc)
            if max_seg > MAX_SEG:
                continue
            cost = (full_i // resident) * num_k + ipc + SK_FIXUP_ITERS * (max_seg - 1)
            if cost < best_cost:
                best_cost = cost
                plan = {
                    "num_items": full_i + g2,
                    "full_items": full_i,
                    "sk_ipc": ipc,
                    "sk_max_seg": max_seg,
                    "sk_total": total,
                    "sk_tiles": sk_tiles,
                }
    return plan


def norm_early_trigger(gemm_ctas: int, norm_ctas: int, sm_count: int) -> bool:
    """Fire the norm's PDL trigger at kernel start when the GEMM's early launch cannot serialise items."""
    return gemm_ctas > sm_count or gemm_ctas + norm_ctas <= sm_count


def prefill_tail_plan(M: int, tp: int, sm_count: int = SM_COUNT) -> dict[str, Any]:
    """Host plan of the prefill tail chain (norm launch + persistent GEMM) for ``M`` tokens."""
    k_up = k_up_for_tp(tp)
    i_local = i_local_for_tp(tp)
    norm_grid = (M + NORM_ROWS_PER_CTA - 1) // NORM_ROWS_PER_CTA
    m_tiles = m_tiles_for(M)
    cluster_tiles = (m_tiles // CTA_GROUP) * TAIL_N_TILES
    num_k = (k_up + i_local) // BLOCK_K
    sk = split_plan(cluster_tiles, num_k, sm_count, min(GROUP_M, m_tiles) // CTA_GROUP)
    early = norm_early_trigger(sk["num_items"] * CTA_GROUP, norm_grid, sm_count)
    return dict(
        M=M,
        tp=tp,
        k_up=k_up,
        i_local=i_local,
        norm_grid=norm_grid,
        m_tiles=m_tiles,
        cluster_tiles=cluster_tiles,
        num_k=num_k,
        gemm_grid=sk["num_items"] * CTA_GROUP,
        early_trigger=bool(early),
        **sk,
    )


# ---------------------------------------------------------------------------
# Route resolution (logical kernel keys of every stage)
# ---------------------------------------------------------------------------


def route_kernel_keys(
    stage: str, tp: int, num_tokens: int, sm_count: int = SM_COUNT
) -> tuple[str, ...]:
    """Logical kernel keys launched by the production route of ``(stage, tp, num_tokens)``."""
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    i_local = i_local_for_tp(tp)
    if stage == "front":
        if num_tokens <= DECODE_MAX_T:
            return (
                decode_kernel_key(decode_front_plan(num_tokens, i_local, sm_count)),
            )
        return (front_kernel_key(i_local),)
    if stage == "tail":
        if num_tokens <= DECODE_MAX_T:
            return (
                decode_kernel_key(
                    decode_tail_plan(num_tokens, i_local, tp, 1, sm_count)
                ),
            )
        plan = prefill_tail_plan(num_tokens, tp, sm_count)
        return (norm_kernel_key(plan["early_trigger"]), tail_gemm_kernel_key(tp))
    raise ValueError(f"stage must be 'front' or 'tail', got {stage!r}")


def required_kernel_keys(sm_count: int = SM_COUNT) -> tuple[str, ...]:
    """Every logical kernel the validated route set can select (both stages, TP 1 / 8, all row tokens)."""
    keys: list[str] = []
    for stage in ("front", "tail"):
        for tp in SUPPORTED_TP:
            for tokens in ROW_TOKENS:
                for key in route_kernel_keys(stage, tp, tokens, sm_count):
                    if key not in keys:
                        keys.append(key)
    return tuple(keys)


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 LatentMoE front/tail programs require compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def _check_sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = int(torch.cuda.get_device_properties(index).multi_processor_count)
    if count != SM_COUNT:
        raise RuntimeError(
            f"the generated programs were planned for {SM_COUNT} SMs; device {index} has {count}"
        )
    return index


def generated_program_available(
    device: torch.device,
    stage: Optional[str] = None,
    tp: Optional[int] = None,
    num_tokens: Optional[int] = None,
) -> bool:
    """True when this checkout registers the programs for ``device`` (optionally: one exact route)."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None or not MODULES:
        return False
    table = KERNELS.get(arch, {})
    if stage is None and tp is None and num_tokens is None:
        return set(required_kernel_keys()) <= set(table)
    if stage is None or tp is None or num_tokens is None:
        raise ValueError("pass stage, tp and num_tokens together or none of them")
    try:
        keys = route_kernel_keys(stage, int(tp), int(num_tokens))
    except ValueError:
        return False
    return all(key in table for key in keys)


# ---------------------------------------------------------------------------
# Per-device scratch (the decode kernels' fixed ABI buffers)
# ---------------------------------------------------------------------------

_SCRATCH: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_TAIL_WS: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fp32 dummy, u32 arrival counters, u64 timeline buffer)`` per device.

    The counters are zeroed once: the fused-norm protocol wraps them back to 0
    on every launch (``atom.inc`` with limit GRID-1), so eager launches and
    graph replays share one zero-initialised counter.  The timeline buffer is
    written only by diagnostic instances (never registered here).
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    entry = _SCRATCH.get(index)
    if entry is None:
        dev = torch.device("cuda", index)
        entry = (
            torch.empty(1024, dtype=torch.float32, device=dev),
            torch.zeros(4, dtype=torch.uint32, device=dev),
            torch.zeros(TL_MAX_GRID * TL_SLOTS, dtype=torch.uint64, device=dev),
        )
        _SCRATCH[index] = entry
    return entry


def _tail_workspace(
    device: torch.device, sk_tiles: int, max_seg: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream-K fp32 partial workspace + self-resetting arrival counters (per device and plan class)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, int(sk_tiles), int(max_seg))
    entry = _TAIL_WS.get(key)
    if entry is None:
        dev = torch.device("cuda", index)
        n = max(1, sk_tiles * max_seg * CTA_GROUP)
        entry = (
            torch.empty(n * BLOCK_M * BLOCK_N, dtype=torch.float32, device=dev),
            torch.zeros(max(2, sk_tiles * CTA_GROUP), dtype=torch.int32, device=dev),
        )
        _TAIL_WS[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def _bind(module_name: str, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` and load its entry."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_latent_moe_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


@dataclass(frozen=True)
class _Launch:
    stage: str
    key: str
    module: str
    kwargs: dict[str, Any] = field(repr=False)
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)


@dataclass(frozen=True)
class KimiK3LatentMoeRunner:
    """The prepared launch sequence of one front or tail call.

    ``launch()`` submits the bound programs in order on the current torch
    stream with no CUDA allocation and no host synchronisation; the kernels
    read every operand on device at launch, so the runner (or a CUDA Graph
    capturing it) replays for new values written into the same buffers.
    Prepare a new runner when a shape or a tensor binding changes.
    """

    stage: str
    tp: int
    rank: int
    num_tokens: int
    arch: str
    route: str  # "decode" or "prefill"
    plan: dict[str, Any] = field(repr=False)
    launches: tuple[_Launch, ...] = field(repr=False)
    outputs: tuple[torch.Tensor, ...] = field(repr=False)

    @property
    def kernel_keys(self) -> tuple[str, ...]:
        return tuple(launch.key for launch in self.launches)

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(launch.module for launch in self.launches)

    @property
    def launch_count(self) -> int:
        return len(self.launches)

    def launch(self) -> tuple[torch.Tensor, ...]:
        with tvm_ffi.use_torch_stream():
            for launch in self.launches:
                launch()
        return self.outputs

    __call__ = launch


def _check(
    t: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}, got {tuple(t.shape)} {t.dtype}"
        )


def _same_device(tensors: dict[str, torch.Tensor]) -> torch.device:
    devices = {t.device for t in tensors.values()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError(
            f"every operand must live on one CUDA device, got {sorted(map(str, devices))}"
        )
    return next(iter(devices))


# ---------------------------------------------------------------------------
# Front
# ---------------------------------------------------------------------------


def prepare_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> KimiK3LatentMoeRunner:
    """Validate the front operands, plan the route and bind its launch(es).

    ``shared_gate_up_weight`` is the rank-local ``[2 * I_local, 7168]`` concatenation of the
    shared experts' gate rows followed by their up rows (``I_local = 6144 / TP``).  The JIT
    module(s) are built and loaded here; prepare outside CUDA Graph capture.
    """
    T = int(x.shape[0]) if x.dim() == 2 else -1
    _check(x, (T, HIDDEN), "x")
    i_local = (
        int(shared_gate_up_weight.shape[0]) // 2
        if shared_gate_up_weight.dim() == 2
        else 0
    )
    if i_local not in (SHARED_INTERMEDIATE // tp for tp in SUPPORTED_TP):
        raise ValueError(
            f"shared_gate_up_weight must have 2 * (6144 / TP) rows for TP in {SUPPORTED_TP}, "
            f"got {tuple(shared_gate_up_weight.shape)}"
        )
    tp = SHARED_INTERMEDIATE // i_local
    _check(gate_weight, (NUM_EXPERTS, HIDDEN), "gate_weight")
    _check(down_weight, (LATENT, HIDDEN), "down_weight")
    _check(shared_gate_up_weight, (2 * i_local, HIDDEN), "shared_gate_up_weight")
    _check(logits, (T, NUM_EXPERTS), "logits", torch.float32)
    _check(latent, (T, LATENT), "latent")
    _check(shared_act, (T, i_local), "shared_act")
    device = _same_device(
        dict(
            x=x,
            gate_weight=gate_weight,
            down_weight=down_weight,
            shared_gate_up_weight=shared_gate_up_weight,
            logits=logits,
            latent=latent,
            shared_act=shared_act,
        )
    )
    arch = _device_arch(device)
    index = _check_sm_count(device)
    launches: tuple[_Launch, ...]
    plan: dict[str, Any]
    with torch.cuda.device(index):
        if T <= DECODE_MAX_T:
            plan = decode_front_plan(T, i_local)
            key = decode_kernel_key(plan)
            f32_dummy, counters, tl = _scratch(device)
            kwargs = dict(
                A_R=gate_weight,
                A_L=down_weight,
                A_S=shared_gate_up_weight,
                A_2=down_weight,
                B_1=x,
                B_2=x,
                out_r=logits,
                out_l=latent,
                out_s=shared_act,
                counters=counters,
                routed=latent,
                norm_w=latent,
                y_out=latent,
                tl=tl,
                num_tokens=T,
                k1_off=0,
                num_partials=0,
                eps=0.0,
                grid=(int(plan["grid"]), 1, 1),
            )
            assert tuple(kwargs) == DECODE_KWARGS
            module = kernel_module_name(arch, key)
            entry, arguments = _bind(module, kwargs)
            launches = (_Launch("front_decode", key, module, kwargs, entry, arguments),)
            route = "decode"
        else:
            m_tiles = m_tiles_for(T)
            plan = dict(
                M=T,
                i_local=i_local,
                m_tiles=m_tiles,
                grid=front_grid(m_tiles, i_local),
                n_tiles=front_n_tiles(i_local),
            )
            key = front_kernel_key(i_local)
            kwargs = dict(
                A=x,
                WG=gate_weight,
                WD=down_weight,
                WS=shared_gate_up_weight,
                logits=logits,
                latent=latent,
                shared_act=shared_act,
                M=T,
                m_tiles=m_tiles,
                grid=(int(plan["grid"]), 1, 1),
            )
            assert tuple(kwargs) == FRONT_KWARGS
            module = kernel_module_name(arch, key)
            entry, arguments = _bind(module, kwargs)
            launches = (_Launch("front_gemm", key, module, kwargs, entry, arguments),)
            route = "prefill"
    return KimiK3LatentMoeRunner(
        "front", tp, 0, T, arch, route, plan, launches, (logits, latent, shared_act)
    )


def kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete front operator for one rank into the caller-owned ``logits`` / ``latent`` / ``shared_act``."""
    return prepare_kimi_k3_latent_moe_front(
        x, gate_weight, down_weight, shared_gate_up_weight, logits, latent, shared_act
    )()


# ---------------------------------------------------------------------------
# Tail
# ---------------------------------------------------------------------------


def prepare_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> KimiK3LatentMoeRunner:
    """Validate the tail operands, plan the route and bind its launch(es).

    ``routed`` is ``[P, T, 3584]`` BF16 (``P`` un-reduced routed-expert partials, summed on
    device); ``y_workspace`` is a caller-owned BF16 ``[T, 3584]`` buffer that receives the
    normalised latent (the GEMM's B operand).  ``up_weight`` is the full ``[7168, 3584]``
    replicated up-projection; this rank multiplies its ``3584 / tp`` column slice.
    ``shared_down_weight`` is the rank-local ``[7168, 6144 / tp]`` shard.
    """
    tp, rank = int(tp), int(rank)
    if tp not in SUPPORTED_TP:
        raise ValueError(f"tp must be one of {SUPPORTED_TP}, got {tp}")
    if not 0 <= rank < tp:
        raise ValueError(f"rank {rank} out of range for tp={tp}")
    if not isinstance(routed, torch.Tensor) or routed.dim() != 3:
        raise ValueError("routed must be a contiguous bf16 [P, T, 3584] tensor")
    P, T = int(routed.shape[0]), int(routed.shape[1])
    _check(routed, (P, T, LATENT), "routed")
    i_local = i_local_for_tp(tp)
    _check(norm_weight, (LATENT,), "norm_weight")
    _check(up_weight, (HIDDEN, LATENT), "up_weight")
    _check(shared_act, (T, i_local), "shared_act")
    _check(shared_down_weight, (HIDDEN, i_local), "shared_down_weight")
    _check(out, (T, HIDDEN), "out")
    _check(y_workspace, (T, LATENT), "y_workspace")
    device = _same_device(
        dict(
            routed=routed,
            norm_weight=norm_weight,
            up_weight=up_weight,
            shared_act=shared_act,
            shared_down_weight=shared_down_weight,
            out=out,
            y_workspace=y_workspace,
        )
    )
    arch = _device_arch(device)
    index = _check_sm_count(device)
    launches: tuple[_Launch, ...]
    plan: dict[str, Any]
    with torch.cuda.device(index):
        if T <= DECODE_MAX_T:
            plan = decode_tail_plan(T, i_local, tp, P)
            key = decode_kernel_key(plan)
            f32_dummy, counters, tl = _scratch(device)
            kwargs = dict(
                A_R=up_weight,
                A_L=up_weight,
                A_S=up_weight,
                A_2=shared_down_weight,
                B_1=y_workspace,
                B_2=shared_act,
                out_r=f32_dummy,
                out_l=out,
                out_s=out,
                counters=counters,
                routed=routed,
                norm_w=norm_weight,
                y_out=y_workspace,
                tl=tl,
                num_tokens=T,
                k1_off=rank * int(plan["k1"]),
                num_partials=P,
                eps=float(RMS_EPS),
                grid=(int(plan["grid"]), 1, 1),
            )
            assert tuple(kwargs) == DECODE_KWARGS
            module = kernel_module_name(arch, key)
            entry, arguments = _bind(module, kwargs)
            launches = (
                _Launch("tail_decode_fused", key, module, kwargs, entry, arguments),
            )
            route = "decode"
        else:
            plan = prefill_tail_plan(T, tp)
            norm_key = norm_kernel_key(plan["early_trigger"])
            norm_kwargs = dict(
                routed=routed,
                norm_weight=norm_weight,
                y_out=y_workspace,
                M=T,
                num_partials=P,
                eps=float(RMS_EPS),
                grid=(int(plan["norm_grid"]), 1, 1),
            )
            assert tuple(norm_kwargs) == NORM_KWARGS
            norm_module = kernel_module_name(arch, norm_key)
            norm_entry, norm_arguments = _bind(norm_module, norm_kwargs)
            ws, counters = _tail_workspace(device, plan["sk_tiles"], plan["sk_max_seg"])
            gemm_key = tail_gemm_kernel_key(tp)
            gemm_kwargs = dict(
                A1=y_workspace,
                B1=up_weight,
                A2=shared_act,
                B2=shared_down_weight,
                out=out,
                ws=ws,
                counters=counters,
                M=T,
                m_tiles=int(plan["m_tiles"]),
                k0_blocks=rank * plan["k_up"] // BLOCK_K,
                num_items=int(plan["num_items"]),
                full_items=int(plan["full_items"]),
                sk_ipc=int(plan["sk_ipc"]),
                sk_max_seg=int(plan["sk_max_seg"]),
                sk_total=int(plan["sk_total"]),
                grid=(int(plan["gemm_grid"]), 1, 1),
            )
            assert tuple(gemm_kwargs) == TAIL_GEMM_KWARGS
            gemm_module = kernel_module_name(arch, gemm_key)
            gemm_entry, gemm_arguments = _bind(gemm_module, gemm_kwargs)
            launches = (
                _Launch(
                    "tail_norm",
                    norm_key,
                    norm_module,
                    norm_kwargs,
                    norm_entry,
                    norm_arguments,
                ),
                _Launch(
                    "tail_gemm",
                    gemm_key,
                    gemm_module,
                    gemm_kwargs,
                    gemm_entry,
                    gemm_arguments,
                ),
            )
            route = "prefill"
    return KimiK3LatentMoeRunner(
        "tail", tp, rank, T, arch, route, plan, launches, (y_workspace, out)
    )


def kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> torch.Tensor:
    """Complete tail for one rank into the caller-owned ``out`` (``y_workspace`` receives the normalised latent)."""
    prepare_kimi_k3_latent_moe_tail(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
    )()
    return out


__all__ = [
    "DECODE_MAX_T",
    "HIDDEN",
    "LATENT",
    "NUM_EXPERTS",
    "RMS_EPS",
    "ROW_TOKENS",
    "SHARED_INTERMEDIATE",
    "SM_COUNT",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SUPPORTED_TP",
    "KimiK3LatentMoeRunner",
    "decode_front_plan",
    "decode_kernel_key",
    "decode_symbol",
    "decode_tail_plan",
    "front_kernel_key",
    "generated_program_available",
    "i_local_for_tp",
    "k_up_for_tp",
    "kimi_k3_latent_moe_front",
    "kimi_k3_latent_moe_tail",
    "norm_kernel_key",
    "prefill_tail_plan",
    "prepare_kimi_k3_latent_moe_front",
    "prepare_kimi_k3_latent_moe_tail",
    "required_kernel_keys",
    "route_kernel_keys",
    "split_plan",
    "tail_gemm_kernel_key",
]
