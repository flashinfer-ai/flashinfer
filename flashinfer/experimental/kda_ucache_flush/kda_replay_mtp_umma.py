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

---------------------------------------------------------------------------
KDA (Kimi K3) ReplaySSM verify + ring append + per-request flush on
tcgen05 / UMMA / TMEM (SM10x), the ``backend="umma"`` arm of
``kda_delta_rule_mtp_ucache_flush``.

A fork of the B200 GDN UMMA kernel
(``flashinfer/gdn_kernels/gdn_replay_mtp_umma.py``) for bf16 operands and the
per-channel KDA gate. Contract validation lives in the dispatcher
(``kda_decode_bf16_wy_ucache_flush.kda_delta_rule_mtp_ucache_flush``); this
module only builds and launches the kernel.

Status: plumbing + bf16 UMMA helpers. The kernel itself is not implemented
yet, so ``kda_umma_unavailable_reason`` always reports a reason and the
dispatcher never selects this backend.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import torch
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.gdn_kernels import _umma_helpers as U

RING_SLOTS = 32
RING_MASK = RING_SLOTS - 1
K_DIM = 128
V_DIM = 128
XC_PITCH = 272  # bytes per row of the row-major X_c / Ktil tiles (136 bf16)

# Test hooks (default None = automatic choice). Not part of the API.
_FORCE_U_SRC: Optional[str] = None  # "smem" | "ldg"
_FORCE_HPC: Optional[int] = None
# Number of UMMA launches issued (dispatch tests read it).
_LAUNCHES = 0

# Stage-supported configurations; widened as the kernel grows.
_SUPPORTED_T: frozenset = frozenset()
_SUPPORTED_W_RING: frozenset = frozenset()
_KERNEL_IMPLEMENTED = False


# ----------------------------------------------------------------------------- helpers
# KDA-local variants of the GDN UMMA helpers (flashinfer/gdn_kernels/_umma_helpers.py is
# shared and stays untouched).


def _bf16_idesc(m: int, n: int, *, b_mn_major: bool = False):
    """kind::f16 instruction descriptor with BF16 A and B (a_format / b_format = 1, bits
    7 and 10), fp32 D. ``U.f16_idesc`` leaves both format fields 0 (= F16)."""
    return U.f16_idesc(m, n, b_mn_major=b_mn_major) | Uint32((1 << 7) | (1 << 10))


def _sel(cond, a, b, ty):
    """Branch-free typed select (arith.select)."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


@dsl_user_op
def _ldmatrix_x4(addr: Int32, *, loc=None, ip=None):
    """ldmatrix.sync.aligned.x4.m8n8.shared.b16 from this lane's row address. For an
    m16n8k16 A fragment of a row-major tile: lane l addresses row (l % 16) at column
    offset (l // 16) * 8 elements; returns (a0, a1, a2, a3)."""
    st = llvm.StructType.get_literal([_T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ldmatrix_x2(addr: Int32, *, loc=None, ip=None):
    """ldmatrix.sync.aligned.x2.m8n8.shared.b16 from this lane's row address. For an
    m16n8k16 B fragment of an [n][k] row-major tile: lane l (l < 16 used) addresses row
    n = l % 8 at column offset ((l // 8) % 2) * 8 elements; returns (b0, b1)."""
    st = llvm.StructType.get_literal([_T.i32()] * 2)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {$0,$1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _ex2(x: Float32, *, loc=None, ip=None) -> Float32:
    """ex2.approx.ftz.f32 (the HMMA kernel's _exp2_approx_f32)."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrt.approx.ftz.f32 (the HMMA kernel's _rsqrt_approx_f32)."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _bar_sync_ua(barrier_id: int, number_of_threads: int, *, loc=None, ip=None) -> None:
    """Named-barrier sync without ``.aligned`` (``barrier.cta.sync``), for every named
    barrier used at more than one site (compute-sanitizer synccheck flags a multi-site
    ``.aligned`` barrier as divergent). Arrive side: ``cute.arch.barrier_arrive(...,
    aligned=False)``."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


def _x_sw128_off(row, c):
    """Byte offset of element (row, k = c) in a 16-row K-major SW128 operand tile (X_r,
    and K̃ per 16-row j-block): two 64-element K halves 2048 B apart, 128 B rows, 16 B
    chunks XOR-swizzled with (row % 8). Works on Python ints and DSL integers."""
    return (
        (c // 64) * 2048
        + row * 128
        + ((((c % 64) // 8) ^ (row % 8)) * 16)
        + (c % 8) * 2
    )


def _c_sw32_off(xc, j):
    """Byte offset of C[x = xc][j] (j < 16) in one SW32 [16 x][16 j] bf16 j-block."""
    return xc * 32 + (((j // 8) ^ ((xc >> 2) & 1)) * 16) + (j % 8) * 2


def kda_umma_unavailable_reason(
    dev: torch.device, T: int, w_ring: int, pdl_trigger: bool
) -> Optional[str]:
    """``None`` when the UMMA backend can run this configuration, else why not."""
    if torch.cuda.get_device_capability(dev)[0] != 10:
        cc = torch.cuda.get_device_capability(dev)
        return f"needs SM10x (tcgen05), got SM{cc[0]}{cc[1]}"
    if not _KERNEL_IMPLEMENTED:
        return "the UMMA kernel is not implemented yet"
    if T not in _SUPPORTED_T:
        return f"T={T} not supported (supported: {sorted(_SUPPORTED_T)})"
    if w_ring not in _SUPPORTED_W_RING:
        return f"w_ring={w_ring} not supported (supported: {sorted(_SUPPORTED_W_RING)})"
    if pdl_trigger:
        return "pdl_trigger=True not supported yet"
    return None


def launch_umma(
    q,
    k,
    v,
    g,
    b,
    A_log,
    dt_bias,
    h0,
    initial_state_indices,
    k_cache,
    u_cache,
    g_cache,
    hist_len,
    cache_base,
    out_t,
    scale: float,
    lower_bound: float,
    flush_min: int,
    w_ring: int,
) -> None:
    """Launch the UMMA kernel on already-validated, contiguous inputs."""
    raise NotImplementedError("KDA UMMA kernel not implemented yet")


__all__ = ["kda_umma_unavailable_reason", "launch_umma"]
