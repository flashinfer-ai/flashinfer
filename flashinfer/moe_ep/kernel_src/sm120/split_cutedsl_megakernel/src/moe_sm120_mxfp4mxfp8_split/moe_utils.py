# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared helpers used by the MoE schedulers and kernels."""

from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Boolean, Int32, T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


# =============================================================================
# Pointer Utilities
# =============================================================================


@dsl_user_op
def _nanosleep(
    sleep_time: int,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Compatibility wrapper for wheels without ``cute.arch.nanosleep``."""
    if cutlass.const_expr(hasattr(cute.arch, "nanosleep")):
        cute.arch.nanosleep(sleep_time=sleep_time, loc=loc, ip=ip)
    else:
        llvm.inline_asm(
            res=None,
            operands_=[Int32(sleep_time).ir_value(loc=loc, ip=ip)],
            asm_string="nanosleep.u32 $0;",
            constraints="r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
@cute.jit
def spin_wait(
    ptr: Pointer,
    condition: Callable[[Int32], bool],
    fail_sleep_cycles: int = 100,
    peek_only: bool = False,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Boolean:
    """Spin until condition is true, or do one condition check with peek_only."""
    current = cute.arch.load(
        ptr, ptr.dtype, sem="acquire", scope="sys", loc=loc, ip=ip
    )
    if cutlass.const_expr(peek_only):
        # One-shot peek: forward the condition Boolean to the caller.
        return Boolean(condition(current))
    while not condition(current):
        if cutlass.const_expr(fail_sleep_cycles > 0):
            _nanosleep(fail_sleep_cycles, loc=loc, ip=ip)
        current = cute.arch.load(
            ptr, ptr.dtype, sem="acquire", scope="sys", loc=loc, ip=ip
        )
    # Spin-path: condition was satisfied; uniformize return type with the
    # peek path so callers always see a Boolean.
    return Boolean(True)


@dsl_user_op
@cute.jit
def spin_wait_i32_ge_inline(
    ptr: Pointer,
    threshold: Int32,
    fail_sleep_cycles: int = 100,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Poll a device-local Int32 counter inside one inline-PTX op."""
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(threshold).ir_value(),
            Int32(fail_sleep_cycles).ir_value(),
        ],
        (
            "{\n\t"
            ".reg .b32 %cur; .reg .pred %ready;\n\t"
            "WAIT_READY: \n\t"
            "ld.acquire.gpu.global.u32 %cur, [$0];\n\t"
            "setp.ge.u32 %ready, %cur, $1;\n\t"
            "@%ready bra READY;\n\t"
            "nanosleep.u32 $2;\n\t"
            "bra WAIT_READY;\n\t"
            "READY: \n\t"
            "}"
        ),
        "l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Cluster-DSMEM helpers (for atomic_counter dynamic scheduler)
# =============================================================================
#
# Ported from cute_dsl_kernel_library/dsl_kernels/moe/moe_persistent_scheduler.py
# (lines 79-145).  Used by the fused fc1+fc2 mega scheduler when
# load_balance_mode == 'atomic_counter' to
# implement the leader-CTA atom.add + DSMEM broadcast cluster-tile-idx
# fetch protocol.  ``atom.add`` itself uses cute.arch.atomic_add (the
# upstream cute_dsl wrapper) instead of a hand-rolled helper.


@dsl_user_op
def store_i32_to_peer_cluster_smem_async(
    smem_ptr,
    value: Int32,
    mbar_ptr,
    cta_rank_in_cluster,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Store one int32 to a peer CTA's SMEM via st.async.shared::cluster.

    Uses ``mapa.shared::cluster`` to translate ``smem_ptr`` / ``mbar_ptr``
    (this CTA's SMEM addresses) into the peer CTA's address space, then
    issues ``st.async.shared::cluster.mbarrier::complete_tx::bytes.u32``
    which both writes the int32 AND signals completion on the peer
    mbarrier.  The peer mbarrier's expect_tx must be set up beforehand
    (see ``mbarrier_arrive_expect_tx_on_peer``).
    """
    smem_addr = llvm.ptrtoint(T.i32(), smem_ptr.llvm_ptr, loc=loc, ip=ip)
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            smem_addr,
            value.ir_value(loc=loc, ip=ip),
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_addr;
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_addr, $0, $3;
            mapa.shared::cluster.u32 remote_mbar, $2, $3;
            st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [remote_addr], $1, [remote_mbar];
        }}""",
        constraints="r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_on_peer(
    mbar_ptr,
    tx_count: Int32,
    cta_rank_in_cluster,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Set expect_tx on a peer CTA's mbarrier via inline PTX.

    Pairs with ``store_i32_to_peer_cluster_smem_async``: this side
    declares "I expect ``tx_count`` bytes via st.async on this peer
    mbarrier"; the store side then completes the transaction.
    """
    mbar_addr = llvm.ptrtoint(T.i32(), mbar_ptr.llvm_ptr, loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            mbar_addr,
            Int32(cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
            tx_count.ir_value(loc=loc, ip=ip),
        ],
        asm_string="""{{
            .reg .u32 remote_mbar;
            mapa.shared::cluster.u32 remote_mbar, $0, $1;
            mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote_mbar], $2;
        }}""",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
# =============================================================================
# MoE Utilities
# =============================================================================


@dsl_user_op
@cute.jit
def compute_expert_token_range(
    offs: cute.Tensor,
    expert_idx: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Int32, Int32]:
    """
    Compute token offset and count for a given expert from the cumsum offs tensor.

    :param offs: Cumulative sum tensor of token counts per expert, shape (experts,)
    :param expert_idx: Index of the expert
    :return: (token_offset, tokens_i) where token_offset is the start position
             and tokens_i is the number of tokens for this expert
    """
    token_offset = Int32(0)
    if expert_idx > Int32(0):
        token_offset = offs[expert_idx - 1]  # type: ignore[assignment]
    tokens_i = offs[expert_idx] - token_offset
    return token_offset, tokens_i


@dsl_user_op
@cute.jit
def compute_expert_token_count_from_sizes(
    sizes: cute.Tensor,
    expert_idx: Int32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """
    Read per-expert token count from a raw sizes tensor.

    This is the sizes-mode counterpart of ``compute_expert_token_range``: it
    returns *only* the count for ``expert_idx``; the cumulative token offset
    is the caller's responsibility (typically maintained as a running cumul in
    scheduler register state, updated when ``expert_idx`` advances).  Used by
    the MegaMoE-fused fc12 scheduler when sizes are exposed as a direct view
    of ``expert_recv_count_sum`` (e.g. via ``i32 stride=(2,)`` over an i64
    tensor) and no cumulative sum kernel was run on the host.
    """
    return sizes[expert_idx]


@dsl_user_op
def rewrite_tensor_shape(
    tensor: cute.Tensor,
    new_shape: Tuple,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Tensor:
    """
    Rewrite tensor shape while keeping the same stride and iterator.

    This is primarily for debug friendliness - shows the actual expert's shape
    instead of the fake global shape. No runtime overhead as it becomes
    dead code in non-debug builds.

    :param tensor: Source tensor whose stride and iterator to preserve
    :param new_shape: New shape to apply
    :return: New tensor with the given shape but original stride and iterator
    """
    new_layout = cute.make_layout(new_shape, stride=tensor.stride, loc=loc, ip=ip)
    return cute.make_tensor(tensor.iterator, new_layout, loc=loc, ip=ip)
