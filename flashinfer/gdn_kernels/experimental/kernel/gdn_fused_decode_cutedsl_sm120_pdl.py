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

Fused GDN Decode Step - two-launch CuTe-DSL impl with PDL (SM120)
=================================================================

Host and kernel side of the ``cutedsl_sm120_pdl`` registry impl: a
two-launch CuTe-DSL implementation of the fused GDN decode step, compiled
per (batch, scale, conv-state layout) and launched on the current torch
stream.  The conv-state pool arrives as a logical [P, QKV_DIM,
CONV_STATE_LEN] view of either a transposed SD pool (vLLM default) or a
DS-dense pool; indexing is stride-generic, so the layouts differ only in
which mode carries the static unit stride. The first launch
fuses the depthwise conv1d update with a heavily K-split b/a GEMV (fp32
partials accumulated with device-scope atomics); the second applies the gated
delta rule. Both launches use programmatic dependent launch (PDL); the
wait/trigger contract that makes that safe is stated in full above
:func:`pre_kernel` and pinned by ``tests/gdn/test_fused_decode.py``.

Implements the impl-module interface documented in ../README.md.
Compilation is lazy: the first eager :func:`execute` of a (batch, scale,
conv-state layout) variant compiles it and allocates the per-(batch,
device) workspace; once warm, calls are capture-safe (kernel launches plus
one stream memset).  Consumed by
:mod:`flashinfer.gdn_kernels.experimental.gdn_fused_decode_specialized`;
import errors are tolerated there (the cutlass DSL dependency stays
optional).
"""

from typing import Any

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda_driver
from cutlass.cute.runtime import from_dlpack

from ....cuda_utils import checkCudaErrors
from ._stream_order import order_after_previous_stream

HIDDEN = 5120
N_BA = 96
QKV_DIM = 10240
H_Q = 16
HV = 48
D = 128
CONV_WIDTH = 4
CONV_STATE_LEN = 3
GQA = HV // H_Q  # 3
RPB = 8  # v-rows per block in delta kernel (8 warps = 256 thr)
NRB = D // RPB  # row-blocks per head = 16
KS = 512  # K-split factor for the GEMV (atomic fp32 partials)
KCHUNK = HIDDEN // KS  # 10

NCONV = QKV_DIM // 256  # conv tiles per batch = 40


# ---------------------------------------------------------------------------
# PDL contract for this op -- read this before moving either griddepcontrol.
# ---------------------------------------------------------------------------
# Both launches carry ``use_pdl=True``
# (CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION), which tells the
# driver it "is safe to launch the secondary kernel early and not wait on the
# completion and memory flush of the primary before launching the secondary"
# (CUDA programming guide, Programmatic Dependent Launch).  Two DIFFERENT
# hardware events then keep that safe, and conflating them is the way this
# code goes wrong:
#
#   griddepcontrol.launch_dependents -- a SCHEDULING gate only.  PTX: the
#       designated dependents "can be scheduled as soon as all other CTAs in
#       the grid issue the same instruction or have completed".  It publishes
#       nothing, orders nothing, and is not what a dependent's wait observes.
#   griddepcontrol.wait -- the VISIBILITY barrier.  PTX: the thread waits
#       "until all prerequisite grids in flight have completed and all the
#       memory operations from the prerequisite grids are performed and made
#       visible to the current grid", i.e. until the predecessor grid's
#       grid-ending membar has drained.  This, and only this, makes a read of
#       a predecessor's output legal.
#
# Who waits, who triggers, and what each ordering protects:
#
#   pre_kernel WAITS FIRST, unconditionally, before its first global load and
#       before the conv/GEMV block split so that every block reaches it.
#       Everything it loads is produced by a stream predecessor -- mixed_qkv
#       and hidden by this layer's projections, state_indices by the caller's
#       metadata prep, conv_state by an earlier decode step, and ``part`` by
#       the cuMemsetD32Async this launch is queued behind.  The launch
#       attribute above explicitly releases the driver from waiting on those,
#       so without this wait the op's correctness would rest on whether some
#       unrelated upstream kernel happens to fire a trigger -- a property this
#       op cannot see and must not depend on.
#   pre_kernel THEN TRIGGERS, in the same unconditional prologue.  Order
#       matters and is not cosmetic: the trigger is what places delta_kernel's
#       CTAs on the SMs, and delta_kernel reads ssm_state BEFORE its own wait.
#       Triggering ahead of pre_kernel's wait would schedule delta while
#       pre_kernel is itself still unordered against its predecessors, and
#       delta's pre-wait prefetch would inherit exactly that -- no ordering
#       against whoever last wrote ssm_state/state_indices.  Waiting first is
#       what hands the dependent an already-ordered state.
#   pre_kernel triggers HERE rather than after its stores on purpose.  The
#       trigger does not publish qkv_act/part and is not what releases
#       delta_kernel's wait; delta's wait does not return until pre_kernel's
#       whole grid has completed and flushed, which covers every store in this
#       kernel wherever it sits.  Firing at entry is what buys the overlap the
#       impl exists for (upstream's gdn_decode_bf16_wy_ucache_flush.py fires at
#       entry for the same reason).  What it does cost is SM competition:
#       delta's CTAs are resident and parked on their wait while pre_kernel
#       runs.  That trade is measured, not assumed -- see ../README.md.
#   delta_kernel WAITS in both arms of its padded-row branch, after the
#       ssm_state prefetch and before the first read of qkv_act/part.  That
#       wait is the ordering that protects the conv output and the GEMV
#       partials.
#   delta_kernel does NOT trigger, so nothing downstream is scheduled early on
#       its account and its ssm_state/output stores need no further signal
#       here: a successor is scheduled at this grid's completion.  Adding a
#       trigger to delta_kernel would be a performance change that also
#       obliges every PDL successor to wait -- do not add one without the
#       matching wait on the other side.
#
# Pinned by ``test_pdl_kernels_wait_before_their_first_launch_dependents`` and
# ``test_every_pdl_kernel_waits_on_all_paths`` in tests/gdn/test_fused_decode.py.
@cute.kernel
def pre_kernel(
    mixed_qkv: cute.Tensor,  # [B, QKV_DIM] bf16
    conv_weight: cute.Tensor,  # [QKV_DIM, CONV_WIDTH] bf16
    conv_bias: cute.Tensor,  # [QKV_DIM] bf16
    conv_state: cute.Tensor,  # [P, QKV_DIM, CONV_STATE_LEN] bf16 (in-place)
    state_indices: cute.Tensor,  # [B] int32
    qkv_act: cute.Tensor,  # [B, QKV_DIM] bf16 scratch
    hidden: cute.Tensor,  # [B, HIDDEN] bf16
    w_ba: cute.Tensor,  # [HIDDEN, N_BA] bf16
    part: cute.Tensor,  # [B, N_BA] f32, zeroed before launch
    Bc: cutlass.Constexpr,
):
    """First of the two launches: conv1d state update and the b/a GEMV.

    One grid does both, since neither depends on the other: blocks
    ``[0, B*NCONV)`` shift the conv-state rows, append the new input and write
    the silu-activated conv output to ``qkv_act``; the remaining blocks
    accumulate K-split partials of ``hidden @ w_ba`` into ``part`` with
    gpu-scope atomics (hence the memset of ``part`` before the launch).  It
    waits on its own stream predecessors and then signals PDL launch
    dependents up front, so ``delta_kernel``'s ``ssm_state`` prefetch overlaps
    this entire body -- see the PDL contract block above.
    """
    # Combined conv + K-split GEMV in one launch. Blocks [0, B*NCONV) do conv;
    # blocks [B*NCONV, ...) accumulate GEMV partials with gpu-scope atomics.
    #
    # PDL prologue, above the block split so EVERY block runs both halves.
    # (1) Wait: this grid is launched with programmatic stream serialization,
    # so it may already be running while its predecessors are still draining.
    # Every global load below -- state_indices, conv_state, mixed_qkv, hidden,
    # the weights, and the atomics into the freshly memset `part` -- reads what
    # those predecessors produced, so this is the barrier that makes them
    # legal, not an optimization.
    cute.arch.griddepcontrol_wait()
    # (2) Only then release delta_kernel onto the SMs, so its long-latency
    # ssm_state prefetch overlaps this whole body.  Never before the wait:
    # delta reads global memory before its own wait and would inherit this
    # grid's unordered state.  This does not publish anything -- delta's wait
    # covers pre_kernel's stores wherever they sit.
    cute.arch.griddepcontrol_launch_dependents()

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    if bidx < Bc * NCONV:
        b_idx = bidx // NCONV
        tile = bidx % NCONV
        c = tile * 256 + tidx
        pp = cutlass.Int32(state_indices[b_idx])
        # A negative slot index (vLLM's PAD_SLOT_ID = -1) marks a padded batch
        # row -- what a CUDA-graph replay carries between the live request
        # count and the captured batch size.  It owns no conv-state row, so
        # neither read nor write one for it; ``delta_kernel`` skips the same
        # row and zeroes its output.  The predicate depends only on ``bidx``,
        # so it is block-uniform: one taken branch per block, not divergence.
        # ``qkv_act`` is left untouched for such a row -- it is per-call
        # scratch that only the skipped delta blocks would read.
        if pp >= 0:
            st0_bf = conv_state[pp, c, 0]
            st1_bf = conv_state[pp, c, 1]
            st2_bf = conv_state[pp, c, 2]
            st0 = st0_bf.to(cutlass.Float32)
            st1 = st1_bf.to(cutlass.Float32)
            st2 = st2_bf.to(cutlass.Float32)
            xbf = mixed_qkv[b_idx, c]
            xx = xbf.to(cutlass.Float32)
            w0 = conv_weight[c, 0].to(cutlass.Float32)
            w1 = conv_weight[c, 1].to(cutlass.Float32)
            w2 = conv_weight[c, 2].to(cutlass.Float32)
            w3 = conv_weight[c, 3].to(cutlass.Float32)
            bb = conv_bias[c].to(cutlass.Float32)
            y = cutlass.Float32(st0 * w0 + st1 * w1 + st2 * w2 + xx * w3 + bb)
            ey = cute.math.exp(cutlass.Float32(-y), fastmath=False)
            yv = cutlass.Float32(y / cutlass.Float32(cutlass.Float32(1.0) + ey))
            qkv_act[b_idx, c] = yv.to(cutlass.BFloat16)
            conv_state[pp, c, 0] = st1_bf
            conv_state[pp, c, 1] = st2_bf
            conv_state[pp, c, 2] = xbf
    else:
        g = bidx - Bc * NCONV
        b_idx = g // KS
        ks = g % KS
        n = tidx
        if n < N_BA:
            base = ks * KCHUNK
            acc = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(KCHUNK):
                gi = base + i
                hv = hidden[b_idx, gi].to(cutlass.Float32)
                acc = acc + hv * w_ba[gi, n].to(cutlass.Float32)
            cute.arch.atomic_add(part.iterator + (b_idx * N_BA + n), acc, scope="gpu")


@cute.kernel
def delta_kernel(
    qkv_act: cute.Tensor,  # [B, QKV_DIM] bf16
    part: cute.Tensor,  # [B, N_BA] f32
    A_log: cute.Tensor,  # [HV] f32
    dt_bias: cute.Tensor,  # [HV] bf16
    ssm_state: cute.Tensor,  # [P, HV, D, D] f32 (in-place)
    state_indices: cute.Tensor,  # [B] int32
    output: cute.Tensor,  # [B, 1, HV, D] bf16
    scale: cutlass.Constexpr,
):
    """Second launch: gates, qk-L2-norm and the gated delta-rule update.

    One warp owns one ``ssm_state`` row: it prefetches the row before waiting
    on PDL (``pre_kernel`` never touches ``ssm_state``, so the long-latency
    fp32 loads overlap the whole first kernel), reduces the b/a GEMV partials
    into the beta and decay gates, and writes the updated row back in place
    plus this row's slice of the attention output.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    hh = bidx // NRB  # global head index (b_idx*HV + h)
    rb = bidx % NRB  # row-block within head
    b_idx = hh // HV
    h = hh % HV
    qhd = h // GQA
    warp = tidx // 32
    lane = tidx % 32
    v = rb * RPB + warp  # ssm row index this warp owns

    # Second global read issued before this kernel's PDL wait (with the
    # ssm_state prefetch below).  pre_kernel does not write state_indices;
    # ordering against whoever did is inherited from pre_kernel waiting before
    # it triggers -- see the PDL contract block at the top of this file.
    pp = cutlass.Int32(state_indices[b_idx])

    # A negative slot index (vLLM's PAD_SLOT_ID = -1) marks a padded batch
    # row: it owns no pool slot, so neither state pool may be read or written
    # for it and its output row is zero -- the same contract the fp32 path of
    # gated_delta_rule_decode_pretranspose already documents.  The predicate
    # depends only on bidx, so it is block-uniform: one taken branch per
    # block, and the warp-wide shuffles below are never reached under a
    # partial mask.  Both arms wait on PDL: the padded arm reads nothing
    # pre_kernel wrote and would be correct without it, but keeping the wait
    # uniform across blocks keeps the launch's dependency semantics simple.
    if pp >= 0:
        # fp32 state loads issued before the PDL wait: pre_kernel never writes
        # ssm_state, so these long-latency loads overlap its whole body.  What
        # orders them against the PREVIOUS step's writer of ssm_state is
        # pre_kernel waiting before it triggers: this grid cannot be scheduled
        # until pre_kernel has released it, and pre_kernel only releases after
        # its own wait has drained the predecessors.  Keep that order there.
        k0 = lane
        k1 = lane + 32
        k2 = lane + 64
        k3 = lane + 96
        sv0 = ssm_state[pp, h, v, k0]
        sv1 = ssm_state[pp, h, v, k1]
        sv2 = ssm_state[pp, h, v, k2]
        sv3 = ssm_state[pp, h, v, k3]

        cute.arch.griddepcontrol_wait()

        qbase = qhd * D
        kbase = H_Q * D + qhd * D
        q0 = qkv_act[b_idx, qbase + k0].to(cutlass.Float32)
        q1 = qkv_act[b_idx, qbase + k1].to(cutlass.Float32)
        q2 = qkv_act[b_idx, qbase + k2].to(cutlass.Float32)
        q3 = qkv_act[b_idx, qbase + k3].to(cutlass.Float32)
        kk0 = qkv_act[b_idx, kbase + k0].to(cutlass.Float32)
        kk1 = qkv_act[b_idx, kbase + k1].to(cutlass.Float32)
        kk2 = qkv_act[b_idx, kbase + k2].to(cutlass.Float32)
        kk3 = qkv_act[b_idx, kbase + k3].to(cutlass.Float32)
        vvv = qkv_act[b_idx, 2 * H_Q * D + h * D + v].to(cutlass.Float32)

        # Redundant per-thread gate computation (no smem/sync needed). The fp32
        # gate sums round through bf16 to match the composable path's GEMV output
        # dtype before the sigmoid/softplus gates.
        b_sum = part[b_idx, h]
        a_sum = part[b_idx, HV + h]
        b_bf = b_sum.to(cutlass.BFloat16).to(cutlass.Float32)
        a_bf = a_sum.to(cutlass.BFloat16).to(cutlass.Float32)
        eb = cute.math.exp(cutlass.Float32(-b_bf), fastmath=False)
        beta = cutlass.Float32(1.0) / cutlass.Float32(cutlass.Float32(1.0) + eb)
        dtb = dt_bias[h].to(cutlass.Float32)
        x = cutlass.Float32(a_bf + dtb)
        # softplus via the overflow-free identity
        #     softplus(x) = max(x, 0) + log(1 + exp(-|x|)),
        # branch-free.  The naive log(1 + exp(x)) sends exp(x) to +inf for
        # x > ~88.7 in fp32, which collapses the decay gate to exactly 0 instead
        # of exp(-exp(A_log) * x) -- a silently wrong gate whenever exp(A_log) is
        # small enough for the true gate to stay O(1).  The identity agrees with
        # torch.nn.functional.softplus (threshold=20, i.e. the composable path)
        # and with the CUDA impl's `x > 20 ? x : log1p(exp(x))` to fp32 rounding
        # over the whole range: worst case 1e-6 on the resulting gate, three
        # orders below the bf16 tolerance the correctness tests use.
        #
        # ``max(x, 0)`` is built out of ``absf`` rather than a max primitive on
        # purpose.  ``cute.math`` only gained ``max`` in nvidia-cutlass-dsl 4.6,
        # where the module became a re-export of ``cutlass._mlir_helpers.math``;
        # 4.5's hand-written ``cute.math`` exports 19 names and ``max`` is not
        # among them, so ``cute.math.max`` raises AttributeError there.  ``absf``
        # is in every release this repo supports -- 4.5 defines it, and 4.6+
        # deliberately keeps ``absf`` as an alias of the new ``abs``.  The
        # supported set is pinned by
        # ``test_cutedsl_impl_only_uses_portable_cute_math_primitives``.
        #
        #     max(x, 0) == 0.5*x + 0.5*|x|
        #
        # exact in fp32 for either sign (halving is exact and the two halves are
        # either equal or exact opposites) and, unlike ``0.5*(x + |x|)``, free of
        # the overflow that doubling a large x would introduce.
        x_abs = cute.math.absf(x, fastmath=False)
        x_pos = cutlass.Float32(cutlass.Float32(0.5) * x + cutlass.Float32(0.5) * x_abs)
        ex = cute.math.exp(cutlass.Float32(-x_abs), fastmath=False)
        sp = cutlass.Float32(
            x_pos
            + cute.math.log(cutlass.Float32(cutlass.Float32(1.0) + ex), fastmath=False)
        )
        eal = cute.math.exp(A_log[h], fastmath=False)
        arg = cutlass.Float32(-cutlass.Float32(eal * sp))
        g = cute.math.exp(arg, fastmath=False)

        qsq = cutlass.Float32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        ksq = cutlass.Float32(kk0 * kk0 + kk1 * kk1 + kk2 * kk2 + kk3 * kk3)
        qsq = qsq + cute.arch.shuffle_sync_bfly(qsq, 16)
        qsq = qsq + cute.arch.shuffle_sync_bfly(qsq, 8)
        qsq = qsq + cute.arch.shuffle_sync_bfly(qsq, 4)
        qsq = qsq + cute.arch.shuffle_sync_bfly(qsq, 2)
        qsq = qsq + cute.arch.shuffle_sync_bfly(qsq, 1)
        ksq = ksq + cute.arch.shuffle_sync_bfly(ksq, 16)
        ksq = ksq + cute.arch.shuffle_sync_bfly(ksq, 8)
        ksq = ksq + cute.arch.shuffle_sync_bfly(ksq, 4)
        ksq = ksq + cute.arch.shuffle_sync_bfly(ksq, 2)
        ksq = ksq + cute.arch.shuffle_sync_bfly(ksq, 1)
        q_rms = cute.math.rsqrt(
            cutlass.Float32(qsq + cutlass.Float32(1e-6)), fastmath=False
        )
        k_rms = cute.math.rsqrt(
            cutlass.Float32(ksq + cutlass.Float32(1e-6)), fastmath=False
        )
        qn0 = cutlass.Float32(q0 * q_rms)
        qn1 = cutlass.Float32(q1 * q_rms)
        qn2 = cutlass.Float32(q2 * q_rms)
        qn3 = cutlass.Float32(q3 * q_rms)
        kn0 = cutlass.Float32(kk0 * k_rms)
        kn1 = cutlass.Float32(kk1 * k_rms)
        kn2 = cutlass.Float32(kk2 * k_rms)
        kn3 = cutlass.Float32(kk3 * k_rms)

        ov = cutlass.Float32(
            kn0 * cutlass.Float32(g * sv0)
            + kn1 * cutlass.Float32(g * sv1)
            + kn2 * cutlass.Float32(g * sv2)
            + kn3 * cutlass.Float32(g * sv3)
        )
        ov = ov + cute.arch.shuffle_sync_bfly(ov, 16)
        ov = ov + cute.arch.shuffle_sync_bfly(ov, 8)
        ov = ov + cute.arch.shuffle_sync_bfly(ov, 4)
        ov = ov + cute.arch.shuffle_sync_bfly(ov, 2)
        ov = ov + cute.arch.shuffle_sync_bfly(ov, 1)
        old_v = ov  # all lanes hold the sum
        new_v = cutlass.Float32(
            beta * vvv + cutlass.Float32(cutlass.Float32(1.0) - beta) * old_v
        )
        d = cutlass.Float32(new_v - old_v)

        hs0 = cutlass.Float32(g * sv0 + kn0 * d)
        hs1 = cutlass.Float32(g * sv1 + kn1 * d)
        hs2 = cutlass.Float32(g * sv2 + kn2 * d)
        hs3 = cutlass.Float32(g * sv3 + kn3 * d)
        ssm_state[pp, h, v, k0] = hs0
        ssm_state[pp, h, v, k1] = hs1
        ssm_state[pp, h, v, k2] = hs2
        ssm_state[pp, h, v, k3] = hs3
        outp = cutlass.Float32(qn0 * hs0 + qn1 * hs1 + qn2 * hs2 + qn3 * hs3)
        outp = outp + cute.arch.shuffle_sync_bfly(outp, 16)
        outp = outp + cute.arch.shuffle_sync_bfly(outp, 8)
        outp = outp + cute.arch.shuffle_sync_bfly(outp, 4)
        outp = outp + cute.arch.shuffle_sync_bfly(outp, 2)
        outp = outp + cute.arch.shuffle_sync_bfly(outp, 1)
        if lane == 0:
            out_v = cutlass.Float32(outp * cutlass.Float32(scale))
            output[b_idx, 0, h, v] = out_v.to(cutlass.BFloat16)
    else:
        cute.arch.griddepcontrol_wait()
        if lane == 0:
            output[b_idx, 0, h, v] = cutlass.BFloat16(0.0)


@cute.jit
def fused_launch(
    hidden,
    w_ba,
    mixed_qkv,
    conv_weight,
    conv_bias,
    conv_state,
    A_log,
    dt_bias,
    ssm_state,
    state_indices,
    output,
    qkv_act,
    part,
    stream: cuda_driver.CUstream,
    B: cutlass.Constexpr,
    scale: cutlass.Constexpr,
):
    """Host-side launcher: ``pre_kernel`` then ``delta_kernel``, PDL-chained.

    Compiled once per (batch size, scale, conv-state stride mode); ``B`` and
    ``scale`` are constexpr because both shape the generated code.

    ``use_pdl=True`` on BOTH launches is a promise about the kernels, not just
    a scheduling hint: each of them may start before its stream predecessor has
    completed and flushed, so each must issue ``griddepcontrol_wait()`` before
    reading what a predecessor produced.  Do not add a third PDL launch here
    without reading the contract block above ``pre_kernel``.
    """
    pre_kernel(
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        state_indices,
        qkv_act,
        hidden,
        w_ba,
        part,
        B,
    ).launch(
        grid=[B * NCONV + B * KS, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True
    )
    delta_kernel(
        qkv_act,
        part,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        output,
        scale,
    ).launch(grid=[B * HV * NRB, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True)


# Keyed by the three things that change the generated kernel -- batch size,
# the softmax scale baked in as a constant, and the conv-state stride mode --
# and, for the workspace, by (batch size, device).  The compiled-kernel value
# is whatever ``cute.compile`` returns, which has no public static type.
_compiled: dict[tuple[int, float, int], Any] = {}
_workspace_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_launch_count = 0

# Stream that last used the per-(batch, device) workspace, per device.  The
# workspace (``part``/``qkv_act``) is shared by every call with that batch
# size, so two calls in flight on different streams would interleave writes
# into the same buffers; order_after_previous_stream() (see _stream_order.py)
# makes the later call wait on the earlier one instead.
_workspace_stream: dict[str, torch.cuda.Stream] = {}


def conv_state_leading_dim(conv_state: torch.Tensor) -> int:
    """Stride-1 mode of the logical [P, QKV_DIM, CONV_STATE_LEN] conv-state
    view: 1 for a transposed SD pool (vLLM default), 2 for a DS-dense pool.
    The dispatch gate has already validated the stride pattern."""
    return 1 if conv_state.stride(1) == 1 else 2


def _cache_has(
    B: int, scale: float, device: torch.device, conv_leading_dim: int = 1
) -> bool:
    """True if the (B, scale, conv-state layout) variant is compiled AND the
    per-(B, device) workspace exists — i.e. a call is capture-safe (kernel
    launches plus one stream memset, no compilation or allocation)."""
    return (int(B), float(scale), int(conv_leading_dim)) in _compiled and (
        int(B),
        str(device),
    ) in _workspace_cache


def ready_for_graph_capture(
    hidden_states: torch.Tensor, conv_state: torch.Tensor, scale: float
) -> bool:
    """True when this exact call (batch size, scale, conv-state layout) can
    be recorded into a CUDA graph without compiling or allocating."""
    return _cache_has(
        int(hidden_states.shape[0]),
        float(scale),
        hidden_states.device,
        conv_state_leading_dim(conv_state),
    )


def execute(
    hidden_states,
    w_ba,
    mixed_qkv,
    conv_weight,
    conv_bias,
    conv_state,
    A_log,
    dt_bias,
    scale,
    ssm_state,
    state_indices,
    out=None,
):
    """Run the fused step on the caller's current stream; raise on failure.

    The dispatch layer has already validated the call against the registry
    and the op contract.  Both state pools are updated in place.
    """
    global _launch_count
    B = int(hidden_states.shape[0])
    scale_f = float(scale)
    dev = hidden_states.device

    output = (
        out
        if out is not None
        else torch.empty((B, 1, HV, D), dtype=torch.bfloat16, device=dev)
    )

    # Persistent workspace: the GEMV partials are accumulated with atomics, so
    # `part` is re-zeroed each call with a driver memset on the current stream
    # (no extra torch kernel launch; captured as a memset node under graphs).
    wkey = (B, str(dev))
    workspace = _workspace_cache.get(wkey)
    if workspace is None:
        workspace = (
            torch.zeros((B, N_BA), dtype=torch.float32, device=dev),
            torch.empty((B, QKV_DIM), dtype=torch.bfloat16, device=dev),
        )
        _workspace_cache[wkey] = workspace
    part, qkv_act = workspace
    order_after_previous_stream(_workspace_stream, dev)
    # Take the stream from the tensors' device explicitly rather than from the
    # ambient one: the dispatcher makes that device current before calling in,
    # but an impl that reads torch.cuda.current_stream() with no argument would
    # silently follow the ambient device if that ever stopped being true.
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    # Checked, not fire-and-forget: `part` is the accumulator the K-split GEMV
    # atomically adds into, so a memset that did not run leaves the previous
    # step's partials in place and both gates are silently wrong -- exactly the
    # class of failure the impl attestation exists to make visible. Raising
    # instead hands the call to the dispatch layer, which latches this impl off
    # and serves the composable path.
    checkCudaErrors(
        cuda_driver.cuMemsetD32Async(int(part.data_ptr()), 0, B * N_BA, stream)
    )

    # conv_state is a logical [P, QKV_DIM, CONV_STATE_LEN] view over either a
    # transposed SD pool (stride-1 channels, vLLM default) or a DS-dense pool
    # (stride-1 time steps); pick the matching leading dim and compile one
    # variant per layout (indexing is stride-generic, only the layout's
    # static stride-1 mode differs).
    cs_ld = conv_state_leading_dim(conv_state)

    key = (B, scale_f, cs_ld)
    fn = _compiled.get(key)
    if fn is None:
        # The DLPack markers exist only to describe the argument layouts to
        # cute.compile; the compiled kernel is invoked with the torch tensors
        # themselves. Building them on a warm call would be thirteen
        # from_dlpack round-trips of pure host overhead per decode step per
        # layer, so they are built only on the compiling call.
        m_hidden = from_dlpack(hidden_states, enable_tvm_ffi=True).mark_layout_dynamic()
        m_wba = from_dlpack(w_ba, enable_tvm_ffi=True).mark_layout_dynamic()
        m_qkv = from_dlpack(mixed_qkv, enable_tvm_ffi=True).mark_layout_dynamic()
        m_cw = from_dlpack(conv_weight, enable_tvm_ffi=True).mark_layout_dynamic()
        m_cb = from_dlpack(conv_bias, enable_tvm_ffi=True).mark_layout_dynamic()
        m_cs = from_dlpack(conv_state, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=cs_ld
        )
        m_alog = from_dlpack(A_log, enable_tvm_ffi=True).mark_layout_dynamic()
        m_dtb = from_dlpack(dt_bias, enable_tvm_ffi=True).mark_layout_dynamic()
        m_ssm = from_dlpack(ssm_state, enable_tvm_ffi=True).mark_layout_dynamic()
        m_si = from_dlpack(state_indices, enable_tvm_ffi=True).mark_layout_dynamic()
        m_out = from_dlpack(output, enable_tvm_ffi=True).mark_layout_dynamic()
        m_qa = from_dlpack(qkv_act, enable_tvm_ffi=True).mark_layout_dynamic()
        m_part = from_dlpack(part, enable_tvm_ffi=True).mark_layout_dynamic()
        fn = cute.compile(
            fused_launch,
            m_hidden,
            m_wba,
            m_qkv,
            m_cw,
            m_cb,
            m_cs,
            m_alog,
            m_dtb,
            m_ssm,
            m_si,
            m_out,
            m_qa,
            m_part,
            stream,
            B,
            scale_f,
            options="--enable-tvm-ffi",
        )
        _compiled[key] = fn
    fn(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        output,
        qkv_act,
        part,
        stream,
    )
    _launch_count += 1
    return output, conv_state, ssm_state


def launch_count() -> int:
    """Host-side dispatches so far (a CUDA-graph capture counts once)."""
    return _launch_count


def compiled_variant_keys() -> list:
    """Compiled-kernel descriptors resident in this process."""
    return [
        f"b{B}_scale{scale}_ld{leading_dim}"
        for (B, scale, leading_dim) in sorted(_compiled)
    ]


def variant_plan(rows) -> set:
    """Distinct compiled kernels this impl needs for its registry rows: one
    per (batch size, conv-state layout).  The query scale is also part of
    the compile key but is a runtime value (a serving process uses one
    scale), so the plan counts (b, conv_layout) pairs."""
    return {f"b{row['b']}_{row['conv_layout']}" for row in rows}
