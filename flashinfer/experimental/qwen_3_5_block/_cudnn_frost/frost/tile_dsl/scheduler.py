# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from .constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL  # noqa: F401

from typing import NamedTuple

from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

from .barrier import PipelineState, advance, wait, arrive_expect_tx


@cute.jit
def lpt_tile_coords(linear, q_h, batch, q_tiles):
    hb = q_h * batch
    row_rank = linear // hb
    within = linear % hb
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    head = within % q_h
    batch_idx = within // q_h
    return row, head, batch_idx


@cute.jit
def lpt_l2_tile_coords(linear, q_h, batch, q_tiles, heads_per_kv, seqlen_kv, kv_bytes_per_row, l2_bytes):
    n_kh = q_h // heads_per_kv
    num_groups = n_kh * batch
    # Guard per_group == 0: a zero KV length would divide by zero below.
    per_group_raw = seqlen_kv * cutlass.Int32(kv_bytes_per_row)
    per_group = cutlass.Int32(
        arith.select(
            (per_group_raw < cutlass.Int32(1)).ir_value(),
            cutlass.Int32(1).ir_value(),
            per_group_raw.ir_value(),
        )
    )
    ag_raw = cutlass.Int32(l2_bytes) // per_group
    ag_min1 = cutlass.Int32(
        arith.select(
            (ag_raw < cutlass.Int32(1)).ir_value(),
            cutlass.Int32(1).ir_value(),
            ag_raw.ir_value(),
        )
    )
    active_groups = cutlass.Int32(
        arith.select(
            (ag_min1 > num_groups).ir_value(),
            num_groups.ir_value(),
            ag_min1.ir_value(),
        )
    )
    # Same block count, groups spread evenly: a short last block leaves SMs idle at the tail.
    budget_blocks = (num_groups + active_groups - cutlass.Int32(1)) // active_groups
    active_groups = (num_groups + budget_blocks - cutlass.Int32(1)) // budget_blocks

    tiles_per_grp = q_tiles * heads_per_kv
    tiles_per_blk = active_groups * tiles_per_grp
    num_blocks = (num_groups + active_groups - cutlass.Int32(1)) // active_groups
    block_idx = linear // tiles_per_blk
    within_blk = linear % tiles_per_blk
    # The last block may hold fewer than active_groups groups — clamp so the
    # decoded group never lands outside [0, num_groups).
    is_last_block = (block_idx + cutlass.Int32(1)) == num_blocks
    agroup_eff = cutlass.Int32(
        arith.select(
            is_last_block.ir_value(),
            (num_groups - block_idx * active_groups).ir_value(),
            active_groups.ir_value(),
        )
    )

    row_rank = within_blk // (agroup_eff * heads_per_kv)
    in_rank = within_blk % (agroup_eff * heads_per_kv)
    # in_rank lays out as (sub_head, kv_group): every Q-head sharing a kv_head
    # lands in the same block, so they hit the same resident K/V.
    sub_head = in_rank // agroup_eff
    kv_group = (in_rank % agroup_eff) + block_idx * active_groups
    kv_head = kv_group % n_kh
    batch_idx = kv_group // n_kh
    head = kv_head * heads_per_kv + sub_head
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    return row, head, batch_idx


def _read_tile_id_lane_geometry(cga_size: int):
    """Lane geometry of ``read_tile_id_arrive`` for a ``cga_size``-CTA cluster: ``(lane_stride, log2(lane_stride))``.

    One lane per peer CTA fires, at lanes ``0, lane_stride, 2 * lane_stride, ...`` with ``lane_stride = 32 // cga_size``,
    so ``cga_size`` must be a power of two in [2, 32] (``cga_size == 1`` takes the plain elect path and never gets here).
    A plain Python helper because the check has to ``raise`` at trace time and the DSL rejects a ``raise`` under a staged
    ``if`` inside a ``@cute.jit`` body.
    """
    if cga_size not in (2, 4, 8, 16, 32):
        raise ValueError(f"read_tile_id_arrive: cga_size must be a power of two in [2, 32], got {cga_size}")
    lane_stride = 32 // cga_size
    return lane_stride, lane_stride.bit_length() - 1


@cute.jit
def read_tile_id_arrive(mb, cga_size: int, predicated: cutlass.Constexpr[bool] = True):
    """Credit the scheduler for one decoded ``mb_read_tile_id`` slot: this warp lands ONE arrive on EVERY CTA of the cluster.

    Each CTA's ``mb_read_tile_id[slot]`` is initialised to ``READ_TILE_ARRIVERS`` = the number of warps in the WHOLE cluster
    that call this per work item (25 on the d512 cga4x1 role-split kernels, 29 on the 2-CTA d128 / d192x128 kernels and 30
    on their MXFP8 siblings, whose non-leader MMA warp also credits), so every calling warp has to deliver exactly one arrive
    to each of the ``cga_size`` CTAs.

    ``predicated`` selects the LOWERING of the cluster arm only: the single predicated arrive (default) or the lane-compare
    branch form.  Both fire the same lanes at the same peers -- SUM(issuing lanes with peer == c) == 1 per calling warp for
    every CTA c -- so no ``READ_TILE_ARRIVERS`` init count depends on it.  A kernel spells its choice ONCE as a module
    constant (``PREDICATED_CREDIT_ARRIVE``), a plain Python bool derived from its compile-time mask bits where the sign
    differs per specialization; the sign is MEASURED per kernel and per specialization (see the branch arm below).
    """
    # const_expr, not a plain ``if``: the DSL stages every ``if`` into both arms even when the condition is a Python bool, so
    # a plain ``if cga_size == 1`` also TRACES the cluster arm at cga_size == 1 (the old spelling got away with it because
    # that arm was harmless dead IR there; the helper below would raise).  ``const_expr`` emits the chosen arm only.
    if cutlass.const_expr(cga_size == 1):
        if nvvm.elect_sync():
            nvvm.mbarrier_arrive(mb)
    elif cutlass.const_expr(not predicated):
        # The BRANCH form -- the spelling every kernel used before the predicated arrive below landed (PR #1169): one
        # lane-compare arm per peer CTA.  Same lanes (0, lane_stride, 2 * lane_stride, ...), same peers, same ``.release.cta``
        # arrive, so SUM(issuing lanes with peer == c) == 1 per calling warp for every CTA c and every READ_TILE_ARRIVERS init
        # count is the one the predicated form needs -- only the lowering differs (LDC + BRX jump table + BSSY / BSYNC per arm
        # on sm_107a; ISETP + BSSY + two divergent BRA on sm_100a).  Which form a kernel takes is a MEASURED per-kernel,
        # per-SPECIALIZATION choice, spelled once per kernel as a module constant (the ``SPIN_RING_WAITS`` precedent), never
        # a library default: on sm_100a (B200, A/B/A x3, CUPTI medians, O / Stats / Amax_O bit-identical) the predicated
        # form is a win on per-tensor fp8 d128 (+3.1 %) and bf16 d192x128 (+1.3 %), neutral on bf16 d128 and fp8 d192x128,
        # and a LOSS on the UNMASKED specialization of the two MXFP8 prefill kernels (d128 -5.5 % at B=1 H=24/8 S=16K
        # dense, -5.1 / -5.9 % at S=8K / 32K; d192x128 -6.3 % at H=128 S=8K dense) -- not at the credit sites, which shrink
        # there too, but through a kernel-wide ptxas reschedule that sinks the softmax alpha / stats publish to the end of
        # the exp burst -- while on the d128 kernel's CAUSAL specialization the branch form is the loss (-5.1 % at S=16K,
        # LPT).  So those two pass ``predicated=(CFG.MASK_FLAGS != 0)``: the branch form on their unmasked specialization
        # (byte-identical to the pre-#1169 cubin; +8.1 % / +6.8 % on top of the exp2 split), the predicated form on every
        # masked one (byte-identical to develop's).  A kernel that does not pass the kwarg is untouched.  On every sm107
        # cga >= 2 kernel the predicated form is the measured win (-3.6 % time on d512 fp8), so the default stays True.
        lane_stride = 32 // cga_size
        lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
        for i in cutlass.range_constexpr(cga_size):
            target_lane = i * lane_stride
            if lane == cutlass.Int32(target_lane):
                peer_mb = nvvm.mapa(mb, cutlass.Int32(i))
                nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CTA)
    else:
        # ONE predicated arrive per warp, no lane branches.  The previous spelling,
        #     for i in range_constexpr(cga_size): if lane == i * lane_stride: mbarrier_arrive(mapa(mb, i), ...)
        # lowered to LDC + BRX jump tables + BSSY/BSYNC/BREAK reconverges + one arrive per arm: 51-56 instructions per warp
        # per work item on the sm107 d512 fp8 prefill (the C++ reference does it in 7 plus one ``@P0`` arrive).
        #
        # LANE GEOMETRY (lane = tid & 31):
        #     lane_stride = 32 // cga_size              cga 2 -> 16, cga 4 -> 8, cga 8 -> 4
        #     peer        = lane >> log2(lane_stride)   in [0, cga_size) for EVERY lane (32 = cga_size * lane_stride)
        #     pred        = (lane & (lane_stride - 1)) == 0  and  peer < cga_size
        # so the lanes 0, lane_stride, 2*lane_stride, ... fire: cga 4 -> lanes 0/8/16/24 -> peers 0/1/2/3; cga 2 -> lanes 0/16
        # -> peers 0/1.  Exactly the lanes the branch form fired (``lane == i * lane_stride``), each at a DIFFERENT peer.
        # The ``peer < cga_size`` term is true for every lane < 32 and ptxas folds it (the emitted predicate is
        # ``(tid & (lane_stride - 1)) != 0``); it stays as the written-down guard of the arithmetic.  The shift replaces
        # ``lane // lane_stride`` (a signed floor division on Int32 = shift plus sign fix-ups in SASS).
        #
        # ARRIVE COUNT, per CTA c of the cluster, per calling warp:
        #     SUM(issuing lanes with peer == c) = 1         (the single lane c * lane_stride)
        # so each CTA's mb_read_tile_id[slot] receives READ_TILE_ARRIVERS (= calling warps in the cluster) x 1 arrives per
        # phase = its init count.  UNCHANGED from the branch form: same lanes, same targets, same count; no init changes.
        #
        # ``mapa`` runs unpredicated on every lane (a pure address translation, valid for every peer in [0, cga_size)); only
        # the arrive is predicated.  The inline-PTX template has a single REGISTER operand, the shape ``mbar_arrive_on_peer
        # (pred=)`` in barrier.py already ships; the ``inline_ptx(predicate=)`` hazard (the DSL lowering the predicate onto a
        # trailing IMMEDIATE operand's value) needs an immediate operand, which this template does not have.
        #
        # ORDERING (unchanged): one predicated ``.release.cta`` arrive INSTRUCTION per warp; each issuing lane (cga_size of
        # them) performs its own release-arrive on its peer, exactly as the branch arms did.  This arrive is the credit that
        # lets the scheduler refill the very slot this warp just decoded, so the payload loads must be ordered-before the arrive becomes
        # visible; a relaxed arrive would let the refill race a still-in-flight decode.  CTA scope is enough for that order
        # (the loads are THIS thread's and the refill lands in THIS CTA's slot): ``.release.cluster`` made ptxas drain
        # MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR before every one of these arrives (23 per work item on the d512 kernel, -4.2 %
        # measured 2026-09-18), ``.release.cta`` is one MEMBAR.ALL.CTA.
        #
        # MEASURED (sm107 d512 fp8 prefill, B=1 H=128 S=8192 dense, LSE on, Rubin node locked at 2376 MHz, A/B/A x3 with 50
        # iterations per slot, CUPTI kernel time): 3.0974 -> 2.9854 ms = -3.62 % alone, -4.34 % together with the hint-less
        # ``barrier.wait``; O and LSE bit-identical, S=1024 fp64 validate unchanged.  sm_107a SASS: BRX 16 -> 1, BSSY 17 -> 7,
        # BREAK 25 -> 0, the TMA-STG warp's per-work-item body 127 -> 76 instructions, 20 uniform
        # USYNCS.ARRIVE.TRANS64.RED.ACT0 (the jump-table arms) -> 5 predicated SYNCS.ARRIVE.TRANS64.RED.A1T0 (one per call
        # site), MEMBAR.ALL.CTA 13 unchanged.
        lane_stride, shift = _read_tile_id_lane_geometry(cga_size)  # trace-time; raises on a non-power-of-two cga_size
        lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
        peer = lane >> cutlass.Int32(shift)
        pred = ((lane & cutlass.Int32(lane_stride - 1)) == cutlass.Int32(0)) & (peer < cutlass.Int32(cga_size))
        peer_mb = nvvm.mapa(mb, peer)
        nvvm.inline_ptx(
            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [{$r0}];",
            read_only_args=[peer_mb],
            predicate=pred,
        )


class Sched(NamedTuple):
    mb_scheduler: object
    mb_read_tile_id: object
    tile_id_smem: object
    bidx_init: object
    bidy_init: object
    bidz_init: object


@cute.jit
def read_clc_payload(sched, base_word, *, warp_broadcast: cutlass.Constexpr[bool] = False):
    """Decode one scheduler response slot before returning its slot credit.

    The default requests a 128-bit vector load, mirroring the canonical
    ``cute.arch.clc_response`` decode. The compiler may scalarize it when
    components are unused, so the source-level vector is not an atomicity
    guarantee. A refill during a partial decode could mix two responses.

    ``warp_broadcast`` makes lane zero read the payload and distributes the
    decoded values with full-warp shuffles before the consumer returns its
    slot credit. The single-CTA THD pipeline uses this mode to keep the
    local producer from overwriting a slot while another lane still reads it.

    The trailing cross-proxy fence orders this generic-proxy read before the
    scheduler's NEXT async-proxy write into the slot, which is the DSL's
    documented requirement (see the ``insert_fence`` note in
    ``dynamic_persistent_tile_scheduler.py``).

    ``base_word`` is the Int32 index of the slot's word 0 -- i.e. the same
    expression the raw loads used to subview.  It is taken directly rather
    than derived from a stage index because the per-stage stride is not
    uniform: the predecode kernels size ``tile_id_smem`` by
    ``SCHED_PAYLOAD_WORDS`` (8, 12 or 16) to carry decoded fields after the
    16-byte response.  Every such stride is a multiple of 4 words, so a slot
    base is always 16-byte aligned and the vector load is legal.

    Returns ``(first_ctaid_x, first_ctaid_y, is_valid)`` with is_valid 0/1.
    """
    if cutlass.const_expr(warp_broadcast):
        # The local persistent producer must wait for the last payload reader.
        # Read through lane zero and broadcast before returning slot credit,
        # rather than letting independently scheduled lanes
        # retain shared-memory reads after the elected release-arrive.
        x = cutlass.Int32(0)
        y = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        if (cute.arch.thread_idx()[0] & cutlass.Int32(31)) == 0:
            vec = sched.tile_id_smem.load(base_word, vector_size=4, alignment=16)
            x, y, valid = vec[0], vec[1], vec[2]
        x = cute.arch.shuffle_sync(x, 0)
        y = cute.arch.shuffle_sync(y, 0)
        valid = cute.arch.shuffle_sync(valid, 0)
    else:
        vec = sched.tile_id_smem.load(base_word, vector_size=4, alignment=16)
        x, y, valid = vec[0], vec[1], vec[2]
    nvvm.fence_proxy("async.shared", space="cta")
    return x, y, valid & cutlass.Int32(1)


@cute.jit
def scheduler_warp_loop_persistent(
    sched,
    sched_stages: int,
    is_cga_first_cta,
    meta_t,
    ctr_off,
    live_off,
    cga_size: int,
    cga_m: int,
):
    """Persistent tile scheduler over a LIVE-ONLY unit range (THD).

    CLC sizes the grid to the work list, which for THD means the plan-time
    envelope and therefore dead clusters. Here the grid is occupancy-sized and
    the bound is a DEVICE value (``meta[live_off]``, written by the setup
    launch), so no unit past the live total is ever handed out.

    The cluster lead claims one unit with a global atomic. For a multi-CTA
    cluster it pushes the payload into each CTA's ``tile_id_smem`` over DSMEM
    and completes each peer's scheduler mbarrier. A single CTA publishes to
    local shared memory and arrives its own barrier instead.
    Multicast is not available here: it is a clusterlaunchcontrol facility, so
    dynamic claiming needs an explicit peer write.

    ``cga_size`` is the DSMEM broadcast fan-out (CTAs per cluster); ``cga_m`` is
    the stride the consumer's decode divides back out. They are equal while
    CGA_N == 1, which is every config today -- taking both keeps the handout
    correct if that ever stops holding.
    """
    meta = cutlass.make_array_view(meta_t)
    ctr_ptr = Pointer(meta_t.iterator.raw_ptr(), dtype=cutlass.Int32) + ctr_off
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        # Every CTA expects the 16-byte payload on its own mbarrier, exactly as
        # the CLC path did; the lead's remote store delivers it and completes
        # the barrier through the transaction count.
        if cutlass.const_expr(cga_size > 1):
            if nvvm.elect_sync():
                arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)

        if nvvm.elect_sync() and is_cga_first_cta:
            uid = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
            live = cutlass.Int32(meta[live_off])
            valid = cutlass.Int32(arith.select((uid < live).ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
            # Stride by CGA_M, not the cluster size: the consumer decodes the
            # unit id back out as linear // CGA_M.
            linear = uid * cutlass.Int32(cga_m)
            # store_async_dsmem wants CuTe pointers; the smem arrays hand out
            # base-DSL ones, so convert through the raw addresses.
            _tile_ptr = cute.make_ptr(
                cutlass.Int32,
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)).data_ptr().toint(cutlass.Int32),
                cutlass.AddressSpace.smem,
                assumed_align=16,
            )
            _mbar_ptr = cute.make_ptr(
                cutlass.Int64,
                sched.mb_scheduler.subview(state.idx).data_ptr().toint(cutlass.Int32),
                cutlass.AddressSpace.smem,
                assumed_align=8,
            )
            # One word at a time rather than a v4 payload: store_async_dsmem
            # accepts a 2/4-tuple per its contract, but lowers whatever it was
            # handed through Int32(value), so a tuple raises at trace time. Four
            # scalar stores carry the same 16 bytes and so satisfy the same
            # transaction count the arrive above expects.
            _payload = (linear, cutlass.Int32(0), valid, cutlass.Int32(0))
            if cutlass.const_expr(cga_size == 1):
                # No remote CTA exists. Publish through ordinary shared memory
                # and a release arrive; DSMEM async stores are not a legal
                # one-CTA substitute (Compute Sanitizer rejects the launch).
                tile = cute.make_tensor(_tile_ptr, cute.make_layout(4))
                for w in cutlass.range_constexpr(4):
                    tile[w] = _payload[w]
                nvvm.mbarrier_arrive(sched.mb_scheduler.subview(state.idx))
            else:
                for i in cutlass.range_constexpr(cga_size):
                    for w in cutlass.range_constexpr(4):
                        cute.arch.store_async_dsmem(_tile_ptr + w, _payload[w], _mbar_ptr, i)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        _m, _n, is_valid = read_clc_payload(sched, state.idx * cutlass.Int32(8))

        state = advance(state, sched_stages)


@cute.jit
def scheduler_warp_loop(sched, sched_stages: int, is_cga_first_cta, cga_size: int):
    """``cga_size`` deliberately has NO default.  Only the leader arms the
    barriers below, so a caller that omitted it would leave every peer CTA
    without an expect_tx while the multicast still delivers that peer its 16
    bytes -- the peer's wait would then never complete.  Requiring the
    argument turns that into a trace-time error rather than a cluster hang.
    """
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        # Canonical CLC ordering (CUTLASS PipelineClcFetchAsync): the FIRST
        # CTA's scheduler warp arms arrive+expect_tx(16) on EVERY CTA's
        # barrier, program-ordered BEFORE it issues the multicast try_cancel.
        # Arming per-CTA locally instead leaves a peer's expect_tx unordered
        # against the leader's issue -- both CTAs leave the mb_read_tile_id
        # wait independently, so the multicast response's complete-tx can
        # reach a peer barrier that has not been armed for this phase yet.
        if nvvm.elect_sync() and is_cga_first_cta:
            for i in cutlass.range_constexpr(cga_size):
                if cutlass.const_expr(i == 0):
                    arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
                else:
                    peer_mb = nvvm.mapa(sched.mb_scheduler.subview(state.idx), cutlass.Int32(i))
                    # The arm only has to be program-ordered before the multicast try_cancel issued below by the SAME
                    # thread; a complete-tx that lands before the arm leaves the tx-count transiently negative, which
                    # the mbarrier permits, so no cluster-scope release (= a GPU-scope drain) is needed here either.
                    nvvm.mbarrier_arrive_expect_tx(peer_mb, 16, scope=nvvm.MemScope.CTA)
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        _m, _n, is_valid = read_clc_payload(sched, state.idx * cutlass.Int32(8))

        state = advance(state, sched_stages)
