"""Configuration for the SM90 native BF16 (Cake-generated GEMMs) push mega-MoE backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig:
    """Static dimensions and protocol choices for the Hopper native BF16 backend.

    Precision contract: bf16 activations and weights into both expert GEMMs,
    fp32 accumulation, bf16 intermediate, bf16 dispatch payload and combine
    wire, bf16 output.  Nothing is quantized to FP8.

    ``clamp_limit`` optionally clamps the FC1 gate/up pre-activations to
    ``[-clamp_limit, clamp_limit]`` (fp32, before SwiGLU); ``None`` disables
    the clamp, which is the reference behaviour.
    ``init_timeout_s`` is applied to the EP process group only while the
    workspace is set up (JIT builds, peer handle exchange); the group's
    previous per-backend timeouts are restored afterwards when PyTorch
    exposes them (``backend.options._timeout``), otherwise the init timeout
    stays in effect for later collectives on that group.

    ``combine_wire`` selects the combine (expert rank -> token owner) wire
    format; every EP rank must use the same value (checked at init):

    * ``"prereduced"`` (default for ``max_tokens_per_rank > 8``): the expert
      rank pre-reduces all routes of a
      token that landed on it in fp32 (``fmaf`` in ascending route order),
      rounds once to bf16 and sends ONE row per (token, source rank); the
      owner sums the <= ep_size rows in ascending rank order in fp32 and
      rounds once.  Fewer bytes on the wire and fewer roundings.
    * ``"prereduced_hilo"``: as ``"prereduced"``, but a source rank holding
      >= 2 routes of a token sends its fp32 partial as two bf16 rows (hi +
      residual) so no group partial is rounded to bf16; single-route groups
      are ``bf16(w * y)`` exactly as the per-route wire.
    * ``"per_route"``: one bf16 row ``bf16(fp32(y_k) * w_k)`` per route, the
      owner sums the top-k rows in fp32 in route order.

    All three are deterministic (fixed reduction order, no floating-point
    atomics); their outputs differ by rounding only.  ``None`` (the default)
    reads the ``FLASHINFER_SM90_CAKE_BF16_COMBINE_WIRE`` environment variable
    when it is set and otherwise selects per shape: ``"per_route"`` when
    ``FleetParams.max_tokens_per_rank <= 8`` (decode rounds at the protocol's
    fixed-cost floor, where the pre-reduced grouping is not repaid; numerics
    identical to the per-route wire), ``"prereduced"`` above.  The capacity is
    a pipe-creation constant identical on every rank, so the per-shape choice
    is rank-consistent; it is still verified by the construction-time
    allgather.

    ``pdl`` launches the backend's own kernels of a round (compact, both
    GEMMs, the combine publish and the combine tail) as programmatic dependent
    launches, so each one is scheduled while its predecessor finishes; a mask
    of the runner's ``PDL_EDGE_*`` bits selects a subset.  The per-route wire
    then publishes through the backend's own copy of the vendored publish
    kernel.  ``pdl_late`` makes the small kernels release their dependents
    only when they exit.  ``fused_begin`` folds the round opening into the
    small-round fused dispatch: ``1`` replaces the tag bump and the ack-wait
    kernels, ``2`` additionally runs the receiver-side prefix in the same
    launch.  Neither switch changes a stored value.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_PDL`` / ``FLASHINFER_SM90_CAKE_BF16_PDL_LATE`` /
    ``FLASHINFER_SM90_CAKE_BF16_FUSED_BEGIN`` (defaults: mask ``6`` = FC1 and FC2,
    ``True``, ``2``).

    ``fused_dispatch_max_tokens`` is the largest round (tokens per rank) that
    takes the single-launch fused dispatch (``None``:
    ``FLASHINFER_SM90_CAKE_BF16_FUSED_DISPATCH_MAX_TOKENS``, default 512, the
    largest value).  It changes no stored value.

    ``grid_dispatch_min_tokens`` routes dedup rounds of at least that many
    tokens per rank through the grid-strided single-launch dispatch (round
    opening, count, reserve, store and, at ``fused_begin`` 2, the receiver
    prefix in one cooperative launch over the co-resident blocks), ahead of
    the fused small-round path.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_GRID_DISPATCH_MIN_TOKENS`` (default ``513``, the
    rounds above the default fused bound; ``0`` = off).  It changes no stored
    value.

    ``compact_form`` and ``publish_form`` select kernel forms that write the
    same bytes: ``1`` runs the latency-pipelined compact (several rows in
    flight, loads issued before stores) and the load-hoisted pre-reduced
    publish (a group's route loads issued before its fp32 fma chain), ``0``
    the one-block-per-row compact and the route-serial publish.  ``None``
    reads ``FLASHINFER_SM90_CAKE_BF16_COMPACT_FORM`` /
    ``FLASHINFER_SM90_CAKE_BF16_PUBLISH_FORM`` (default ``1``).

    ``per_route_publish_pdl`` launches the combine publish of a pipe on the
    per-route wire (by default a token capacity of at most 8) as a
    programmatic dependent launch through the backend's own publish kernel,
    as if ``pdl`` contained the publish edge; pipes on the pre-reduced wires
    are unaffected.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_PER_ROUTE_PUBLISH_PDL`` (default on).  It
    changes no stored value.

    ``fc2_early_max_tokens`` starts FC2 inside FC1's tail wave in rounds with
    at most that many local tokens: FC1 releases a readiness counter per
    128-row tile, FC2 claims its tiles dynamically and reads each tile's input
    once it is released (same tiles, same instruction stream per tile, same
    bits).  ``0`` disables it.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_FC2_EARLY_MAX_TOKENS`` (default ``128``).

    ``fc2_early_windows`` also runs the FC2 early start in larger rounds whose
    local token count lies in the measured window of the pipe's shape
    (hidden size, intermediate size, total experts, top_k, EP size); shapes
    without a window are unaffected.  Same bits.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_FC2_EARLY_WINDOWS`` (default on).

    ``prefix_form`` selects how the single-launch dispatches compute the
    receiver-side prefix (``fused_begin`` 2): ``1`` lets the fused dispatch of
    rounds with ``T * top_k <= 1024`` and the grid dispatch compute it with the
    whole block (an integer block scan over the (expert, source) keys and one
    warp scan per source), ``0`` walks the keys with one thread in every
    dispatch.  Both store the same values.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_PREFIX_FORM`` (default ``1``).

    ``direct_place`` makes the grid dispatch store the first route of each token
    to a rank (its carrier) straight into that rank's activation matrix, which
    becomes a symmetric (peer-writable) buffer, at the row the compact would have
    copied it to; the compact then copies only the later routes of the token to
    the same rank.  It is effective only when every EP rank sets it and every
    rank's pipe can run it (dedup dispatch, the grid dispatch and compact form
    ``1`` on, at most 1024 (expert, rank) keys, ``max_tokens_per_rank`` at least
    the grid-dispatch bound and ``direct_place_min_tokens``); otherwise the layer
    keeps the inbox path and its regular activation matrix without an error.  A
    round is placed directly only when every rank dispatches it through the grid
    dispatch with at least ``direct_place_min_tokens`` local tokens; other rounds
    take the inbox path.
    It changes no stored value.  ``None`` reads
    ``FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE`` (``0`` or ``1``; unset = on at EP
    sizes 8 and 4, where it was measured faster, and off at the others) and
    ``FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE_MIN_TOKENS`` (unset = at EP sizes 8
    and 4 the larger of ``513`` and ``ceil(2**25 / (top_k * hidden_size))``, the
    smallest round whose routes carry ``2**25`` elements: ``586`` at hidden 7168
    and top-k 8, ``781`` at 7168 and 6, ``1366`` at 4096 and 6; at other EP sizes
    ``513``, the grid-dispatch bound).
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm90_bf16_bf16_bf16_push_cake"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    clamp_limit: float | None = None
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0
    combine_wire: str | None = None
    pdl: bool | int | None = None
    pdl_late: bool | None = None
    fused_begin: int | None = None
    fused_dispatch_max_tokens: int | None = None
    grid_dispatch_min_tokens: int | None = None
    compact_form: int | None = None
    publish_form: int | None = None
    per_route_publish_pdl: bool | None = None
    fc2_early_max_tokens: int | None = None
    prefix_form: int | None = None
    direct_place: bool | None = None
    direct_place_min_tokens: int | None = None
    fc2_early_windows: bool | None = None
