"""Deterministic launch heuristics for the SM120 MegaMoE kernel.

This module is deliberately independent of torch, CUDA, NVSHMEM, argparse,
and process environment state.  The FlashInfer-facing call path can therefore
select one JIT configuration with a small amount of integer arithmetic; the
benchmark runner may apply explicit user overrides after selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


Tile = Tuple[int, int, int]

# Same-NUMA EP4 decode buckets validated on RTX Pro 5000. K1 remains N32;
# only K2 benefits from N16 at these graph-row counts. The mapping value is
# the ready-queue bundle selected by the controlled N16 sweep. Keep the
# single-group K2 until the dual-group 76-CTA replay lifecycle is proven safe
# for the full 4096-wide production shape.
_DECODE_SPLIT_K2_N16_BUNDLE_BY_BUCKET = {
    7: 1,
    16: 2,
    32: 1,
    64: 2,
    128: 2,
}
_DECODE_ROW_BUCKETS = (7, 16, 32, 64, 128, 168, 256)

# Keep both decode experiments opt-in until the exact N16 selector and the
# dual-N8 replay/barrier lifecycle complete production validation. Explicit
# benchmark overrides can still instantiate N16 without affecting callers.
_ENABLE_PRODUCTION_DECODE_N16 = False
_ENABLE_EXPERIMENTAL_DUAL_N8 = False


def _select_decode_row_bucket(rows_per_rank: int) -> Optional[int]:
    if rows_per_rank <= 0:
        raise ValueError("rows_per_rank must be positive")
    for bucket in _DECODE_ROW_BUCKETS:
        if rows_per_rank <= bucket:
            return bucket
    return None


# Blocking IBGDA puts avoid a long late-combine quiet/drain for large payloads,
# but make tiny transfers pay the blocking round trip for every chunk.  This
# byte threshold came from the RTX Pro 5000 EP8 scan and is intentionally
# expressed in transport work rather than model names or token-only bands.
_COMBINE_BLOCKING_PUT_BYTES_PER_PEER = 3 * 1024 * 1024
_COMBINE_BLOCKING_PUT_MAX_CHUNK_BYTES = 128 * 1024
_DISPATCH_MAX_IBGDA_CHUNK_BYTES = 384 * 1024

# The measured RTX Pro 5000 presets remain the tuning anchors, but their SM
# counts must scale to the physical device.  Green Context requires every
# explicitly requested group except the final remainder to respect the
# device's co-schedule alignment.
_REFERENCE_SM_COUNT = 110
_LOCAL_N64_SM_PARTITION = (80, 30)
_LOCAL_DEFAULT_SM_PARTITION = (72, 38)
_HYBRID_SM_PARTITION = (48, 16, 16, 30)


def _scale_sm_partition(
    *,
    num_sms: int,
    reference_counts: Tuple[int, ...],
    min_partition: int,
    alignment: int,
) -> Tuple[int, ...]:
    """Scale a 110-SM preset while preserving Green Context constraints."""

    if sum(reference_counts) != _REFERENCE_SM_COUNT:
        raise ValueError(
            f"reference partition must cover {_REFERENCE_SM_COUNT} SMs, "
            f"got {reference_counts}"
        )
    if num_sms < min_partition * len(reference_counts):
        raise ValueError(
            f"{num_sms} SMs cannot form {len(reference_counts)} partitions "
            f"of at least {min_partition} SMs"
        )

    aligned_minimum = (
        (min_partition + alignment - 1) // alignment * alignment
    )
    targets = tuple(
        num_sms * count / _REFERENCE_SM_COUNT
        for count in reference_counts
    )
    counts = []
    for target in targets[:-1]:
        units = int(target / alignment + 0.5)
        counts.append(max(aligned_minimum, units * alignment))

    remainder = num_sms - sum(counts)
    while remainder < min_partition:
        candidates = [
            index
            for index, count in enumerate(counts)
            if count - alignment >= aligned_minimum
        ]
        if not candidates:
            raise ValueError(
                f"cannot align partition {reference_counts} to {num_sms} SMs"
            )
        # Choose the adjustment with the smallest increase in ratio error.
        index = min(
            candidates,
            key=lambda candidate: (
                abs(counts[candidate] - alignment - targets[candidate])
                - abs(counts[candidate] - targets[candidate])
            ),
        )
        counts[index] -= alignment
        remainder += alignment

    return (*counts, remainder)

@dataclass(frozen=True)
class MegaMoEHeuristicInput:
    """Runtime shape, parallel decomposition, and EP-group topology."""

    tokens_per_rank: int
    hidden: int
    intermediate: int
    num_topk: int
    num_total_experts: int
    data_parallel_size: int
    tensor_parallel_size: int
    expert_parallel_size: int
    ep_same_numa_peer_count: int
    ep_cross_numa_peer_count: int
    num_sms: int
    sm_min_partition: int
    sm_partition_alignment: int

    @property
    def expected_rows_per_expert(self) -> float:
        return (
            self.tokens_per_rank
            * self.expert_parallel_size
            * self.num_topk
            / self.num_total_experts
        )

    @property
    def expected_combine_bytes_per_peer(self) -> float:
        """Uniform-route estimate of BF16 FC2 bytes returned to one peer."""

        return (
            self.tokens_per_rank
            * self.num_topk
            / self.expert_parallel_size
            * self.hidden
            * 2
        )

    def validate(self) -> None:
        positive = {
            "tokens_per_rank": self.tokens_per_rank,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "num_topk": self.num_topk,
            "num_total_experts": self.num_total_experts,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "num_sms": self.num_sms,
            "sm_min_partition": self.sm_min_partition,
            "sm_partition_alignment": self.sm_partition_alignment,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"heuristic inputs must be positive, got {invalid}")
        if self.num_total_experts % self.expert_parallel_size:
            raise ValueError(
                "num_total_experts must be divisible by expert_parallel_size"
            )
        if self.ep_same_numa_peer_count < 0 or self.ep_cross_numa_peer_count < 0:
            raise ValueError("EP peer counts must be non-negative")
        if (
            self.ep_same_numa_peer_count
            + self.ep_cross_numa_peer_count
            != self.expert_parallel_size - 1
        ):
            raise ValueError(
                "same-NUMA and cross-NUMA peers must describe the complete "
                "remote EP group"
            )
        if self.num_sms < 2 * self.sm_min_partition:
            raise ValueError(
                f"{self.num_sms} SMs cannot form two Green Context "
                f"partitions of at least {self.sm_min_partition} SMs"
            )


@dataclass(frozen=True)
class MegaMoEHeuristicOverrides:
    """Benchmark-only explicit overrides; ``None`` means use the heuristic."""

    comm_backend: Optional[str] = None
    token_back_mode: Optional[str] = None
    k1_tile: Optional[Tile] = None
    k2_tile: Optional[Tile] = None
    k2_stages: Optional[int] = None
    k2_warps: Optional[int] = None
    ready_queue_bundle: Optional[int] = None
    k1_sms: Optional[int] = None
    k2_sms: Optional[int] = None
    tx_sms: Optional[int] = None
    rx_sms: Optional[int] = None
    k2_tail_reclaim: Optional[bool] = None
    dispatch_chunk_tokens: Optional[int] = None
    combine_chunk_rows: Optional[int] = None
    dispatch_channels: Optional[int] = None
    combine_channels: Optional[int] = None
    dispatch_slots: Optional[int] = None
    combine_slots: Optional[int] = None
    combine_blocking_put: Optional[bool] = None
    dispatch_local_handoff_windows: Optional[int] = None
    dispatch_remote_handoff_windows: Optional[int] = None
    ibgda_rc_per_pe: Optional[int] = None
    ibgda_rc_mapping: Optional[str] = None
    tp_k3_chunks: Optional[int] = None
    dispatch_compute_overlap: Optional[bool] = None


@dataclass(frozen=True)
class MegaMoEKernelConfig:
    """One fully resolved JIT/launch/transport configuration."""

    comm_backend: str
    kernel_comm_backend: str
    token_back_mode: str

    k1_tile: Tile
    k2_tile: Tile
    k2_stages: int
    k2_warps: int

    k1_sms: int
    k2_sms: int
    tx_sms: int
    rx_sms: int
    sm_min_partition: int
    sm_partition_alignment: int

    dispatch_chunk_tokens: int
    combine_chunk_rows: int
    dispatch_channels: int
    combine_channels: int
    dispatch_slots: int
    combine_slots: int
    combine_blocking_put: bool
    dispatch_local_handoff_windows: int
    dispatch_remote_handoff_windows: int
    ibgda_rc_per_pe: int
    ibgda_rc_mapping: str
    tp_k3_chunks: int

    dispatch_pull_mode: str
    dispatch_warps_per_tile: int
    dispatch_compute_overlap: bool
    k1_ready_queue: bool
    k1_ready_queue_m_rotation: int
    k2_ready_queue: bool
    ready_queue_bundle: int
    k2_natural_regs: bool
    k2_min_blocks_per_sm: int
    k2_tail_reclaim: bool

    expected_rows_per_expert: float

    @property
    def total_sms(self) -> int:
        return self.k1_sms + self.k2_sms + self.tx_sms + self.rx_sms

    def with_overrides(
        self, overrides: MegaMoEHeuristicOverrides
    ) -> "MegaMoEKernelConfig":
        if (
            overrides.ready_queue_bundle is not None
            and overrides.ready_queue_bundle <= 0
        ):
            raise ValueError("ready_queue_bundle must be positive")
        updates = {
            name: value
            for name, value in vars(overrides).items()
            if value is not None
        }
        if not updates:
            return self
        sm_names = ("k1_sms", "k2_sms", "tx_sms", "rx_sms")
        provided_sms = [name for name in sm_names if name in updates]
        if provided_sms and len(provided_sms) != len(sm_names):
            raise ValueError(
                "K1/K2/TX/RX SM overrides form one transport preset; "
                f"provide all four, got {provided_sms}"
            )

        deferred = {
            "comm_backend",
            "token_back_mode",
            *sm_names,
        }
        config = replace(
            self,
            **{name: value for name, value in updates.items() if name not in deferred},
        )
        if overrides.k2_tile is not None:
            token_n = config.k2_tile[1]
            config = replace(
                config,
                k2_stages=(
                    overrides.k2_stages
                    if overrides.k2_stages is not None
                    else (
                        3
                        if (
                            token_n == 64
                            or (
                                token_n == 128
                                and config.expected_rows_per_expert <= 512.0
                            )
                        )
                        else 2
                    )
                ),
                k2_natural_regs=(token_n == 32),
                k2_min_blocks_per_sm=2 if token_n == 32 else 1,
            )

        backend = overrides.comm_backend or config.comm_backend
        if backend == "p2p_direct":
            reference = (
                _LOCAL_N64_SM_PARTITION
                if config.k2_tile[1] == 64
                else _LOCAL_DEFAULT_SM_PARTITION
            )
            k1_sms, k2_sms = _scale_sm_partition(
                num_sms=config.total_sms,
                reference_counts=reference,
                min_partition=config.sm_min_partition,
                alignment=config.sm_partition_alignment,
            )
            config = replace(
                config,
                comm_backend=backend,
                kernel_comm_backend="p2p_direct",
                token_back_mode="epi_warps",
                k1_sms=k1_sms,
                k2_sms=k2_sms,
                tx_sms=0,
                rx_sms=0,
            )
        elif backend in ("nvshmem_hybrid", "nvshmem_ibgda"):
            k1_sms, tx_sms, rx_sms, k2_sms = _scale_sm_partition(
                num_sms=config.total_sms,
                reference_counts=_HYBRID_SM_PARTITION,
                min_partition=config.sm_min_partition,
                alignment=config.sm_partition_alignment,
            )
            config = replace(
                config,
                comm_backend=backend,
                kernel_comm_backend="nvshmem_ibgda",
                token_back_mode="reuse_dispatch_warps",
                k1_sms=k1_sms,
                k2_sms=k2_sms,
                tx_sms=tx_sms,
                rx_sms=rx_sms,
            )
        else:
            raise ValueError(f"unsupported comm backend {backend!r}")

        required_token_back = (
            "epi_warps" if backend == "p2p_direct" else "reuse_dispatch_warps"
        )
        if (
            overrides.token_back_mode is not None
            and overrides.token_back_mode != required_token_back
        ):
            raise ValueError(
                f"comm backend {backend!r} requires token_back_mode "
                f"{required_token_back!r}, got {overrides.token_back_mode!r}"
            )
        if provided_sms:
            config = replace(
                config,
                **{name: updates[name] for name in sm_names},
            )
        return config


def select_megamoe_config(
    shape: MegaMoEHeuristicInput,
    overrides: Optional[MegaMoEHeuristicOverrides] = None,
) -> MegaMoEKernelConfig:
    """Select exactly one SM120 configuration without autotuning."""

    shape.validate()
    rows = shape.expected_rows_per_expert
    # Offline RTX Pro 5000 scan: N32 wins through 28 rows/expert, N64
    # wins from 30 through 63, and N128 wins at 64 and above.
    if rows < 30.0:
        token_n = 32
    elif rows < 64.0:
        token_n = 64
    else:
        token_n = 128

    tile = (64, token_n, 128)
    cross_numa = shape.ep_cross_numa_peer_count > 0

    if cross_numa:
        # The RTX Pro 5000 cross-NUMA scan selected 48/16/16/30. Preserve
        # those proportions on other SM120 devices and align explicit groups
        # to the device's Green Context co-schedule granularity.
        comm_backend = "nvshmem_hybrid"
        kernel_comm_backend = "nvshmem_ibgda"
        k1_sms, tx_sms, rx_sms, k2_sms = _scale_sm_partition(
            num_sms=shape.num_sms,
            reference_counts=_HYBRID_SM_PARTITION,
            min_partition=shape.sm_min_partition,
            alignment=shape.sm_partition_alignment,
        )
    else:
        comm_backend = "p2p_direct"
        kernel_comm_backend = "p2p_direct"
        # N64 needs more K1 waves on this kernel; N32/N128 are balanced by the
        # measured 72/38 split.  This depends on the selected tile, not DP.
        reference = (
            _LOCAL_N64_SM_PARTITION
            if token_n == 64
            else _LOCAL_DEFAULT_SM_PARTITION
        )
        k1_sms, k2_sms = _scale_sm_partition(
            num_sms=shape.num_sms,
            reference_counts=reference,
            min_partition=shape.sm_min_partition,
            alignment=shape.sm_partition_alignment,
        )
        tx_sms, rx_sms = 0, 0

    # A dispatch row contains MXFP8 activations, one E8M0 scale per 32
    # elements, and FP32 top-k weights.  Keeping a cross-NUMA IBGDA operation
    # below roughly 384 KiB avoids the wide-hidden head-of-line stalls seen on
    # RTX Pro 5000.  The 512--1024-token band also benefits from a 32-row
    # first handoff even when a 64-row payload fits the byte limit; above that
    # band the extra operations outweigh the earlier first tile.
    dispatch_bytes_per_token = (
        shape.hidden
        + (shape.hidden + 31) // 32
        + shape.num_topk * 4
    )
    dispatch_chunk_tokens = (
        32
        if (
            cross_numa
            and (
                512 <= shape.tokens_per_rank <= 1024
                or (
                    shape.tokens_per_rank >= 128
                    and dispatch_bytes_per_token * 64
                    > _DISPATCH_MAX_IBGDA_CHUNK_BYTES
                )
            )
        )
        else 64
    )
    dispatch_channels = 2
    combine_channels = 2
    dispatch_slots = 2
    # The EP8 128-token shape produces roughly three combine chunks per
    # peer/channel.  Keeping only two cyclic slots makes the producer wait for
    # an in-epoch credit while its peer is still completing the reciprocal
    # channel, which can form a cross-NUMA progress cycle.  Four slots cover
    # the complete small-message window; terminal ACKs still retire the epoch
    # before the next replay can reuse it.  Larger shapes stay on the measured
    # two-slot fast path so their streaming reuse distance is unchanged.
    combine_slots = (
        4
        if cross_numa and shape.tokens_per_rank <= 128
        else 2
    )
    combine_chunk_rows = 16
    combine_blocking_put = (
        cross_numa
        and shape.expected_combine_bytes_per_peer
        >= _COMBINE_BLOCKING_PUT_BYTES_PER_PEER
        and combine_chunk_rows * shape.hidden * 2
        <= _COMBINE_BLOCKING_PUT_MAX_CHUNK_BYTES
    )
    remote_handoff_windows = 2 if cross_numa else 1

    # Except for the bounded small-message window above, keep production on
    # the shallow two-slot fast path. Fixed peer/channel CTA ownership and
    # independent terminal credits provide forward progress without growing
    # the reuse distance of wide DSV4 rows.

    config = MegaMoEKernelConfig(
        comm_backend=comm_backend,
        kernel_comm_backend=kernel_comm_backend,
        token_back_mode=(
            "reuse_dispatch_warps" if cross_numa else "epi_warps"
        ),
        k1_tile=tile,
        k2_tile=tile,
        k2_stages=(
            3
            if token_n == 64 or (token_n == 128 and rows <= 512.0)
            else 2
        ),
        k2_warps=8,
        k1_sms=k1_sms,
        k2_sms=k2_sms,
        tx_sms=tx_sms,
        rx_sms=rx_sms,
        sm_min_partition=shape.sm_min_partition,
        sm_partition_alignment=shape.sm_partition_alignment,
        dispatch_chunk_tokens=dispatch_chunk_tokens,
        combine_chunk_rows=combine_chunk_rows,
        dispatch_channels=dispatch_channels,
        combine_channels=combine_channels,
        dispatch_slots=dispatch_slots,
        combine_slots=combine_slots,
        combine_blocking_put=combine_blocking_put,
        # Same-NUMA P2P hands off after one window; cross-NUMA IBGDA after
        # two. This early epoch is independent of terminal slot ACK/credit.
        dispatch_local_handoff_windows=1,
        dispatch_remote_handoff_windows=remote_handoff_windows,
        ibgda_rc_per_pe=1,
        ibgda_rc_mapping="warp",
        tp_k3_chunks=(
            8
            if shape.tensor_parallel_size == 2
            and shape.tokens_per_rank >= 8192
            else 1
        ),
        dispatch_pull_mode="token_strided",
        dispatch_warps_per_tile=8,
        dispatch_compute_overlap=shape.tokens_per_rank >= 1024,
        k1_ready_queue=True,
        # Keep the production selector independent of DP layout.  The old
        # DP-only queue rotation was an experiment, not a portable heuristic.
        k1_ready_queue_m_rotation=0,
        k2_ready_queue=True,
        # Very short expert waves need finer-grained K2 distribution; larger
        # waves amortize queue traffic with wider hidden-tile bundles.
        ready_queue_bundle=(
            4 if rows <= 128.0 else 8 if rows <= 512.0 else 16
        ),
        k2_natural_regs=(token_n == 32),
        k2_min_blocks_per_sm=2 if token_n == 32 else 1,
        # The decode-specific selector below enables tail reclaim only for the
        # validated dual-group N16 path. Other shapes keep the regular K2
        # scheduler/workspace contract.
        k2_tail_reclaim=False,
        expected_rows_per_expert=rows,
    )
    if overrides is not None:
        config = config.with_overrides(overrides)
    # The decode optimization is deliberately expressed in graph rows and
    # topology, not in model names. Explicit K2 overrides denote a controlled
    # benchmark and suppress the production tuning.
    explicit_k2_override = overrides is not None and any(
        value is not None
        for value in (overrides.k2_tile, overrides.k2_stages)
    )
    explicit_tail_override = (
        overrides is not None and overrides.k2_tail_reclaim is not None
    )
    decode_bucket = _select_decode_row_bucket(shape.tokens_per_rank)
    decode_bundle = (
        _DECODE_SPLIT_K2_N16_BUNDLE_BY_BUCKET.get(decode_bucket)
        if _ENABLE_PRODUCTION_DECODE_N16
        else None
    )
    if (
        not explicit_k2_override
        and config.kernel_comm_backend == "p2p_direct"
        and config.k1_tile == (64, 32, 128)
        and config.k2_tile == (64, 32, 128)
        and decode_bundle is not None
    ):
        use_dual_group = _ENABLE_EXPERIMENTAL_DUAL_N8
        config = replace(
            config,
            k2_tile=(64, 16, 128),
            k2_stages=2,
            k2_warps=12 if use_dual_group else 8,
            ready_queue_bundle=decode_bundle,
            k2_natural_regs=True,
            k2_min_blocks_per_sm=2 if use_dual_group else 1,
            k2_tail_reclaim=(
                config.k2_tail_reclaim
                if explicit_tail_override
                else use_dual_group
            ),
        )
    if config.total_sms != shape.num_sms:
        raise ValueError(
            "resolved SM partitions must cover all device SMs: "
            f"K1={config.k1_sms}, K2={config.k2_sms}, "
            f"TX={config.tx_sms}, RX={config.rx_sms}, "
            f"total={config.total_sms}, expected={shape.num_sms}"
        )
    if config.tp_k3_chunks <= 0:
        raise ValueError("tp_k3_chunks must be positive")
    if config.dispatch_local_handoff_windows <= 0:
        raise ValueError("dispatch_local_handoff_windows must be positive")
    if config.dispatch_remote_handoff_windows <= 0:
        raise ValueError("dispatch_remote_handoff_windows must be positive")
    if shape.tensor_parallel_size == 1 and config.tp_k3_chunks != 1:
        raise ValueError("TP1 requires tp_k3_chunks=1")
    return config


__all__ = [
    "MegaMoEHeuristicInput",
    "MegaMoEHeuristicOverrides",
    "MegaMoEKernelConfig",
    "select_megamoe_config",
]
