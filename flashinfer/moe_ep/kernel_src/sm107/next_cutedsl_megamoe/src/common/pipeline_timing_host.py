"""Host reduction for lightweight per-CTA MegaMoE pipeline timestamps."""

from __future__ import annotations


import numpy as np
from typing import Optional
import torch

from .pipeline_timing import (
    COMBINE_TIMING_FC2_WAIT_NS,
    COMBINE_TIMING_LANDMARK_NAMES,
    COMBINE_TIMING_MAX_FC2_WAIT_NS,
    COMBINE_TIMING_MESSAGE_COUNT,
    COMBINE_TIMING_PUBLISH_ISSUE_NS,
    COMBINE_TIMING_REDUCE_NS,
    COMBINE_TIMING_ROUTE_INCLUSIVE_NS,
    COMBINE_TIMING_SYNC_DONE,
    COMBINE_TIMING_WAITED_HIT_COUNT,
    PIPELINE_STAGE_NAMES,
)

# Diagnostic stages a rank may legitimately leave unset in a launch: a relay
# worker stamps its last message's gather-reduce into exactly one of the two
# path stages, so a rank whose workers all ended on the ring has no scratch
# record (and vice versa). The samples report aggregates these over the
# samples and ranks that carry them instead of treating absence as an error.
OPTIONAL_SAMPLE_STAGES = frozenset({"combine_reduce_scratch", "combine_reduce_ring"})

# Per-worker distribution report: the stages whose device records are one
# (start, end) pair per warp with a meaning of their own (a pack, a fan-out
# phase sum, a Stage 3 walk), so percentiles over warps separate "every warp
# is slow" from "a few stragglers" and the start spread shows release skew.
WARP_DIST_STAGES = (
    "push_prologue", "push_pack", "push_pack_route", "push_pack_copy",
    "fanout_route", "fanout_copy", "fanout_publish", "relay_copy",
    "relay_wait", "relay_work", "token_back", "sf_materialize",
)
WARP_DIST_FIELDS = (
    "dur_p50_us", "dur_p90_us", "dur_max_us",
    "start_min_us", "start_p50_us", "start_max_us",
    "end_p50_us", "end_p90_us", "end_max_us",
)


def summarize_pipeline_timing_warp_dist(
    timing: torch.Tensor, *, anchor_ns: int, stages=WARP_DIST_STAGES,
) -> dict[str, dict[str, float]]:
    """Per-warp duration/start/end percentiles of selected stages.

    ``anchor_ns`` is the rank's own dispatch start; every start/end is
    reported relative to it (same GPU clock, so no cross-rank caveat).
    Durations are ``end - start`` of each valid (start > 0, end >= start)
    worker record, i.e. for the per-warp phase sums the summed phase time.
    """
    host = timing.detach().cpu()
    out: dict[str, dict[str, float]] = {}
    for stage_name in stages:
        if stage_name not in PIPELINE_STAGE_NAMES:
            continue
        stage_idx = PIPELINE_STAGE_NAMES.index(stage_name)
        if stage_idx >= host.shape[2]:
            continue
        starts = host[:, :, stage_idx, 0].reshape(-1).numpy().astype(np.int64)
        ends = host[:, :, stage_idx, 1].reshape(-1).numpy().astype(np.int64)
        valid = (starts > 0) & (ends >= starts)
        if not bool(valid.any()):
            continue
        dur_us = (ends[valid] - starts[valid]) / 1000.0
        start_us = (starts[valid] - anchor_ns) / 1000.0
        end_us = (ends[valid] - anchor_ns) / 1000.0
        out[stage_name] = {
            "workers": int(valid.sum()),
            "dur_p50_us": float(np.percentile(dur_us, 50)),
            "dur_p90_us": float(np.percentile(dur_us, 90)),
            "dur_max_us": float(dur_us.max()),
            "start_min_us": float(start_us.min()),
            "start_p50_us": float(np.percentile(start_us, 50)),
            "start_max_us": float(start_us.max()),
            "end_p50_us": float(np.percentile(end_us, 50)),
            "end_p90_us": float(np.percentile(end_us, 90)),
            "end_max_us": float(end_us.max()),
        }
    return out


def summarize_pipeline_timing(timing: torch.Tensor) -> dict[str, dict[str, int]]:
    """Reduce one GPU clock domain to min(start), max(end) per stage."""
    host = timing.detach().cpu()
    records: dict[str, dict[str, int]] = {}
    for stage_idx, stage_name in enumerate(PIPELINE_STAGE_NAMES):
        if stage_idx >= host.shape[2]:
            # Older snapshots carry fewer stage slots than the current schema.
            break
        starts = host[:, :, stage_idx, 0].reshape(-1)
        ends = host[:, :, stage_idx, 1].reshape(-1)
        valid = (starts > 0) & (ends >= starts)
        worker_count = int(valid.sum().item())
        if worker_count == 0:
            continue
        start_ns = int(starts[valid].min().item())
        end_ns = int(ends[valid].max().item())
        records[stage_name] = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "workers": worker_count,
        }
    return records


def summarize_combine_timing_landmarks(
    landmarks: torch.Tensor,
) -> dict[str, int]:
    """Summarize lifecycle envelopes and worker-local combine work.

    Lifecycle fields use min/max timestamps and are critical-path envelopes.
    Attribution fields are sums or distributions of each workers measured
    message work; they are never produced by subtracting maxima from different
    workers.
    """
    host = landmarks.detach().cpu().reshape(
        -1, len(COMBINE_TIMING_LANDMARK_NAMES)
    )
    completed = host[:, COMBINE_TIMING_SYNC_DONE] > 0
    for edge_idx in range(2, COMBINE_TIMING_SYNC_DONE + 1):
        completed &= host[:, edge_idx] >= host[:, edge_idx - 1]
    worker_count = int(completed.sum().item())
    started = completed & (host[:, 0] > 0) & (host[:, 1] >= host[:, 0])
    started_worker_count = int(started.sum().item())
    if started_worker_count == 0:
        return {}

    start_ns = int(host[started, 0].min().item())
    endpoint_ns = [
        int(host[completed, edge_idx].max().item())
        for edge_idx in range(1, COMBINE_TIMING_SYNC_DONE + 1)
    ]
    if endpoint_ns != sorted(endpoint_ns):
        raise ValueError(
            "combine timing envelope endpoints are not monotonic: "
            f"{endpoint_ns}"
        )
    (
        route_done_ns,
        reduce_done_ns,
        issue_done_ns,
        flush_credit_done_ns,
        arrival_done_ns,
        sync_done_ns,
    ) = endpoint_ns

    active = completed & (host[:, COMBINE_TIMING_MESSAGE_COUNT] > 0)
    active_worker_count = int(active.sum().item())
    if active_worker_count:
        wait_by_worker = host[active, COMBINE_TIMING_FC2_WAIT_NS]
        route_inclusive_by_worker = host[
            active, COMBINE_TIMING_ROUTE_INCLUSIVE_NS
        ]
        if bool(torch.any(route_inclusive_by_worker < wait_by_worker).item()):
            raise ValueError("FC2 wait exceeds its worker route interval")
        route_exclusive_by_worker = (
            route_inclusive_by_worker - wait_by_worker
        )
        reduce_by_worker = host[active, COMBINE_TIMING_REDUCE_NS]
        issue_by_worker = host[
            active, COMBINE_TIMING_PUBLISH_ISSUE_NS
        ]
        message_by_worker = host[
            active, COMBINE_TIMING_MESSAGE_COUNT
        ]
        active_preissue_by_worker = (
            wait_by_worker
            + route_exclusive_by_worker
            + reduce_by_worker
            + issue_by_worker
        )
        per_message = (
            active_preissue_by_worker.to(torch.float64)
            / message_by_worker.to(torch.float64)
        )

        def _quantile_ns(q: float) -> int:
            return int(torch.quantile(per_message, q).item())

        fc2_wait_ns = int(wait_by_worker.sum().item())
        route_inclusive_ns = int(route_inclusive_by_worker.sum().item())
        route_exclusive_ns = int(route_exclusive_by_worker.sum().item())
        reduce_ns = int(reduce_by_worker.sum().item())
        publish_issue_ns = int(issue_by_worker.sum().item())
        message_count = int(message_by_worker.sum().item())
        waited_hit_count = int(
            host[active, COMBINE_TIMING_WAITED_HIT_COUNT].sum().item()
        )
        max_fc2_wait_ns = int(
            host[active, COMBINE_TIMING_MAX_FC2_WAIT_NS].max().item()
        )
        active_preissue_ns = int(active_preissue_by_worker.sum().item())
        per_message_p50_ns = _quantile_ns(0.50)
        per_message_p90_ns = _quantile_ns(0.90)
        per_message_max_ns = int(per_message.max().item())
    else:
        fc2_wait_ns = 0
        route_inclusive_ns = 0
        route_exclusive_ns = 0
        reduce_ns = 0
        publish_issue_ns = 0
        message_count = 0
        waited_hit_count = 0
        max_fc2_wait_ns = 0
        active_preissue_ns = 0
        per_message_p50_ns = 0
        per_message_p90_ns = 0
        per_message_max_ns = 0

    return {
        "start_ns": start_ns,
        "route_done_ns": route_done_ns,
        "reduce_done_ns": reduce_done_ns,
        "issue_done_ns": issue_done_ns,
        "flush_credit_done_ns": flush_credit_done_ns,
        "arrival_done_ns": arrival_done_ns,
        "sync_done_ns": sync_done_ns,
        "fc2_wait_route_lifecycle_ns": route_done_ns - start_ns,
        "reduce_lifecycle_ns": reduce_done_ns - route_done_ns,
        "publish_issue_lifecycle_ns": issue_done_ns - reduce_done_ns,
        "preissue_lifecycle_ns": issue_done_ns - start_ns,
        "flush_credit_lifecycle_ns": flush_credit_done_ns - issue_done_ns,
        "remote_arrival_lifecycle_ns": (
            arrival_done_ns - flush_credit_done_ns
        ),
        "final_sync_lifecycle_ns": sync_done_ns - arrival_done_ns,
        "duration_ns": sync_done_ns - start_ns,
        "workers": worker_count,
        "started_workers": started_worker_count,
        "active_workers": active_worker_count,
        "fc2_wait_active_sum_ns": fc2_wait_ns,
        "route_inclusive_active_sum_ns": route_inclusive_ns,
        "route_exclusive_active_sum_ns": route_exclusive_ns,
        "reduce_active_sum_ns": reduce_ns,
        "publish_issue_active_sum_ns": publish_issue_ns,
        "active_preissue_sum_ns": active_preissue_ns,
        "message_count": message_count,
        "waited_hit_count": waited_hit_count,
        "max_fc2_wait_ns": max_fc2_wait_ns,
        "worker_preissue_per_message_p50_ns": per_message_p50_ns,
        "worker_preissue_per_message_p90_ns": per_message_p90_ns,
        "worker_preissue_per_message_max_ns": per_message_max_ns,
    }


def report_pipeline_timing(
    timing: torch.Tensor,
    combine_landmarks: torch.Tensor,
    *,
    rank: int,
    theory_us: Optional[float] = None,
) -> None:
    """Print local envelopes and rank-aggregated customer metrics."""
    local = summarize_pipeline_timing(timing)
    for stage_name, record in local.items():
        print(
            "[pipeline-timing] "
            f"rank={rank} stage={stage_name} "
            f"start_ns={record['start_ns']} end_ns={record['end_ns']} "
            f"duration_us={record['duration_ns'] / 1000.0:.3f} "
            f"workers={record['workers']} clock_domain=per_gpu_globaltimer",
            flush=True,
        )

    if "dispatch" in local:
        for stage_name, dist in summarize_pipeline_timing_warp_dist(
            timing, anchor_ns=local["dispatch"]["start_ns"],
        ).items():
            print(
                "[pipeline-timing-warp-dist] "
                f"rank={rank} stage={stage_name} workers={dist['workers']} "
                + " ".join(
                    f"{field}={dist[field]:.3f}" for field in WARP_DIST_FIELDS
                )
                + " rel=dispatch_start clock_domain=per_gpu_globaltimer",
                flush=True,
            )

    # Per-warp role exit of the GEMM stage: which warp of a CTA leaves last
    # (max end over CTAs per warp index, relative to the dispatch start).
    fc_idx = PIPELINE_STAGE_NAMES.index("fc1_fc2")
    host_timing = timing.detach().cpu()
    if fc_idx < host_timing.shape[2] and "dispatch" in local:
        t0 = local["dispatch"]["start_ns"]
        starts = host_timing[:, :, fc_idx, 0]
        ends = host_timing[:, :, fc_idx, 1]
        valid = (starts > 0) & (ends >= starts)
        cells = []
        for warp in range(host_timing.shape[1]):
            col = ends[:, warp][valid[:, warp]]
            if col.numel():
                cells.append(f"w{warp}={(int(col.max().item()) - t0) / 1000.0:.1f}")
        if cells:
            print(
                "[pipeline-timing-warp-exit] "
                f"rank={rank} stage=fc1_fc2 rel_dispatch_start_us "
                + " ".join(cells),
                flush=True,
            )

    local_detail = summarize_combine_timing_landmarks(combine_landmarks)
    if local_detail:
        fc_record = local.get("fc1_fc2")
        local_detail["post_fc_exposed_preissue_ns"] = (
            max(0, local_detail["issue_done_ns"] - fc_record["end_ns"])
            if fc_record is not None else 0
        )
        print(
            "[pipeline-timing-detail] "
            f"rank={rank} stage=combine "
            f"start_ns={local_detail['start_ns']} "
            f"issue_done_ns={local_detail['issue_done_ns']} "
            f"sync_done_ns={local_detail['sync_done_ns']} "
            f"preissue_lifecycle_us="
            f"{local_detail['preissue_lifecycle_ns'] / 1000.0:.3f} "
            f"post_fc_exposed_preissue_us="
            f"{local_detail['post_fc_exposed_preissue_ns'] / 1000.0:.3f} "
            f"fc2_wait_active_sum_us="
            f"{local_detail['fc2_wait_active_sum_ns'] / 1000.0:.3f} "
            f"route_exclusive_active_sum_us="
            f"{local_detail['route_exclusive_active_sum_ns'] / 1000.0:.3f} "
            f"reduce_active_sum_us="
            f"{local_detail['reduce_active_sum_ns'] / 1000.0:.3f} "
            f"publish_issue_active_sum_us="
            f"{local_detail['publish_issue_active_sum_ns'] / 1000.0:.3f} "
            f"max_fc2_wait_us="
            f"{local_detail['max_fc2_wait_ns'] / 1000.0:.3f} "
            f"worker_preissue_per_message_p50_us="
            f"{local_detail['worker_preissue_per_message_p50_ns'] / 1000.0:.3f} "
            f"worker_preissue_per_message_p90_us="
            f"{local_detail['worker_preissue_per_message_p90_ns'] / 1000.0:.3f} "
            f"worker_preissue_per_message_max_us="
            f"{local_detail['worker_preissue_per_message_max_ns'] / 1000.0:.3f} "
            f"duration_us={local_detail['duration_ns'] / 1000.0:.3f} "
            f"messages={local_detail['message_count']} "
            f"waited_hits={local_detail['waited_hit_count']} "
            f"workers={local_detail['workers']} "
            f"started_workers={local_detail['started_workers']} "
            f"active_workers={local_detail['active_workers']} "
            "attribution=worker_local_sum lifecycle=rank_wall_envelope "
            "clock_domain=per_gpu_globaltimer",
            flush=True,
        )

    gathered: list[dict[str, dict[str, int]]] = [local]
    gathered_detail: list[dict[str, int]] = [local_detail]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [{} for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, local)
        gathered_detail = [
            {} for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered_detail, local_detail)
    if rank != 0:
        return

    for stage_name in PIPELINE_STAGE_NAMES:
        ranked = [
            (peer_rank, peer[stage_name])
            for peer_rank, peer in enumerate(gathered)
            if stage_name in peer
        ]
        if not ranked:
            continue
        max_rank, max_record = max(
            ranked, key=lambda item: item[1]["duration_ns"]
        )
        min_start_rank, min_start_record = min(
            ranked, key=lambda item: item[1]["start_ns"]
        )
        max_end_rank, max_end_record = max(
            ranked, key=lambda item: item[1]["end_ns"]
        )
        max_rank_us = max_record["duration_ns"] / 1000.0
        raw_cross_rank_us = (
            max_end_record["end_ns"] - min_start_record["start_ns"]
        ) / 1000.0
        theory_suffix = ""
        if stage_name == "combine" and theory_us is not None:
            theory_suffix = (
                f" theory_us={theory_us:.3f} "
                f"max_rank_minus_theory_us={max_rank_us - theory_us:.3f} "
                f"max_rank_over_theory={max_rank_us / theory_us:.3f}"
            )
        print(
            "[pipeline-timing-global] "
            f"stage={stage_name} ranks={len(ranked)} "
            f"max_rank={max_rank} max_rank_duration_us={max_rank_us:.3f} "
            f"raw_cross_rank_envelope_us={raw_cross_rank_us:.3f} "
            f"raw_min_start_rank={min_start_rank} "
            f"raw_max_end_rank={max_end_rank}"
            f"{theory_suffix} "
            "cross_rank_clock_caveat=globaltimer_epoch_not_cuda_guaranteed",
            flush=True,
        )

    ranked_detail = [
        (peer_rank, record)
        for peer_rank, record in enumerate(gathered_detail)
        if record
    ]
    if ranked_detail:
        max_rank, record = max(
            ranked_detail, key=lambda item: item[1]["duration_ns"]
        )
        tail_rank, tail_record = max(
            ranked_detail,
            key=lambda item: item[1]["post_fc_exposed_preissue_ns"],
        )
        print(
            "[pipeline-timing-detail-global] "
            f"stage=combine ranks={len(ranked_detail)} max_rank={max_rank} "
            f"preissue_lifecycle_us="
            f"{record['preissue_lifecycle_ns'] / 1000.0:.3f} "
            f"post_fc_exposed_preissue_us="
            f"{tail_record['post_fc_exposed_preissue_ns'] / 1000.0:.3f} "
            f"post_fc_max_rank={tail_rank} "
            f"fc2_wait_active_sum_us="
            f"{record['fc2_wait_active_sum_ns'] / 1000.0:.3f} "
            f"route_exclusive_active_sum_us="
            f"{record['route_exclusive_active_sum_ns'] / 1000.0:.3f} "
            f"reduce_active_sum_us="
            f"{record['reduce_active_sum_ns'] / 1000.0:.3f} "
            f"publish_issue_active_sum_us="
            f"{record['publish_issue_active_sum_ns'] / 1000.0:.3f} "
            f"max_fc2_wait_us={record['max_fc2_wait_ns'] / 1000.0:.3f} "
            f"worker_preissue_per_message_p50_us="
            f"{record['worker_preissue_per_message_p50_ns'] / 1000.0:.3f} "
            f"worker_preissue_per_message_p90_us="
            f"{record['worker_preissue_per_message_p90_ns'] / 1000.0:.3f} "
            f"worker_preissue_per_message_max_us="
            f"{record['worker_preissue_per_message_max_ns'] / 1000.0:.3f} "
            f"duration_us={record['duration_ns'] / 1000.0:.3f} "
            f"messages={record['message_count']} "
            f"waited_hits={record['waited_hit_count']} "
            "selection=max_combine_lifecycle_rank "
            "post_fc_selection=max_same_rank_tail "
            "attribution=selected_rank_worker_local_sum "
            "clock_domain=per_gpu_globaltimer",
            flush=True,
        )


def report_pipeline_timing_samples(
    snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    rank: int,
    full_call_critical_ranks: list[int] | None = None,
) -> None:
    """Summarize timed launches and same-rank full-call milestones."""
    local_samples: list[
        tuple[dict[str, dict[str, int]], dict[str, int]]
    ] = []
    for timing, combine_landmarks in snapshots:
        stages = summarize_pipeline_timing(timing)
        detail = summarize_combine_timing_landmarks(combine_landmarks)
        fc_record = stages.get("fc1_fc2")
        if detail:
            detail["post_fc_exposed_preissue_ns"] = (
                max(0, detail["issue_done_ns"] - fc_record["end_ns"])
                if fc_record is not None else 0
            )
        local_samples.append((stages, detail))

    gathered_samples = [local_samples]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_samples = [
            [] for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(
            gathered_samples, local_samples
        )
    if rank != 0:
        return

    sample_counts = {len(samples) for samples in gathered_samples}
    if len(sample_counts) != 1 or not sample_counts or 0 in sample_counts:
        raise ValueError(
            "pipeline timing sample-count mismatch across ranks: "
            f"{sorted(sample_counts)}"
        )
    sample_count = next(iter(sample_counts))
    rank_count = len(gathered_samples)

    if full_call_critical_ranks is not None:
        if len(full_call_critical_ranks) != sample_count:
            raise ValueError(
                "full-call critical-rank sample-count mismatch: "
                f"{len(full_call_critical_ranks)} != {sample_count}"
            )
        invalid_ranks = [
            critical_rank
            for critical_rank in full_call_critical_ranks
            if not 0 <= critical_rank < rank_count
        ]
        if invalid_ranks:
            raise ValueError(
                "full-call critical ranks outside gathered rank range: "
                f"{invalid_ranks} for rank_count={rank_count}"
            )

    for stage_name in PIPELINE_STAGE_NAMES:
        presence = [
            stage_name in stages
            for samples in gathered_samples
            for stages, _ in samples
        ]
        if not any(presence):
            # No device writer in this configuration (for example the GIN-only
            # push / relay_copy sub-envelopes in an NVLink-only build).
            continue
        optional_stage = stage_name in OPTIONAL_SAMPLE_STAGES
        rank_samples_us = []
        for peer_rank, samples in enumerate(gathered_samples):
            peer_values = []
            for sample_idx, (stages, _) in enumerate(samples):
                if stage_name not in stages:
                    if optional_stage:
                        peer_values.append(np.nan)
                        continue
                    raise ValueError(
                        f"stage={stage_name} sample={sample_idx} "
                        f"missing rank={peer_rank}"
                    )
                peer_values.append(
                    stages[stage_name]["duration_ns"] / 1000.0
                )
            rank_samples_us.append(peer_values)
        matrix = np.asarray(rank_samples_us, dtype=np.float64)
        stage_sample_count = sample_count
        if optional_stage:
            # Keep the samples where at least one rank carries the stage;
            # ranks without it in a sample do not take part in that sample.
            keep = ~np.all(np.isnan(matrix), axis=0)
            matrix = matrix[:, keep]
            stage_sample_count = int(matrix.shape[1])
            if stage_sample_count == 0:
                continue
            values = np.nanmax(matrix, axis=0)
            critical_ranks = np.nanargmax(matrix, axis=0).tolist()
            rank_present = ~np.all(np.isnan(matrix), axis=1)
            rank_means = np.full(matrix.shape[0], -1.0)
            rank_means[rank_present] = np.nanmean(matrix[rank_present], axis=1)
        else:
            values = np.max(matrix, axis=0)
            critical_ranks = np.argmax(matrix, axis=0).tolist()
            rank_means = np.mean(matrix, axis=1)
        max_rank_mean_rank = int(np.argmax(rank_means))
        critical_rank_text = ",".join(map(str, critical_ranks))
        print(
            "[pipeline-timing-samples-global] "
            f"stage={stage_name} ranks={rank_count} samples={stage_sample_count} "
            f"iter_rank_max_mean_us={np.mean(values):.3f} "
            f"iter_rank_max_median_us={np.median(values):.3f} "
            f"iter_rank_max_p90_us={np.percentile(values, 90):.3f} "
            f"iter_rank_max_min_us={np.min(values):.3f} "
            f"iter_rank_max_max_us={np.max(values):.3f} "
            f"max_rank_mean_rank={max_rank_mean_rank} "
            f"max_rank_mean_us={rank_means[max_rank_mean_rank]:.3f} "
            f"max_rank_median_us="
            f"{np.median(matrix[max_rank_mean_rank]):.3f} "
            f"critical_ranks={critical_rank_text} "
            "headline=mean_of_iteration_rank_max "
            "deepep_compatible_headline=max_of_rank_means "
            "measurement=instrumented_globaltimer_lifecycle",
            flush=True,
        )

    exposed_tail_us = []
    critical_lifecycle_wait_us = []
    critical_lifecycle_preissue_us = []
    critical_lifecycle_duration_us = []
    critical_lifecycle_ranks = []
    critical_lifecycle_partitions_us = {
        "fc2_wait_route": [],
        "reduce": [],
        "publish_issue": [],
        "flush_credit": [],
        "remote_arrival": [],
        "final_sync": [],
    }
    for sample_idx in range(sample_count):
        ranked_detail = [
            (peer_rank, samples[sample_idx][1])
            for peer_rank, samples in enumerate(gathered_samples)
            if samples[sample_idx][1]
        ]
        if len(ranked_detail) != rank_count:
            raise ValueError(
                f"combine detail sample={sample_idx} rank coverage "
                f"{len(ranked_detail)} != {rank_count}"
            )
        _, tail_record = max(
            ranked_detail,
            key=lambda item: item[1]["post_fc_exposed_preissue_ns"],
        )
        lifecycle_rank, lifecycle_record = max(
            ranked_detail, key=lambda item: item[1]["duration_ns"]
        )
        exposed_tail_us.append(
            tail_record["post_fc_exposed_preissue_ns"] / 1000.0
        )
        critical_lifecycle_wait_us.append(
            lifecycle_record["fc2_wait_active_sum_ns"] / 1000.0
        )
        critical_lifecycle_partitions_us["fc2_wait_route"].append(
            lifecycle_record["fc2_wait_route_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_partitions_us["reduce"].append(
            lifecycle_record["reduce_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_partitions_us["publish_issue"].append(
            lifecycle_record["publish_issue_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_partitions_us["flush_credit"].append(
            lifecycle_record["flush_credit_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_partitions_us["remote_arrival"].append(
            lifecycle_record["remote_arrival_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_partitions_us["final_sync"].append(
            lifecycle_record["final_sync_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_preissue_us.append(
            lifecycle_record["preissue_lifecycle_ns"] / 1000.0
        )
        critical_lifecycle_duration_us.append(
            lifecycle_record["duration_ns"] / 1000.0
        )
        critical_lifecycle_ranks.append(lifecycle_rank)
    tail_values = np.asarray(exposed_tail_us, dtype=np.float64)
    wait_values = np.asarray(
        critical_lifecycle_wait_us, dtype=np.float64
    )
    preissue_values = np.asarray(
        critical_lifecycle_preissue_us, dtype=np.float64
    )
    lifecycle_values = np.asarray(
        critical_lifecycle_duration_us, dtype=np.float64
    )
    partition_values = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in critical_lifecycle_partitions_us.items()
    }
    partition_sum_values = sum(partition_values.values())
    partition_residual_ns = np.rint(
        (lifecycle_values - partition_sum_values) * 1000.0
    ).astype(np.int64)
    if bool(np.any(partition_residual_ns != 0)):
        raise ValueError(
            "combine lifecycle partitions do not conserve the selected "
            f"rank envelope: residual_ns={partition_residual_ns.tolist()}"
        )
    print(
        "[pipeline-timing-detail-samples-global] "
        f"stage=combine ranks={rank_count} samples={sample_count} "
        f"post_fc_exposed_preissue_mean_us={np.mean(tail_values):.3f} "
        f"post_fc_exposed_preissue_median_us={np.median(tail_values):.3f} "
        f"post_fc_exposed_preissue_p90_us="
        f"{np.percentile(tail_values, 90):.3f} "
        f"critical_lifecycle_preissue_mean_us="
        f"{np.mean(preissue_values):.3f} "
        f"critical_lifecycle_fc2_wait_route_mean_us="
        f"{np.mean(partition_values['fc2_wait_route']):.3f} "
        f"critical_lifecycle_reduce_mean_us="
        f"{np.mean(partition_values['reduce']):.3f} "
        f"critical_lifecycle_publish_issue_mean_us="
        f"{np.mean(partition_values['publish_issue']):.3f} "
        f"critical_lifecycle_flush_credit_mean_us="
        f"{np.mean(partition_values['flush_credit']):.3f} "
        f"critical_lifecycle_remote_arrival_mean_us="
        f"{np.mean(partition_values['remote_arrival']):.3f} "
        f"critical_lifecycle_final_sync_mean_us="
        f"{np.mean(partition_values['final_sync']):.3f} "
        f"critical_lifecycle_envelope_mean_us="
        f"{np.mean(lifecycle_values):.3f} "
        f"critical_rank_fc2_wait_active_sum_mean_us="
        f"{np.mean(wait_values):.3f} "
        f"critical_lifecycle_ranks="
        f"{','.join(map(str, critical_lifecycle_ranks))} "
        "post_fc_selection=max_same_rank_tail_per_iteration "
        "lifecycle_selection=max_combine_lifecycle_rank_per_iteration "
        "lifecycle_partitions=same_rank_wall_envelope "
        "attribution=critical_lifecycle_rank_worker_local_sum",
        flush=True,
    )

    if full_call_critical_ranks is not None:
        milestone_names = (
            "dispatch_done",
            "fc1_fc2_done",
            "combine_start",
            "combine_route_lookup_done",
            "reduce_done",
            "issue_done",
            "combine_done",
        )
        milestone_offsets_us = {name: [] for name in milestone_names}
        required_stages = ("dispatch", "fc1_fc2", "combine")
        for sample_idx, critical_rank in enumerate(full_call_critical_ranks):
            stages, detail = gathered_samples[critical_rank][sample_idx]
            missing_stages = [
                name for name in required_stages if name not in stages
            ]
            if missing_stages or not detail:
                raise ValueError(
                    "full-call critical-rank timing coverage missing: "
                    f"sample={sample_idx} rank={critical_rank} "
                    f"stages={missing_stages} detail={bool(detail)}"
                )
            # The stage envelope only includes workers that entered combine
            # (start > 0).  The landmark envelope intentionally also includes
            # workers that participate only in the tail synchronization, so
            # its sync endpoint may be later than the stage endpoint.
            if (
                detail["start_ns"] != stages["combine"]["start_ns"]
                or detail["sync_done_ns"] < stages["combine"]["end_ns"]
            ):
                raise ValueError(
                    "combine lifecycle mismatch on full-call critical rank: "
                    f"sample={sample_idx} rank={critical_rank} "
                    f"detail_start={detail['start_ns']} "
                    f"stage_start={stages['combine']['start_ns']} "
                    f"detail_sync={detail['sync_done_ns']} "
                    f"stage_end={stages['combine']['end_ns']}"
                )
            origin_ns = min(
                stages[name]["start_ns"] for name in required_stages
            )
            milestones_ns = {
                "dispatch_done": stages["dispatch"]["end_ns"],
                "fc1_fc2_done": stages["fc1_fc2"]["end_ns"],
                "combine_start": detail["start_ns"],
                "combine_route_lookup_done": detail["route_done_ns"],
                "reduce_done": detail["reduce_done_ns"],
                "issue_done": detail["issue_done_ns"],
                "combine_done": detail["sync_done_ns"],
            }
            negative = {
                name: value - origin_ns
                for name, value in milestones_ns.items()
                if value < origin_ns
            }
            if negative:
                raise ValueError(
                    "negative full-call critical-rank milestone offset: "
                    f"sample={sample_idx} rank={critical_rank} "
                    f"origin_ns={origin_ns} offsets_ns={negative}"
                )
            for name, value in milestones_ns.items():
                milestone_offsets_us[name].append(
                    (value - origin_ns) / 1000.0
                )
        milestone_text = " ".join(
            f"{name}_offset_mean_us="
            f"{np.mean(milestone_offsets_us[name]):.3f}"
            for name in milestone_names
        )
        print(
            "[pipeline-timing-full-call-rank-samples-global] "
            f"ranks={rank_count} samples={sample_count} "
            "pipeline_work_origin=min_same_rank_stage_start "
            f"{milestone_text} "
            "full_call_critical_ranks="
            f"{','.join(map(str, full_call_critical_ranks))} "
            "selection=full_call_cuda_event_critical_rank_per_iteration "
            "clock_domain=same_rank_per_gpu_globaltimer",
            flush=True,
        )


def reset_pipeline_timing(
    timing: torch.Tensor, combine_landmarks: torch.Tensor
) -> None:
    """Queue a same-stream clear immediately before the measured launch."""
    timing.zero_()
    combine_landmarks.zero_()
