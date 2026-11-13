"""Host-side reduction of the per-tile FC1/FC2 timeline (MEGA_FC_TILE_TIMELINE=1).

The device buffer holds one record per work tile per CTA (see
``common.pipeline_timing.FC_TILE_FIELD_NAMES``). This module turns it into
three kinds of lines, all on one GPU's ``%globaltimer`` clock and relative to
the rank's first tile claim:

* ``[fc-tile-timeline]`` one summary per rank: the FC window, the four
  per-tile sums divided by the CTA count (readiness wait, mainloop,
  epilogue, gap), the busy fraction, and the FC1-to-FC2 boundary.
* ``[fc-tile-occupancy]`` one histogram per rank: for equal-width bins over
  the FC window, the fraction of CTAs that have a claimed tile and the
  fraction still waiting for readiness.
* ``[fc-tile]`` one line per recorded tile when ``MEGA_FC_TILE_TIMELINE_DUMP``
  is not ``0``: identity plus the four edges.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .pipeline_timing import FC_TILE_FIELD_NAMES, FC_TILE_PHASE_FC1, FC_TILE_PHASE_FC2

_F = {name: index for index, name in enumerate(FC_TILE_FIELD_NAMES)}


def _quantiles_us(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "p50=nan p90=nan max=nan"
    sorted_values = torch.sort(values.double()).values
    p50 = sorted_values[(sorted_values.numel() - 1) // 2].item() / 1000.0
    p90 = sorted_values[int(0.9 * (sorted_values.numel() - 1))].item() / 1000.0
    return f"p50={p50:.1f} p90={p90:.1f} max={sorted_values[-1].item() / 1000.0:.1f}"


def _median(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.sort(values.double()).values[(values.numel() - 1) // 2].item())


def summarize_fc_tile_timeline(timeline: torch.Tensor) -> Dict[str, object]:
    """Reduce the raw buffer into per-rank numbers (nanoseconds unless noted).

    A tile counts as recorded when its claim edge is non-zero. Tiles whose
    later edges are still zero (a warp dropped the slot, or the buffer was
    captured mid-flight) are excluded from the sums that need them.
    """

    if timeline.dim() != 3 or timeline.shape[2] != len(FC_TILE_FIELD_NAMES):
        raise ValueError("fc tile timeline must have shape (ctas, tiles, fields).")
    tl = timeline.detach().to("cpu", torch.int64)
    claim = tl[..., _F["claim_ns"]]
    recorded = claim > 0
    per_cta_tiles = recorded.sum(dim=1)
    active_ctas = int((per_cta_tiles > 0).sum().item())
    if active_ctas == 0:
        return {"tiles": 0, "ctas": 0}
    truncated_ctas = int((recorded[:, -1]).sum().item())

    phase = tl[..., _F["phase"]][recorded]
    claim_r = claim[recorded]
    ready = tl[..., _F["ready_ns"]][recorded]
    mma_end = tl[..., _F["mma_end_ns"]][recorded]
    epi_end = tl[..., _F["epi_end_ns"]][recorded]
    origin = int(claim_r.min().item())
    complete = (ready > 0) & (mma_end > 0) & (epi_end > 0)
    window_end = int(epi_end[complete].max().item()) if complete.any() else origin
    window = window_end - origin

    def total(mask, a, b):
        m = mask & complete
        return int((b[m] - a[m]).clamp_min(0).sum().item())

    is_fc1 = phase == FC_TILE_PHASE_FC1
    is_fc2 = phase == FC_TILE_PHASE_FC2
    all_mask = torch.ones_like(is_fc1)

    # Gap between consecutive tiles on one CTA: the next claim after this
    # tile's epilogue end. Negative gaps mean the next tile was already
    # claimed while this one was finishing (the warps pipeline tiles).
    gap_total = 0
    for cta in torch.nonzero(per_cta_tiles > 0, as_tuple=False).reshape(-1).tolist():
        rows = torch.nonzero(recorded[cta], as_tuple=False).reshape(-1)
        c = tl[cta, rows, _F["claim_ns"]]
        e = tl[cta, rows, _F["epi_end_ns"]]
        for i in range(rows.numel() - 1):
            if e[i] > 0:
                gap_total += max(0, int(c[i + 1].item() - e[i].item()))

    tmab_wait = tl[..., _F["tmab_wait_cyc"]][recorded].double()
    tmab_loop = tl[..., _F["tmab_loop_cyc"]][recorded].double().clamp_min(1)
    mma_wait = tl[..., _F["mma_wait_cyc"]][recorded].double()
    mma_loop = tl[..., _F["mma_loop_cyc"]][recorded].double().clamp_min(1)
    mma_parts = {
        name: tl[..., _F[name]][recorded].double()
        for name in ("mma_acquire_cyc", "mma_s2t_cyc", "mma_issue_cyc", "mma_release_cyc")
    }
    fc1_last_epi_end = int(epi_end[is_fc1 & complete].max().item()) if (is_fc1 & complete).any() else 0
    fc2_first_claim = int(claim_r[is_fc2].min().item()) if is_fc2.any() else 0
    fc2_first_ready = int(ready[is_fc2 & complete].min().item()) if (is_fc2 & complete).any() else 0

    return {
        "tiles": int(recorded.sum().item()),
        "fc1_tiles": int(is_fc1.sum().item()),
        "fc2_tiles": int(is_fc2.sum().item()),
        "ctas": active_ctas,
        "truncated_ctas": truncated_ctas,
        "tiles_per_cta_min": int(per_cta_tiles[per_cta_tiles > 0].min().item()),
        "tiles_per_cta_max": int(per_cta_tiles.max().item()),
        "origin_ns": origin,
        "window_ns": window,
        "ready_wait_total_ns": total(all_mask, claim_r, ready),
        "mainloop_total_ns": total(all_mask, ready, mma_end),
        "epilogue_total_ns": total(all_mask, mma_end, epi_end),
        "tile_span_total_ns": total(all_mask, claim_r, epi_end),
        "gap_total_ns": gap_total,
        "fc1_ready_wait_total_ns": total(is_fc1, claim_r, ready),
        "fc2_ready_wait_total_ns": total(is_fc2, claim_r, ready),
        "fc1_mainloop_ns": (mma_end - ready)[is_fc1 & complete],
        "fc2_mainloop_ns": (mma_end - ready)[is_fc2 & complete],
        "fc1_tmab_wait_frac": (tmab_wait / tmab_loop)[is_fc1 & complete],
        "fc2_tmab_wait_frac": (tmab_wait / tmab_loop)[is_fc2 & complete],
        "fc1_mma_wait_frac": (mma_wait / mma_loop)[is_fc1 & complete],
        "fc2_mma_wait_frac": (mma_wait / mma_loop)[is_fc2 & complete],
        "fc1_mma_part_frac": {
            name: (values / mma_loop)[is_fc1 & complete] for name, values in mma_parts.items()
        },
        "fc2_mma_part_frac": {
            name: (values / mma_loop)[is_fc2 & complete] for name, values in mma_parts.items()
        },
        "fc1_mma_loop_cyc": mma_loop[is_fc1 & complete],
        "fc2_mma_loop_cyc": mma_loop[is_fc2 & complete],
        "fc1_last_epi_end_rel_ns": fc1_last_epi_end - origin if fc1_last_epi_end else 0,
        "fc2_first_claim_rel_ns": fc2_first_claim - origin if fc2_first_claim else 0,
        "fc2_first_ready_rel_ns": fc2_first_ready - origin if fc2_first_ready else 0,
    }


def occupancy_bins(timeline: torch.Tensor, *, bins: int = 20) -> Dict[str, List[float]]:
    """Fraction of active CTAs with a claimed tile / a readiness wait per bin."""

    tl = timeline.detach().to("cpu", torch.int64)
    claim = tl[..., _F["claim_ns"]]
    ready = tl[..., _F["ready_ns"]]
    epi_end = tl[..., _F["epi_end_ns"]]
    recorded = claim > 0
    if not recorded.any():
        return {"bin_us": 0.0, "busy": [], "waiting": []}
    active_ctas = int((recorded.sum(dim=1) > 0).sum().item())
    origin = int(claim[recorded].min().item())
    complete = recorded & (epi_end > 0)
    end = int(epi_end[complete].max().item()) if complete.any() else origin + 1
    width = max(1, (end - origin) // bins)
    busy: List[float] = []
    waiting: List[float] = []
    for b in range(bins):
        t = origin + b * width + width // 2
        in_tile = recorded & (claim <= t) & ((epi_end == 0) | (epi_end > t))
        in_wait = recorded & (claim <= t) & (ready > t)
        busy.append(round(float(in_tile.any(dim=1).sum().item()) / active_ctas, 3))
        waiting.append(round(float(in_wait.any(dim=1).sum().item()) / active_ctas, 3))
    return {"bin_us": width / 1000.0, "busy": busy, "waiting": waiting}


def report_fc_tile_timeline(
    timeline: torch.Tensor,
    *,
    rank: int,
    dispatch_start_ns: int | None = None,
    dump: bool = True,
) -> None:
    """Print the summary, the occupancy histogram, and the per-tile lines.

    ``dispatch_start_ns`` (the rank's earliest dispatch-stage start from the
    pipeline-timing buffer, same GPU clock) places the FC window on the
    kernel's timeline: ``first_claim_after_dispatch_start_us`` is when the
    first FC tile left the scheduler ring.
    """

    summary = summarize_fc_tile_timeline(timeline)
    if not summary.get("tiles"):
        print(f"[fc-tile-timeline] rank={rank} tiles=0", flush=True)
        return
    ctas = summary["ctas"]
    window = max(1, summary["window_ns"])
    alignment = ""
    if dispatch_start_ns:
        first = (summary["origin_ns"] - dispatch_start_ns) / 1000.0
        alignment = (
            f"first_claim_after_dispatch_start_us={first:.1f} "
            f"last_epi_end_after_dispatch_start_us={first + window / 1000.0:.1f} "
        )
    print(
        "[fc-tile-timeline] "
        f"rank={rank} ctas={ctas} tiles={summary['tiles']} "
        f"{alignment}"
        f"fc1_tiles={summary['fc1_tiles']} fc2_tiles={summary['fc2_tiles']} "
        f"tiles_per_cta_min={summary['tiles_per_cta_min']} "
        f"tiles_per_cta_max={summary['tiles_per_cta_max']} "
        f"truncated_ctas={summary['truncated_ctas']} "
        f"window_us={window / 1000.0:.1f} "
        f"ready_wait_per_cta_us={summary['ready_wait_total_ns'] / ctas / 1000.0:.1f} "
        f"mainloop_per_cta_us={summary['mainloop_total_ns'] / ctas / 1000.0:.1f} "
        f"epilogue_per_cta_us={summary['epilogue_total_ns'] / ctas / 1000.0:.1f} "
        f"gap_per_cta_us={summary['gap_total_ns'] / ctas / 1000.0:.1f} "
        f"busy_fraction={summary['tile_span_total_ns'] / ctas / window:.3f} "
        f"fc1_ready_wait_per_cta_us={summary['fc1_ready_wait_total_ns'] / ctas / 1000.0:.1f} "
        f"fc2_ready_wait_per_cta_us={summary['fc2_ready_wait_total_ns'] / ctas / 1000.0:.1f} "
        f"fc1_mainloop_us[{_quantiles_us(summary['fc1_mainloop_ns'])}] "
        f"fc2_mainloop_us[{_quantiles_us(summary['fc2_mainloop_ns'])}] "
        f"fc1_tmab_wait_frac_p50={_median(summary['fc1_tmab_wait_frac']):.2f} "
        f"fc1_mma_wait_frac_p50={_median(summary['fc1_mma_wait_frac']):.2f} "
        f"fc2_tmab_wait_frac_p50={_median(summary['fc2_tmab_wait_frac']):.2f} "
        f"fc2_mma_wait_frac_p50={_median(summary['fc2_mma_wait_frac']):.2f} "
        + "".join(
            f"{phase}_mma_{name.replace('mma_', '').replace('_cyc', '')}_frac_p50="
            f"{_median(summary[f'{phase}_mma_part_frac'][name]):.2f} "
            for phase in ("fc1", "fc2")
            for name in ("mma_acquire_cyc", "mma_s2t_cyc", "mma_issue_cyc", "mma_release_cyc")
        )
        + f"fc1_mma_loop_cyc_p50={_median(summary['fc1_mma_loop_cyc']):.0f} "
        f"fc2_mma_loop_cyc_p50={_median(summary['fc2_mma_loop_cyc']):.0f} "
        f"fc1_last_epi_end_us={summary['fc1_last_epi_end_rel_ns'] / 1000.0:.1f} "
        f"fc2_first_claim_us={summary['fc2_first_claim_rel_ns'] / 1000.0:.1f} "
        f"fc2_first_ready_us={summary['fc2_first_ready_rel_ns'] / 1000.0:.1f} "
        "clock_domain=per_gpu_globaltimer",
        flush=True,
    )
    occupancy = occupancy_bins(timeline)
    print(
        "[fc-tile-occupancy] "
        f"rank={rank} bin_us={occupancy['bin_us']:.1f} "
        f"busy={occupancy['busy']} waiting={occupancy['waiting']}",
        flush=True,
    )
    if not dump:
        return
    tl = timeline.detach().to("cpu", torch.int64)
    origin = summary["origin_ns"]
    for cta in range(tl.shape[0]):
        for seq in range(tl.shape[1]):
            record = tl[cta, seq]
            claim = int(record[_F["claim_ns"]].item())
            if claim == 0:
                break

            def rel(name: str) -> int:
                value = int(record[_F[name]].item())
                return value - origin if value else -1

            print(
                "[fc-tile] "
                f"rank={rank} cta={cta} seq={seq} "
                f"ph={int(record[_F['phase']].item())} "
                f"e={int(record[_F['expert']].item())} "
                f"m={int(record[_F['tile_m']].item())} "
                f"n={int(record[_F['tile_n']].item())} "
                f"c={claim - origin} r={rel('ready_ns')} "
                f"me={rel('mma_end_ns')} ee={rel('epi_end_ns')} "
                f"bw={int(record[_F['tmab_wait_cyc']].item())} "
                f"bl={int(record[_F['tmab_loop_cyc']].item())} "
                f"mw={int(record[_F['mma_wait_cyc']].item())} "
                f"ml={int(record[_F['mma_loop_cyc']].item())} "
                f"ma={int(record[_F['mma_acquire_cyc']].item())} "
                f"ms={int(record[_F['mma_s2t_cyc']].item())} "
                f"mi={int(record[_F['mma_issue_cyc']].item())} "
                f"mr={int(record[_F['mma_release_cyc']].item())}",
                flush=True,
            )


__all__ = [
    "occupancy_bins",
    "report_fc_tile_timeline",
    "summarize_fc_tile_timeline",
]
