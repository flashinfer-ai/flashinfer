# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import flashinfer.prims_ts.moe.tools.sweep.report as report
from flashinfer.prims_ts.moe.tools.sweep.report import (
    capture_environment_provenance,
    estimate_weight_bandwidth_tb_s,
    make_summary,
    render_markdown,
    write_summary,
)


def test_summary_writes_json_and_markdown(tmp_path):
    fc1_dir = tmp_path / "fc1"
    fc1 = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={128: [{"tile_n": 128}, {"tile_n": 128}]},
        per_shape={},
        verified=True,
    )
    write_summary(fc1_dir, fc1)
    assert json.loads((fc1_dir / "summary.json").read_text())["fc"] == "fc1"
    assert (fc1_dir / "report.md").is_file()


def test_markdown_reports_menu_speedup_and_size_curve():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={128: [{"tile_n": 128}]},
        per_shape={
            "shape": {
                "best_existing_us": 10.0,
                "best_found_us": 4.0,
                "selected_us": 5.0,
                "menu_speedup_vs_incumbent": 2.0,
            }
        },
        verified=True,
        activation_params={
            "activation": "Situ",
            "bias": False,
            "alpha": 4.0,
            "beta": 25.0,
            "clamp": None,
        },
    )
    summary["portfolio"] = {
        "gain_vs_size": [
            {
                "entries": 1,
                "geomean_speedup_vs_incumbent": 2.0,
                "minimum_speedup_vs_incumbent": 2.0,
                "covered_shapes": 1,
            }
        ]
    }

    markdown = render_markdown(summary)

    assert "| `shape` | 10.000 | 4.000 | 2.500× | 5.000 | 2.000× |" in markdown
    assert (
        "- Activation parameters: `activation=Situ, bias=False, alpha=4.0, "
        "beta=25.0, clamp=None`"
    ) in markdown
    assert "## Menu size trade-off" in markdown
    assert "| 1 | `—` | 2.000× | 2.000× | 1 |" in markdown


def test_markdown_starts_with_finalization_and_reports_empty_menu_shapes():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={},
        per_shape={
            "T=64,E=96": {
                "best_existing_us": 20.0,
                "best_found_us": 16.0,
                "selected_us": None,
                "active_experts": 96,
                "expert_weight_bytes": 1_000_000,
            }
        },
        verified=False,
    )
    summary["finalization"] = {
        "complete": False,
        "successfully_retimed": 5,
        "retime_pool_size": 7,
        "reference_errors": 1,
        "verified_count": 2,
        "unverified_count": 2,
        "failed_count": 1,
    }

    markdown = render_markdown(summary)

    assert markdown.index("**Finalization: FAILED**") < markdown.index("- Model:")
    assert "verified: 2; unverified: 2; failed: 1" in markdown
    assert "re-timed: 5/7; reference errors: 1" in markdown
    assert "| `T=64,E=96` | 20.000 | 16.000 | 1.250× | — | — | 6.000 |" in markdown
    assert estimate_weight_bandwidth_tb_s(96, 1_000_000, 16) == 6.0


def test_markdown_distinguishes_finalization_labels():
    def status(*, menu, verified, complete, unverified=0, failed=0):
        summary = make_summary(
            fc="fc1",
            model="deepseek-v4-pro-nvfp4",
            sm_arch=107,
            selected_by_tile_n=menu,
            per_shape={},
            verified=verified,
        )
        summary["finalization"] = {
            "complete": complete,
            "verified_count": int(verified),
            "unverified_count": unverified,
            "failed_count": failed,
        }
        return render_markdown(summary).splitlines()[2]

    assert "Finalization: VERIFIED" in status(
        menu={64: [{"tile_n": 64}]},
        verified=True,
        complete=True,
    )
    assert "Finalization: UNVERIFIED" in status(
        menu={64: [{"tile_n": 64}]},
        verified=False,
        complete=False,
        unverified=1,
    )
    assert "Finalization: FAILED" in status(
        menu={64: []},
        verified=False,
        complete=False,
        failed=1,
    )
    assert "Finalization: INCOMPLETE" in status(
        menu={},
        verified=False,
        complete=False,
        unverified=1,
    )


def test_markdown_adds_per_routing_columns_when_present():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={},
        per_shape={
            "shape": {
                "best_existing_us": 10.0,
                "best_found_us": 8.0,
                "per_routing": {
                    "uniform": {
                        "best_existing_us": 10.0,
                        "best_found_us": 8.0,
                        "menu_speedup_vs_pre_sweep": 1.2,
                        "estimated_weight_bandwidth_tb_s": 7.5,
                    },
                    "zipf:0.8": {
                        "best_existing_us": 9.0,
                        "best_found_us": 7.5,
                    },
                },
            }
        },
        verified=False,
    )

    markdown = render_markdown(summary)

    assert "uniform best / best speedup / menu speedup / BW" in markdown
    assert "zipf:0.8 best / best speedup / menu speedup / BW" in markdown
    assert "8.000 / 1.250× / 1.200× / 7.500" in markdown
    assert "7.500 / 1.200× / — / —" in markdown


def test_markdown_renders_structured_failure_table():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={},
        per_shape={},
        verified=False,
        correctness_failures=[
            {
                "candidate_id": "candidate-a",
                "candidate": {"tile_n": 64},
                "move_kind": "stages_all",
                "result": {
                    "status": "failed",
                    "checks": [
                        {
                            "shape_id": "shape-a",
                            "passed": False,
                            "failure_reason": "output mismatch",
                            "max_abs_error": 0.125,
                            "max_rel_error": 0.25,
                        }
                    ],
                },
            }
        ],
    )
    summary["finalization"] = {
        "complete": False,
        "verification_errors": [
            {
                "candidate_id": "candidate-b",
                "tile_n": 128,
                "status": "oom",
                "error": "out of memory",
            }
        ],
    }

    markdown = render_markdown(summary)

    assert "| Candidate | tile_n | Move | Shape | Reason | Max error |" in markdown
    assert (
        "| `candidate-a` | 64 | stages_all | shape-a | output mismatch | "
        "abs=0.125, rel=0.250 |"
    ) in markdown
    assert "| `candidate-b` | 128 | — | — | out of memory | — |" in markdown
    assert "{'candidate_id':" not in markdown


def test_markdown_reports_selected_reference_consistency_deviations():
    summary = make_summary(
        fc="fc2",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={32: [{"tile_n": 32}]},
        per_shape={},
        verified=True,
    )
    summary["reference_consistency"] = {
        "flagged_menu_rows": [
            {
                "candidate_id": "candidate-a",
                "shape_id": "shape-a",
                "mean_abs_ratio": 1.5,
                "max_abs_ratio": 1.25,
                "max_rel_matches": False,
            }
        ]
    }

    markdown = render_markdown(summary)

    assert "## Reference consistency deviations" in markdown
    assert "| `candidate-a` | `shape-a` | 1.500× | 1.250× | False |" in markdown


def test_markdown_renders_search_and_statistics_tables():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={},
        per_shape={},
        verified=False,
        move_statistics={
            "64": {
                "stages_one": {
                    "proposed": 12,
                    "attempts": 10,
                    "successes": 3,
                    "failures": 7,
                    "duplicates": 1,
                    "rejected": 1,
                    "evaluation_yield": 0.8,
                    "rejection_reasons": {"duplicate_signature": 1},
                }
            }
        },
    )
    summary["search_failures"] = [
        {
            "phase": "search",
            "status": "worker_error",
            "failure": "worker_crash",
            "exit_code": -6,
            "tile_n": 256,
            "move_kind": "skeleton",
            "count": 76,
            "example_candidate_ids": ["abcdef1234567890"],
            "error": "CUDA invalid argument",
        }
    ]
    summary["feature_effects"] = {
        "multicast": {
            "64": {
                "paired_measurements": 10.0,
                "feature_on_wins": 7.0,
                "feature_off_wins": 3.0,
                "geomean_on_over_off": 0.98,
            }
        }
    }
    summary["lineage_contributions"] = {
        "incumbent": {"selected_entries": 1},
        "crossover_lineage": {"selected_entries": 2},
    }
    summary["group_dispatch_stats"] = {
        "64": {
            "draws": 100,
            "proposals": 20,
            "drops_by_reason": {"no_move": 80},
            "starvation_warnings": 1,
        }
    }

    markdown = render_markdown(summary)

    assert "search failures: 76" in markdown
    assert (
        "| search | worker_error | worker_crash | -6 | 256 | skeleton | 76 |"
        in markdown
    )
    assert "| 64 | `stages_one` | 12 | 10 | 3 | 7 | 1 | 1 | 0.800 |" in markdown
    assert "| `multicast` | 64 | 10 | 7 | 3 | 0.980× |" in markdown
    assert "Rows overlap. Lineage rows include the selected entry" in markdown
    assert "| `incumbent (own origin)` | 1 |" in markdown
    assert "| `crossover in lineage` | 2 |" in markdown
    assert "Share" not in markdown
    assert "| 64 | 100 | 20 | 80 | no_move (80) | 1 |" in markdown
    assert "{'proposed':" not in markdown


def test_markdown_reports_adaptive_worker_utilization():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={},
        per_shape={},
        verified=False,
    )
    summary["worker_utilization"] = 0.4375
    summary["search_loop_profile"] = {
        "worker_busy_seconds": 700.0,
        "worker_capacity_seconds": 1600.0,
    }

    markdown = render_markdown(summary)

    assert (
        "- Adaptive-search worker utilization: `43.8%` "
        "(700.0 worker-s / 1600.0 capacity-s)"
    ) in markdown


def test_markdown_reports_gpu_clock_summary_and_warning():
    summary = make_summary(
        fc="fc1",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        selected_by_tile_n={},
        per_shape={},
        verified=False,
    )
    summary["gpu_clocks"] = {
        "search": {
            "samples": 20,
            "min_sm_clock_mhz": 2100,
            "median_sm_clock_mhz": 2100,
            "max_sm_clock_mhz": 2424,
            "max_rated_sm_clock_mhz": 2424,
            "low_clock_fraction": 0.5,
            "clock_event_reasons": {"nvmlClocksEventReasonSwPowerCap": 10},
        }
    }

    markdown = render_markdown(summary)

    assert "sustained low GPU SM clocks" in markdown
    assert "## GPU clocks" in markdown
    assert "| search | 20 | 2100 | 2100 | 2424 | 2424 | 50.0% |" in markdown


def test_environment_provenance_reports_unknown_git_state_on_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        report,
        "git_output",
        lambda *args, **_kwargs: "deadbeef" if "rev-parse" in args else None,
    )

    provenance = capture_environment_provenance(tmp_path)

    assert provenance["git_commit"] == "deadbeef"
    assert provenance["git_dirty"] is None
    assert provenance["git_dirty_files"] == -1


def test_environment_provenance_preserves_complete_porcelain_path(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        report,
        "git_output",
        lambda *args, **_kwargs: (
            " M benchmarks/x.py" if "status" in args else "deadbeef"
        ),
    )
    monkeypatch.setattr(report, "nvml_driver_version", lambda: None)

    provenance = capture_environment_provenance(tmp_path)

    assert provenance["git_dirty"] is True
    assert provenance["git_dirty_files"] == ["benchmarks/x.py"]
