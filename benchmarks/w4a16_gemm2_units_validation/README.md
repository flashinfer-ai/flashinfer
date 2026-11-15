# W4A16 GEMM2 work units: validation harness and evidence

Out-of-tree harness and raw evidence for "perf(moe): group short-K W4A16 GEMM2 M tiles into work units".
This commit is reverted immediately on the PR branch; the PR's final diff does not include it.

- Control: upstream `67d996c0a454cf975e91fd97dcce457b4a54b202`. Candidate: `tested-candidate.patch`
  (sha256 in `candidate.patch.sha256`), committed unchanged as the PR's code commit.
- Runtime: 8x NVIDIA B300 SXM6 AC (SM103, 148 SMs), driver 595.91.07, `nvcr.io/nvidia/pytorch:26.05-py3`
  (`sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`), CUDA 13.2, PyTorch
  `2.12.0a0+5aff3928d8.nv26.05`, nvidia-cutlass-dsl 4.8.0, cupti-python 13.5.0. Package list: `environment/`.

| Path | Contents |
|---|---|
| `tools/` | Scripts as they ran (paths are the devbox layout). `README.md` there describes each one. `fc2_unit_sweep.py` later gained a one-line monkeypatch signature change for `_select_m_tiles_per_unit()` losing its argument; the `sweep/` rows were produced before it. |
| `sweep/` | Kernel-level GEMM2 sweep (forced unit sizes vs one-tile units): 1,770 timed rows and per-shape tables. Every output buffer is bit-identical to one-tile units. |
| `campaign/caac1/`, `campaign/caac2/` | PR 7290-style W4A16 matrix: per-phase raw CUPTI samples and tactics (`*.record.json`), benchmark CSV lines, summaries. `caac2` reruns K3 TP8 on the final source and repeats V3 TP8. `merged-tables.md` is what the PR body shows. |
| `codegen/` | Normalized PTX/SASS hashes and resources for upstream, the benchmarked snapshot and the final source. |
| `tests/` | pytest logs (`round1/`: first candidate; `final/`: final source) and the upstream MegaMoE run showing the two pre-existing failures. |
| `differential/` | Whole-MoE outputs at model dimensions, upstream vs candidate, 360 cases bit-identical (`diff-compare-final.log` is the final source). |

Campaign method: each job runs on one GPU as C1 / A-frozen / A-native / A-native / A-frozen / C2 (fresh process per
phase, unchanged `benchmarks/bench_moe_deepseek.py`, 10 warmups, 100 CUPTI cold-L2 CUDA-graph samples per cell, PDL on).
Speedup = sqrt(C1*C2) / sqrt(A1*A2) - 1 on per-phase medians; aggregates are geometric means over tokens >= 4.
