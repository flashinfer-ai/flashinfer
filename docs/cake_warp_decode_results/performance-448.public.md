# NVFP4 warp-decode comparisons (FINAL)

Qualified new rows: 448/448. All seven geometries use all local experts at T1–32.

Times are CUPTI GPU microseconds with cold L2, symmetric external CUDA graphs and same-arm preconditioning; 3 counterbalanced groups, 50 ms warmup and 100 ms measurement per arm/group. Source/export must be at least 0.97; direction and endpoint drift limits are 0.02. Official/export below 1 remains a valid reported comparison.

Geometric means include every identity-matched, correct, timed row, including performance gate failures. Missing and mismatched rows do not enter means. Aggregates are separated by campaign, exact hardware cohort and source cohort; each cohort covers only its explicitly assigned rows. Different GPU UUIDs always form different hardware cohorts, even when their model names match. Native full-denominator completion is separate from this new-row report. Campaigns retained only as failed evidence own zero report rows; their measured phases and physical turnaround remain in the execution totals.

| Campaign | Architecture | Hardware cohort | GPU | Run source cohort | Measurement source cohorts | Assigned rows | Native qualified rows | Segment status |
|---|---|---|---|---|---|---:|---:|---|
| sm_100a_c01 | sm_100a | sm_100a_gpu01 | NVIDIA B200 | sm_100a_source01 | sm_100a_source01 | 2 | 2 | comparison_segment_complete |
| sm_100a_c02 | sm_100a | sm_100a_gpu02 | NVIDIA B200 | sm_100a_source02 | sm_100a_source02 | 62 | 62 | comparison_segment_complete |
| sm_100a_c03 | sm_100a | sm_100a_gpu03 | NVIDIA B200 | sm_100a_source02 | sm_100a_source02 | 96 | 96 | comparison_segment_complete |
| sm_100a_c04 | sm_100a | sm_100a_gpu04 | NVIDIA B200 | sm_100a_source02 |  | 0 | 0 | failed |
| sm_100a_c05 | sm_100a | sm_100a_gpu05 | NVIDIA B200 | sm_100a_source02 | sm_100a_source02 | 64 | 64 | comparison_segment_complete |
| sm_103a_c01 | sm_103a | sm_103a_gpu01 | NVIDIA B300 SXM6 AC | sm_103a_source01 | sm_103a_source01 | 13 | 13 | comparison_segment_complete |
| sm_103a_c02 | sm_103a | sm_103a_gpu02 | NVIDIA B300 SXM6 AC | sm_103a_source02 | sm_103a_source02 | 147 | 147 | comparison_segment_complete |
| sm_103a_c03 | sm_103a | sm_103a_gpu02 | NVIDIA B300 SXM6 AC | sm_103a_source03 | sm_103a_source03 | 64 | 211 | comparison_segment_complete |

| Architecture | Model | Campaign | Hardware cohort | Source cohort | Assigned | Passed | Timed | Source µs | Export µs | Source/export | Official µs | Official-paired export µs | Official/export |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sm_100a | Qwen3-30B-A3B | sm_100a_c01 | sm_100a_gpu01 | sm_100a_source01 | 2 | 2 | 2 | 18.514427 | 18.542689 | 0.998476 | 19.748434 | 18.542689 | 1.065025 |
| sm_100a | Qwen3-30B-A3B | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | 30 | 30 | 30 | 57.475428 | 57.452938 | 1.000391 | 49.671065 | 57.452938 | 0.864552 |
| sm_100a | Qwen3-235B-A22B | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | 32 | 32 | 32 | 169.803911 | 169.754256 | 1.000293 | 149.786334 | 169.754256 | 0.882372 |
| sm_100a | Qwen3.5-35B-A3B | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | 32 | 32 | 32 | 44.811653 | 44.869643 | 0.998708 | 44.383966 | 44.869643 | 0.989176 |
| sm_100a | Qwen3.5-397B-A17B | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | 32 | 32 | 32 | 155.653029 | 155.450290 | 1.001304 | 142.750874 | 155.450290 | 0.918306 |
| sm_100a | MiniMax-M2 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | 32 | 32 | 32 | 139.394385 | 139.493009 | 0.999293 | 147.574405 | 139.493009 | 1.057934 |
| sm_100a | MiniMax-M3 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | 32 | 32 | 32 | 254.608894 | 254.611875 | 0.999988 | 208.829189 | 254.611875 | 0.820186 |
| sm_100a | Kimi-K3 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | 32 | 32 | 32 | 606.174375 | 606.202725 | 0.999953 | 1157.617797 | 606.202725 | 1.909622 |
| sm_103a | Qwen3-30B-A3B | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | 13 | 13 | 13 | 33.800885 | 33.758914 | 1.001243 | 33.480108 | 33.758914 | 0.991741 |
| sm_103a | Qwen3-30B-A3B | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | 19 | 19 | 19 | 71.353384 | 71.400211 | 0.999344 | 58.055742 | 71.400211 | 0.813103 |
| sm_103a | Qwen3-235B-A22B | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | 32 | 32 | 32 | 171.535674 | 171.457261 | 1.000457 | 147.866751 | 171.457261 | 0.862412 |
| sm_103a | Qwen3.5-35B-A3B | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | 32 | 32 | 32 | 44.057335 | 44.097715 | 0.999084 | 43.639331 | 44.097715 | 0.989605 |
| sm_103a | Qwen3.5-397B-A17B | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | 32 | 32 | 32 | 156.624818 | 156.516223 | 1.000694 | 143.174785 | 156.516223 | 0.914760 |
| sm_103a | MiniMax-M2 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | 32 | 32 | 32 | 140.567856 | 140.531180 | 1.000261 | 145.715086 | 140.531180 | 1.036888 |
| sm_103a | MiniMax-M3 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | 32 | 32 | 32 | 258.060569 | 257.957204 | 1.000401 | 211.124536 | 257.957204 | 0.818448 |
| sm_103a | Kimi-K3 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | 32 | 32 | 32 | 623.245929 | 623.758192 | 0.999179 | 1118.473802 | 623.758192 | 1.793121 |

Native CLI states: sm_100a_c01: paused, full 352-row completion=False; sm_100a_c02: paused, full 352-row completion=False; sm_100a_c03: paused, full 352-row completion=False; sm_100a_c04: paused, full 352-row completion=False; sm_100a_c05: paused, full 352-row completion=False; sm_103a_c01: paused, full 352-row completion=False; sm_103a_c02: failed, full 352-row completion=False; sm_103a_c03: paused, full 352-row completion=False

| Architecture | Native measure-shape phases s | Native invocation s | Validation phase attempts s | Performance phase attempts s | Separate validation s | Managed client elapsed sum s | Managed physical sum s | Managed span s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sm_100a | 19708.443539 | ≥ 21018.140215 | 2068.853903 | 19856.955669 | 1176.259910 | 27346.944687 | 27349.859662 | 72591.887083 |
| sm_103a | 16263.715108 | ≥ 18454.049072 | 1795.732549 | 18332.749498 | 545.660777 | 27146.359684 | 27150.105865 | 40318.343757 |

Measure-shape phases include setup and correctness. These nested durations are not additive. Phase sums cover recorded phase durations; unfinished or unrecorded work may be absent. A native invocation total marked ≥ includes nonterminal progress receipts and is an observed lower bound; the unrecorded remainder is unknown. Managed client elapsed totals retain each receipt's elapsed_seconds. Physical totals sum each explicitly supplied step's completed_at minus created_at, including failures, and are unavailable if any required timestamp is invalid or absent. These are client-observed turnarounds, not GPU kernel latency or an inferred orphan scheduler lifetime. Managed span runs from the earliest listed step creation to the latest completion and includes intervening idle/wait time.

| Architecture | Model | T | Campaign | Hardware cohort | Source cohort | Status | Source µs | Export µs | Source/export | Official µs | Official-paired export µs | Official/export |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| sm_100a | Qwen3-30B-A3B | 1 | sm_100a_c01 | sm_100a_gpu01 | sm_100a_source01 | PASS | 16.480000 | 16.607000 | 0.992353 | 17.536000 | 16.607000 | 1.055940 |
| sm_100a | Qwen3-30B-A3B | 2 | sm_100a_c01 | sm_100a_gpu01 | sm_100a_source01 | PASS | 20.800000 | 20.704000 | 1.004637 | 22.240000 | 20.704000 | 1.074189 |
| sm_100a | Qwen3-30B-A3B | 3 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 24.832000 | 24.736000 | 1.003881 | 26.144000 | 24.736000 | 1.056921 |
| sm_100a | Qwen3-30B-A3B | 4 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 28.320000 | 28.576000 | 0.991041 | 29.409000 | 28.576000 | 1.029150 |
| sm_100a | Qwen3-30B-A3B | 5 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 31.584000 | 31.360000 | 1.007143 | 31.232000 | 31.360000 | 0.995918 |
| sm_100a | Qwen3-30B-A3B | 6 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 34.144000 | 34.528000 | 0.988879 | 34.400000 | 34.528000 | 0.996293 |
| sm_100a | Qwen3-30B-A3B | 7 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 38.080000 | 37.600000 | 1.012766 | 37.600000 | 37.600000 | 1.000000 |
| sm_100a | Qwen3-30B-A3B | 8 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 40.896000 | 40.512000 | 1.009479 | 39.488000 | 40.512000 | 0.974724 |
| sm_100a | Qwen3-30B-A3B | 9 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 43.008000 | 43.456000 | 0.989691 | 41.728000 | 43.456000 | 0.960236 |
| sm_100a | Qwen3-30B-A3B | 10 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 45.343000 | 44.992000 | 1.007801 | 43.296000 | 44.992000 | 0.962304 |
| sm_100a | Qwen3-30B-A3B | 11 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 47.776000 | 47.584000 | 1.004035 | 44.800000 | 47.584000 | 0.941493 |
| sm_100a | Qwen3-30B-A3B | 12 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 50.048000 | 50.079500 | 0.999371 | 45.984000 | 50.079500 | 0.918220 |
| sm_100a | Qwen3-30B-A3B | 13 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 52.287000 | 52.480000 | 0.996322 | 47.839000 | 52.480000 | 0.911566 |
| sm_100a | Qwen3-30B-A3B | 14 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 54.592000 | 54.720000 | 0.997661 | 48.511000 | 54.720000 | 0.886531 |
| sm_100a | Qwen3-30B-A3B | 15 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 56.799000 | 56.895000 | 0.998313 | 50.559000 | 56.895000 | 0.888637 |
| sm_100a | Qwen3-30B-A3B | 16 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 58.944000 | 58.879000 | 1.001104 | 51.488000 | 58.879000 | 0.874471 |
| sm_100a | Qwen3-30B-A3B | 17 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 61.439000 | 61.376000 | 1.001026 | 53.376000 | 61.376000 | 0.869656 |
| sm_100a | Qwen3-30B-A3B | 18 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 64.160000 | 64.064000 | 1.001499 | 54.847000 | 64.064000 | 0.856128 |
| sm_100a | Qwen3-30B-A3B | 19 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 66.271000 | 66.112000 | 1.002405 | 55.328000 | 66.112000 | 0.836883 |
| sm_100a | Qwen3-30B-A3B | 20 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 68.095000 | 67.839000 | 1.003774 | 55.808000 | 67.839000 | 0.822654 |
| sm_100a | Qwen3-30B-A3B | 21 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 69.664000 | 69.984000 | 0.995428 | 56.800000 | 69.984000 | 0.811614 |
| sm_100a | Qwen3-30B-A3B | 22 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 71.327000 | 71.232000 | 1.001334 | 58.272000 | 71.232000 | 0.818059 |
| sm_100a | Qwen3-30B-A3B | 23 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 73.215000 | 73.247000 | 0.999563 | 58.272000 | 73.247000 | 0.795555 |
| sm_100a | Qwen3-30B-A3B | 24 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 74.944000 | 74.944000 | 1.000000 | 58.784000 | 74.944000 | 0.784372 |
| sm_100a | Qwen3-30B-A3B | 25 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 76.864000 | 76.895000 | 0.999597 | 60.544000 | 76.895000 | 0.787359 |
| sm_100a | Qwen3-30B-A3B | 26 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 78.719000 | 78.719000 | 1.000000 | 62.303000 | 78.719000 | 0.791461 |
| sm_100a | Qwen3-30B-A3B | 27 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 80.767000 | 80.768000 | 0.999988 | 62.751000 | 80.768000 | 0.776929 |
| sm_100a | Qwen3-30B-A3B | 28 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 82.271000 | 82.271000 | 1.000000 | 63.455000 | 82.271000 | 0.771292 |
| sm_100a | Qwen3-30B-A3B | 29 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 84.767000 | 84.703000 | 1.000756 | 64.416000 | 84.703000 | 0.760493 |
| sm_100a | Qwen3-30B-A3B | 30 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 86.656000 | 86.592000 | 1.000739 | 64.831000 | 86.592000 | 0.748695 |
| sm_100a | Qwen3-30B-A3B | 31 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 88.895000 | 89.087000 | 0.997845 | 65.664000 | 89.087000 | 0.737077 |
| sm_100a | Qwen3-30B-A3B | 32 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 90.847000 | 90.784000 | 1.000694 | 65.632000 | 90.784000 | 0.722947 |
| sm_100a | Qwen3-235B-A22B | 1 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 33.728000 | 33.664000 | 1.001901 | 35.552000 | 33.664000 | 1.056084 |
| sm_100a | Qwen3-235B-A22B | 2 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 43.871000 | 44.000000 | 0.997068 | 52.992000 | 44.000000 | 1.204364 |
| sm_100a | Qwen3-235B-A22B | 3 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 58.879000 | 58.751000 | 1.002179 | 68.224000 | 58.751000 | 1.161240 |
| sm_100a | Qwen3-235B-A22B | 4 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 69.248000 | 69.216000 | 1.000462 | 80.640000 | 69.216000 | 1.165049 |
| sm_100a | Qwen3-235B-A22B | 5 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 79.103000 | 78.912000 | 1.002420 | 91.327000 | 78.912000 | 1.157327 |
| sm_100a | Qwen3-235B-A22B | 6 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 89.023000 | 88.640000 | 1.004321 | 104.191000 | 88.640000 | 1.175440 |
| sm_100a | Qwen3-235B-A22B | 7 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 101.503000 | 101.440000 | 1.000621 | 118.559000 | 101.440000 | 1.168760 |
| sm_100a | Qwen3-235B-A22B | 8 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 112.671000 | 112.576000 | 1.000844 | 126.591000 | 112.576000 | 1.124494 |
| sm_100a | Qwen3-235B-A22B | 9 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 122.431000 | 122.336000 | 1.000777 | 134.336000 | 122.336000 | 1.098091 |
| sm_100a | Qwen3-235B-A22B | 10 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 132.415000 | 132.479000 | 0.999517 | 140.031000 | 132.479000 | 1.057005 |
| sm_100a | Qwen3-235B-A22B | 11 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 144.255000 | 144.159000 | 1.000666 | 145.855000 | 144.159000 | 1.011765 |
| sm_100a | Qwen3-235B-A22B | 12 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 153.984000 | 153.791000 | 1.001255 | 151.648000 | 153.791000 | 0.986066 |
| sm_100a | Qwen3-235B-A22B | 13 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 163.839000 | 164.158000 | 0.998057 | 159.422000 | 164.158000 | 0.971150 |
| sm_100a | Qwen3-235B-A22B | 14 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 175.391000 | 175.359000 | 1.000182 | 161.375000 | 175.359000 | 0.920255 |
| sm_100a | Qwen3-235B-A22B | 15 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 186.302000 | 186.175000 | 1.000682 | 168.608000 | 186.175000 | 0.905643 |
| sm_100a | Qwen3-235B-A22B | 16 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 195.455000 | 195.327000 | 1.000655 | 172.575000 | 195.327000 | 0.883518 |
| sm_100a | Qwen3-235B-A22B | 17 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 206.687000 | 206.847000 | 0.999226 | 178.975000 | 206.847000 | 0.865253 |
| sm_100a | Qwen3-235B-A22B | 18 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 217.726000 | 217.982000 | 0.998826 | 184.127000 | 217.982000 | 0.844689 |
| sm_100a | Qwen3-235B-A22B | 19 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 226.270500 | 225.983000 | 1.001272 | 185.951000 | 225.983000 | 0.822854 |
| sm_100a | Qwen3-235B-A22B | 20 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 235.038000 | 235.423000 | 0.998365 | 187.839000 | 235.423000 | 0.797879 |
| sm_100a | Qwen3-235B-A22B | 21 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 244.254000 | 244.223000 | 1.000127 | 191.743000 | 244.223000 | 0.785114 |
| sm_100a | Qwen3-235B-A22B | 22 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 254.814000 | 254.879000 | 0.999745 | 197.311000 | 254.879000 | 0.774136 |
| sm_100a | Qwen3-235B-A22B | 23 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 265.630000 | 265.502000 | 1.000482 | 197.375000 | 265.502000 | 0.743403 |
| sm_100a | Qwen3-235B-A22B | 24 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 275.294000 | 275.518000 | 0.999187 | 198.878000 | 275.518000 | 0.721833 |
| sm_100a | Qwen3-235B-A22B | 25 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 284.575000 | 284.382000 | 1.000679 | 206.238000 | 284.382000 | 0.725215 |
| sm_100a | Qwen3-235B-A22B | 26 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 295.102000 | 295.390000 | 0.999025 | 214.110000 | 295.390000 | 0.724838 |
| sm_100a | Qwen3-235B-A22B | 27 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 307.390000 | 307.006000 | 1.001251 | 215.807000 | 307.006000 | 0.702941 |
| sm_100a | Qwen3-235B-A22B | 28 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 318.494000 | 318.718000 | 0.999297 | 217.342500 | 318.718000 | 0.681927 |
| sm_100a | Qwen3-235B-A22B | 29 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 328.382000 | 328.029000 | 1.001076 | 220.991000 | 328.029000 | 0.673693 |
| sm_100a | Qwen3-235B-A22B | 30 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 338.494000 | 338.686000 | 0.999433 | 222.718000 | 338.686000 | 0.657594 |
| sm_100a | Qwen3-235B-A22B | 31 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 348.414000 | 348.766000 | 0.998991 | 226.398000 | 348.766000 | 0.649140 |
| sm_100a | Qwen3-235B-A22B | 32 | sm_100a_c02 | sm_100a_gpu02 | sm_100a_source02 | PASS | 359.390000 | 359.102000 | 1.000802 | 226.623000 | 359.102000 | 0.631083 |
| sm_100a | Qwen3.5-35B-A3B | 1 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 14.560000 | 14.687000 | 0.991353 | 16.031000 | 14.687000 | 1.091509 |
| sm_100a | Qwen3.5-35B-A3B | 2 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 17.632000 | 17.600000 | 1.001818 | 19.968000 | 17.600000 | 1.134545 |
| sm_100a | Qwen3.5-35B-A3B | 3 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 21.312000 | 21.216000 | 1.004525 | 23.136000 | 21.216000 | 1.090498 |
| sm_100a | Qwen3.5-35B-A3B | 4 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 24.032000 | 23.968000 | 1.002670 | 26.272000 | 23.968000 | 1.096128 |
| sm_100a | Qwen3.5-35B-A3B | 5 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 26.816000 | 27.007000 | 0.992928 | 28.287000 | 27.007000 | 1.047395 |
| sm_100a | Qwen3.5-35B-A3B | 6 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 29.439000 | 29.311000 | 1.004367 | 31.168000 | 29.311000 | 1.063355 |
| sm_100a | Qwen3.5-35B-A3B | 7 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 31.871000 | 31.744000 | 1.004001 | 33.856000 | 31.744000 | 1.066532 |
| sm_100a | Qwen3.5-35B-A3B | 8 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 34.592000 | 35.039000 | 0.987243 | 36.575000 | 35.039000 | 1.043837 |
| sm_100a | Qwen3.5-35B-A3B | 9 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 36.831000 | 36.959000 | 0.996537 | 38.240000 | 36.959000 | 1.034660 |
| sm_100a | Qwen3.5-35B-A3B | 10 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 38.496000 | 38.911000 | 0.989335 | 39.680000 | 38.911000 | 1.019763 |
| sm_100a | Qwen3.5-35B-A3B | 11 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 40.352000 | 40.511000 | 0.996075 | 41.695000 | 40.511000 | 1.029227 |
| sm_100a | Qwen3.5-35B-A3B | 12 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 42.047000 | 42.400000 | 0.991675 | 42.656000 | 42.400000 | 1.006038 |
| sm_100a | Qwen3.5-35B-A3B | 13 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 43.871000 | 43.776000 | 1.002170 | 44.672000 | 43.776000 | 1.020468 |
| sm_100a | Qwen3.5-35B-A3B | 14 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 45.471000 | 45.344000 | 1.002801 | 45.792000 | 45.344000 | 1.009880 |
| sm_100a | Qwen3.5-35B-A3B | 15 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 46.911000 | 47.168000 | 0.994551 | 47.072000 | 47.168000 | 0.997965 |
| sm_100a | Qwen3.5-35B-A3B | 16 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 48.959000 | 48.895000 | 1.001309 | 48.544000 | 48.895000 | 0.992821 |
| sm_100a | Qwen3.5-35B-A3B | 17 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 50.623000 | 50.784000 | 0.996830 | 50.239000 | 50.784000 | 0.989268 |
| sm_100a | Qwen3.5-35B-A3B | 18 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 52.383000 | 52.287000 | 1.001836 | 51.231000 | 52.287000 | 0.979804 |
| sm_100a | Qwen3.5-35B-A3B | 19 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 54.207000 | 54.271000 | 0.998821 | 52.351000 | 54.271000 | 0.964622 |
| sm_100a | Qwen3.5-35B-A3B | 20 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 55.808000 | 55.680000 | 1.002299 | 53.472000 | 55.680000 | 0.960345 |
| sm_100a | Qwen3.5-35B-A3B | 21 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 57.279000 | 57.184000 | 1.001661 | 54.432000 | 57.184000 | 0.951875 |
| sm_100a | Qwen3.5-35B-A3B | 22 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 58.783000 | 58.655000 | 1.002182 | 55.615000 | 58.655000 | 0.948172 |
| sm_100a | Qwen3.5-35B-A3B | 23 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 60.096000 | 60.159000 | 0.998953 | 56.608000 | 60.159000 | 0.940973 |
| sm_100a | Qwen3.5-35B-A3B | 24 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 61.663000 | 61.823000 | 0.997412 | 57.216000 | 61.823000 | 0.925481 |
| sm_100a | Qwen3.5-35B-A3B | 25 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 63.487000 | 63.583000 | 0.998490 | 58.752000 | 63.583000 | 0.924021 |
| sm_100a | Qwen3.5-35B-A3B | 26 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 65.056000 | 65.183000 | 0.998052 | 60.096000 | 65.183000 | 0.921958 |
| sm_100a | Qwen3.5-35B-A3B | 27 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 66.944000 | 66.943000 | 1.000015 | 61.087000 | 66.943000 | 0.912523 |
| sm_100a | Qwen3.5-35B-A3B | 28 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 69.055000 | 68.927000 | 1.001857 | 62.975000 | 68.927000 | 0.913648 |
| sm_100a | Qwen3.5-35B-A3B | 29 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 70.943000 | 70.879000 | 1.000903 | 64.863000 | 70.879000 | 0.915123 |
| sm_100a | Qwen3.5-35B-A3B | 30 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 72.576000 | 72.479000 | 1.001338 | 66.015000 | 72.479000 | 0.910816 |
| sm_100a | Qwen3.5-35B-A3B | 31 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 74.239000 | 74.271000 | 0.999569 | 67.871000 | 74.271000 | 0.913829 |
| sm_100a | Qwen3.5-35B-A3B | 32 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 75.871000 | 76.223000 | 0.995382 | 68.832000 | 76.223000 | 0.903035 |
| sm_100a | Qwen3.5-397B-A17B | 1 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 31.808000 | 31.680000 | 1.004040 | 29.024000 | 31.680000 | 0.916162 |
| sm_100a | Qwen3.5-397B-A17B | 2 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 40.576000 | 40.223000 | 1.008776 | 42.240000 | 40.223000 | 1.050145 |
| sm_100a | Qwen3.5-397B-A17B | 3 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 52.607000 | 51.935500 | 1.012929 | 54.815000 | 51.935500 | 1.055444 |
| sm_100a | Qwen3.5-397B-A17B | 4 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 64.287000 | 64.128000 | 1.002479 | 66.335000 | 64.128000 | 1.034416 |
| sm_100a | Qwen3.5-397B-A17B | 5 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 75.327000 | 75.199000 | 1.001702 | 77.439000 | 75.199000 | 1.029788 |
| sm_100a | Qwen3.5-397B-A17B | 6 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 86.271000 | 86.175000 | 1.001114 | 88.351000 | 86.175000 | 1.025251 |
| sm_100a | Qwen3.5-397B-A17B | 7 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 94.495000 | 94.623000 | 0.998647 | 95.103000 | 94.623000 | 1.005073 |
| sm_100a | Qwen3.5-397B-A17B | 8 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 103.583000 | 103.455000 | 1.001237 | 103.167000 | 103.455000 | 0.997216 |
| sm_100a | Qwen3.5-397B-A17B | 9 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 113.375000 | 112.991000 | 1.003399 | 112.366500 | 112.991000 | 0.994473 |
| sm_100a | Qwen3.5-397B-A17B | 10 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 123.231000 | 123.359000 | 0.998962 | 122.175000 | 123.359000 | 0.990402 |
| sm_100a | Qwen3.5-397B-A17B | 11 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 130.719000 | 131.166000 | 0.996592 | 128.126000 | 131.166000 | 0.976823 |
| sm_100a | Qwen3.5-397B-A17B | 12 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 139.935000 | 140.190000 | 0.998181 | 133.983000 | 140.190000 | 0.955724 |
| sm_100a | Qwen3.5-397B-A17B | 13 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 149.982000 | 150.046000 | 0.999573 | 142.014000 | 150.046000 | 0.946470 |
| sm_100a | Qwen3.5-397B-A17B | 14 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 159.199000 | 159.422000 | 0.998601 | 149.694000 | 159.422000 | 0.938980 |
| sm_100a | Qwen3.5-397B-A17B | 15 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 169.438000 | 168.894000 | 1.003221 | 158.782000 | 168.894000 | 0.940128 |
| sm_100a | Qwen3.5-397B-A17B | 16 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 178.046000 | 177.598000 | 1.002523 | 166.366000 | 177.598000 | 0.936756 |
| sm_100a | Qwen3.5-397B-A17B | 17 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 186.558000 | 185.886000 | 1.003615 | 173.214000 | 185.886000 | 0.931829 |
| sm_100a | Qwen3.5-397B-A17B | 18 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 196.286000 | 195.870000 | 1.002124 | 181.950000 | 195.870000 | 0.928932 |
| sm_100a | Qwen3.5-397B-A17B | 19 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 206.270000 | 205.597000 | 1.003273 | 189.534000 | 205.597000 | 0.921871 |
| sm_100a | Qwen3.5-397B-A17B | 20 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 215.517000 | 215.325500 | 1.000889 | 195.134000 | 215.325500 | 0.906228 |
| sm_100a | Qwen3.5-397B-A17B | 21 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 224.446000 | 224.061000 | 1.001718 | 200.958000 | 224.061000 | 0.896890 |
| sm_100a | Qwen3.5-397B-A17B | 22 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 233.086000 | 232.382000 | 1.003029 | 205.053500 | 232.382000 | 0.882398 |
| sm_100a | Qwen3.5-397B-A17B | 23 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 242.686000 | 242.749000 | 0.999740 | 210.013000 | 242.749000 | 0.865145 |
| sm_100a | Qwen3.5-397B-A17B | 24 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 250.685000 | 250.877000 | 0.999235 | 215.646000 | 250.877000 | 0.859569 |
| sm_100a | Qwen3.5-397B-A17B | 25 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 260.573000 | 260.862000 | 0.998892 | 221.470000 | 260.862000 | 0.848993 |
| sm_100a | Qwen3.5-397B-A17B | 26 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 269.181000 | 269.822000 | 0.997624 | 226.462000 | 269.822000 | 0.839301 |
| sm_100a | Qwen3.5-397B-A17B | 27 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 278.333000 | 278.557000 | 0.999196 | 231.390000 | 278.557000 | 0.830674 |
| sm_100a | Qwen3.5-397B-A17B | 28 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 288.509000 | 288.925000 | 0.998560 | 234.270000 | 288.925000 | 0.810833 |
| sm_100a | Qwen3.5-397B-A17B | 29 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 298.172500 | 297.949000 | 1.000750 | 241.917000 | 297.949000 | 0.811941 |
| sm_100a | Qwen3.5-397B-A17B | 30 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 306.461000 | 306.653000 | 0.999374 | 244.861000 | 306.653000 | 0.798495 |
| sm_100a | Qwen3.5-397B-A17B | 31 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 316.285000 | 316.477000 | 0.999393 | 250.558000 | 316.477000 | 0.791710 |
| sm_100a | Qwen3.5-397B-A17B | 32 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 326.957500 | 326.141000 | 1.002504 | 254.333000 | 326.141000 | 0.779825 |
| sm_100a | MiniMax-M2 | 1 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 29.696000 | 29.823000 | 0.995742 | 32.032000 | 29.823000 | 1.074070 |
| sm_100a | MiniMax-M2 | 2 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 36.032000 | 36.416000 | 0.989455 | 44.608000 | 36.416000 | 1.224956 |
| sm_100a | MiniMax-M2 | 3 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 47.615000 | 47.647000 | 0.999328 | 57.823000 | 47.647000 | 1.213571 |
| sm_100a | MiniMax-M2 | 4 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 59.167000 | 58.975000 | 1.003256 | 70.399000 | 58.975000 | 1.193709 |
| sm_100a | MiniMax-M2 | 5 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 68.063000 | 67.807000 | 1.003775 | 80.350000 | 67.807000 | 1.184981 |
| sm_100a | MiniMax-M2 | 6 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 77.983000 | 77.663000 | 1.004120 | 93.374000 | 77.663000 | 1.202297 |
| sm_100a | MiniMax-M2 | 7 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 87.327000 | 87.071000 | 1.002940 | 106.014000 | 87.071000 | 1.217558 |
| sm_100a | MiniMax-M2 | 8 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 96.542000 | 96.638000 | 0.999007 | 117.054000 | 96.638000 | 1.211263 |
| sm_100a | MiniMax-M2 | 9 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 105.279000 | 105.567000 | 0.997272 | 126.974000 | 105.567000 | 1.202781 |
| sm_100a | MiniMax-M2 | 10 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 111.742000 | 111.967000 | 0.997990 | 132.862000 | 111.967000 | 1.186617 |
| sm_100a | MiniMax-M2 | 11 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 118.591000 | 118.750000 | 0.998661 | 141.374000 | 118.750000 | 1.190518 |
| sm_100a | MiniMax-M2 | 12 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 126.911000 | 126.911000 | 1.000000 | 145.726000 | 126.911000 | 1.148254 |
| sm_100a | MiniMax-M2 | 13 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 134.463000 | 134.910000 | 0.996687 | 154.206000 | 134.910000 | 1.143029 |
| sm_100a | MiniMax-M2 | 14 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 141.855000 | 142.366000 | 0.996411 | 158.814000 | 142.366000 | 1.115533 |
| sm_100a | MiniMax-M2 | 15 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 150.462000 | 150.702500 | 0.998404 | 164.286000 | 150.702500 | 1.090135 |
| sm_100a | MiniMax-M2 | 16 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 159.135000 | 158.846000 | 1.001819 | 170.334000 | 158.846000 | 1.072322 |
| sm_100a | MiniMax-M2 | 17 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 167.711000 | 167.486000 | 1.001343 | 177.022000 | 167.486000 | 1.056936 |
| sm_100a | MiniMax-M2 | 18 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 174.382000 | 174.270000 | 1.000643 | 181.054000 | 174.270000 | 1.038928 |
| sm_100a | MiniMax-M2 | 19 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 181.982000 | 182.334000 | 0.998069 | 186.654000 | 182.334000 | 1.023693 |
| sm_100a | MiniMax-M2 | 20 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 189.598000 | 189.438500 | 1.000842 | 192.062000 | 189.438500 | 1.013849 |
| sm_100a | MiniMax-M2 | 21 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 197.214000 | 197.342000 | 0.999351 | 196.270000 | 197.342000 | 0.994568 |
| sm_100a | MiniMax-M2 | 22 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 205.182000 | 205.150000 | 1.000156 | 201.534000 | 205.150000 | 0.982374 |
| sm_100a | MiniMax-M2 | 23 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 212.989000 | 213.085000 | 0.999549 | 205.662000 | 213.085000 | 0.965164 |
| sm_100a | MiniMax-M2 | 24 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 221.229500 | 220.958000 | 1.001229 | 208.414000 | 220.958000 | 0.943229 |
| sm_100a | MiniMax-M2 | 25 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 229.597000 | 229.469000 | 1.000558 | 215.517000 | 229.469000 | 0.939199 |
| sm_100a | MiniMax-M2 | 26 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 237.630000 | 238.013000 | 0.998391 | 222.142000 | 238.013000 | 0.933319 |
| sm_100a | MiniMax-M2 | 27 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 246.270000 | 246.750000 | 0.998055 | 226.237500 | 246.750000 | 0.916869 |
| sm_100a | MiniMax-M2 | 28 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 254.941000 | 255.453000 | 0.997996 | 234.302000 | 255.453000 | 0.917202 |
| sm_100a | MiniMax-M2 | 29 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 263.645000 | 263.805000 | 0.999393 | 241.502000 | 263.805000 | 0.915456 |
| sm_100a | MiniMax-M2 | 30 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 271.934000 | 272.157000 | 0.999181 | 248.286000 | 272.157000 | 0.912290 |
| sm_100a | MiniMax-M2 | 31 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 279.357000 | 279.517000 | 0.999428 | 256.733000 | 279.517000 | 0.918488 |
| sm_100a | MiniMax-M2 | 32 | sm_100a_c03 | sm_100a_gpu03 | sm_100a_source02 | PASS | 286.781000 | 287.229000 | 0.998440 | 260.861000 | 287.229000 | 0.908199 |
| sm_100a | MiniMax-M3 | 1 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 42.975000 | 43.200000 | 0.994792 | 38.815000 | 43.200000 | 0.898495 |
| sm_100a | MiniMax-M3 | 2 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 57.886000 | 57.791000 | 1.001644 | 59.775000 | 57.791000 | 1.034331 |
| sm_100a | MiniMax-M3 | 3 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 78.367000 | 78.046000 | 1.004113 | 80.894000 | 78.046000 | 1.036491 |
| sm_100a | MiniMax-M3 | 4 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 97.279000 | 97.055000 | 1.002308 | 99.167000 | 97.055000 | 1.021761 |
| sm_100a | MiniMax-M3 | 5 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 110.687000 | 110.398000 | 1.002618 | 109.054000 | 110.398000 | 0.987826 |
| sm_100a | MiniMax-M3 | 6 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 128.542000 | 128.447000 | 1.000740 | 126.526000 | 128.447000 | 0.985044 |
| sm_100a | MiniMax-M3 | 7 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 146.462000 | 146.270000 | 1.001313 | 143.742000 | 146.270000 | 0.982717 |
| sm_100a | MiniMax-M3 | 8 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 163.742000 | 163.550000 | 1.001174 | 156.766000 | 163.550000 | 0.958520 |
| sm_100a | MiniMax-M3 | 9 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 181.855000 | 181.790000 | 1.000358 | 174.911000 | 181.790000 | 0.962160 |
| sm_100a | MiniMax-M3 | 10 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 194.398000 | 194.142000 | 1.001319 | 179.325000 | 194.142000 | 0.923680 |
| sm_100a | MiniMax-M3 | 11 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 211.550000 | 212.158000 | 0.997134 | 188.030000 | 212.158000 | 0.886273 |
| sm_100a | MiniMax-M3 | 12 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 229.341000 | 229.789000 | 0.998050 | 196.510000 | 229.789000 | 0.855176 |
| sm_100a | MiniMax-M3 | 13 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 246.813000 | 246.685000 | 1.000519 | 214.014000 | 246.685000 | 0.867560 |
| sm_100a | MiniMax-M3 | 14 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 264.029000 | 264.285000 | 0.999031 | 214.206000 | 264.285000 | 0.810511 |
| sm_100a | MiniMax-M3 | 15 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 281.309000 | 281.405000 | 0.999659 | 231.581500 | 281.405000 | 0.822947 |
| sm_100a | MiniMax-M3 | 16 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 298.973500 | 299.293000 | 0.998932 | 240.286000 | 299.293000 | 0.802845 |
| sm_100a | MiniMax-M3 | 17 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 316.668000 | 316.428500 | 1.000757 | 253.085000 | 316.428500 | 0.799817 |
| sm_100a | MiniMax-M3 | 18 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 333.757000 | 334.252500 | 0.998518 | 266.077000 | 334.252500 | 0.796036 |
| sm_100a | MiniMax-M3 | 19 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 348.188000 | 348.220000 | 0.999908 | 274.557000 | 348.220000 | 0.788458 |
| sm_100a | MiniMax-M3 | 20 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 362.620000 | 363.356000 | 0.997974 | 274.462000 | 363.356000 | 0.755353 |
| sm_100a | MiniMax-M3 | 21 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 379.868000 | 380.124000 | 0.999327 | 287.612000 | 380.124000 | 0.756627 |
| sm_100a | MiniMax-M3 | 22 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 396.700000 | 396.540000 | 1.000403 | 296.189000 | 396.540000 | 0.746933 |
| sm_100a | MiniMax-M3 | 23 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 414.299500 | 414.363000 | 0.999847 | 300.621000 | 414.363000 | 0.725502 |
| sm_100a | MiniMax-M3 | 24 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 431.739000 | 432.059000 | 0.999259 | 309.053500 | 432.059000 | 0.715304 |
| sm_100a | MiniMax-M3 | 25 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 449.179000 | 449.403500 | 0.999500 | 326.364000 | 449.403500 | 0.726216 |
| sm_100a | MiniMax-M3 | 26 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 466.331000 | 466.395000 | 0.999863 | 334.844500 | 466.395000 | 0.717942 |
| sm_100a | MiniMax-M3 | 27 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 483.611500 | 483.707000 | 0.999803 | 339.387500 | 483.707000 | 0.701639 |
| sm_100a | MiniMax-M3 | 28 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 501.018000 | 500.922000 | 1.000192 | 348.029000 | 500.922000 | 0.694777 |
| sm_100a | MiniMax-M3 | 29 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 518.346500 | 518.202000 | 1.000279 | 356.636000 | 518.202000 | 0.688218 |
| sm_100a | MiniMax-M3 | 30 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 535.626500 | 535.450000 | 1.000330 | 369.468000 | 535.450000 | 0.690014 |
| sm_100a | MiniMax-M3 | 31 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 552.985000 | 552.826000 | 1.000288 | 382.587000 | 552.826000 | 0.692057 |
| sm_100a | MiniMax-M3 | 32 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 570.490000 | 570.650000 | 0.999720 | 382.716000 | 570.650000 | 0.670667 |
| sm_100a | Kimi-K3 | 1 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 77.727000 | 77.343000 | 1.004965 | 125.374000 | 77.343000 | 1.621013 |
| sm_100a | Kimi-K3 | 2 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 121.118000 | 121.726000 | 0.995005 | 220.125000 | 121.726000 | 1.808365 |
| sm_100a | Kimi-K3 | 3 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 165.662000 | 165.726000 | 0.999614 | 321.436000 | 165.726000 | 1.939563 |
| sm_100a | Kimi-K3 | 4 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 209.149000 | 209.533000 | 0.998167 | 416.635000 | 209.533000 | 1.988398 |
| sm_100a | Kimi-K3 | 5 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 252.844500 | 252.925000 | 0.999682 | 502.905000 | 252.925000 | 1.988356 |
| sm_100a | Kimi-K3 | 6 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 296.317000 | 295.836000 | 1.001626 | 599.416000 | 295.836000 | 2.026177 |
| sm_100a | Kimi-K3 | 7 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 339.868000 | 339.516000 | 1.001037 | 692.055000 | 339.516000 | 2.038358 |
| sm_100a | Kimi-K3 | 8 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 382.523000 | 382.171000 | 1.000921 | 781.750000 | 382.171000 | 2.045550 |
| sm_100a | Kimi-K3 | 9 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 425.339000 | 425.498500 | 0.999625 | 869.685000 | 425.498500 | 2.043920 |
| sm_100a | Kimi-K3 | 10 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 468.442500 | 468.730000 | 0.999387 | 957.461000 | 468.730000 | 2.042671 |
| sm_100a | Kimi-K3 | 11 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 512.058000 | 512.282500 | 0.999562 | 1053.028000 | 512.282500 | 2.055561 |
| sm_100a | Kimi-K3 | 12 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 555.738000 | 556.185000 | 0.999196 | 1146.643000 | 556.185000 | 2.061622 |
| sm_100a | Kimi-K3 | 13 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 598.681000 | 599.193000 | 0.999146 | 1214.866500 | 599.193000 | 2.027504 |
| sm_100a | Kimi-K3 | 14 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 642.041000 | 642.617000 | 0.999104 | 1308.625000 | 642.617000 | 2.036400 |
| sm_100a | Kimi-K3 | 15 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 685.096500 | 685.720000 | 0.999091 | 1397.248500 | 685.720000 | 2.037637 |
| sm_100a | Kimi-K3 | 16 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 728.312000 | 728.184000 | 1.000176 | 1477.967500 | 728.184000 | 2.029662 |
| sm_100a | Kimi-K3 | 17 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 771.447000 | 771.288000 | 1.000206 | 1541.039000 | 771.288000 | 1.998007 |
| sm_100a | Kimi-K3 | 18 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 814.455000 | 814.487000 | 0.999961 | 1597.310000 | 814.487000 | 1.961124 |
| sm_100a | Kimi-K3 | 19 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 857.527000 | 857.367000 | 1.000187 | 1659.694000 | 857.367000 | 1.935803 |
| sm_100a | Kimi-K3 | 20 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 900.790000 | 900.662000 | 1.000142 | 1715.693000 | 900.662000 | 1.904924 |
| sm_100a | Kimi-K3 | 21 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 943.846000 | 943.862000 | 0.999983 | 1778.876500 | 943.862000 | 1.884679 |
| sm_100a | Kimi-K3 | 22 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 986.997000 | 987.173500 | 0.999821 | 1853.132000 | 987.173500 | 1.877210 |
| sm_100a | Kimi-K3 | 23 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1029.141000 | 1028.404000 | 1.000717 | 1929.002500 | 1028.404000 | 1.875724 |
| sm_100a | Kimi-K3 | 24 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1071.956500 | 1071.508000 | 1.000419 | 1992.507000 | 1071.508000 | 1.859535 |
| sm_100a | Kimi-K3 | 25 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1114.371500 | 1114.196000 | 1.000158 | 2022.266000 | 1114.196000 | 1.815000 |
| sm_100a | Kimi-K3 | 26 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1157.827500 | 1157.123500 | 1.000608 | 2097.065500 | 1157.123500 | 1.812309 |
| sm_100a | Kimi-K3 | 27 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1201.299500 | 1200.691000 | 1.000507 | 2154.137000 | 1200.691000 | 1.794081 |
| sm_100a | Kimi-K3 | 28 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1243.443000 | 1243.186000 | 1.000207 | 2216.712000 | 1243.186000 | 1.783090 |
| sm_100a | Kimi-K3 | 29 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1286.098000 | 1286.594500 | 0.999614 | 2268.279500 | 1286.594500 | 1.763010 |
| sm_100a | Kimi-K3 | 30 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1329.746000 | 1329.522000 | 1.000168 | 2319.319000 | 1329.522000 | 1.744476 |
| sm_100a | Kimi-K3 | 31 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1373.361000 | 1373.393500 | 0.999976 | 2363.190500 | 1373.393500 | 1.720694 |
| sm_100a | Kimi-K3 | 32 | sm_100a_c05 | sm_100a_gpu05 | sm_100a_source02 | PASS | 1415.744500 | 1416.368500 | 0.999559 | 2431.414000 | 1416.368500 | 1.716654 |
| sm_103a | Qwen3-30B-A3B | 1 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 15.808000 | 15.904000 | 0.993964 | 16.800000 | 15.904000 | 1.056338 |
| sm_103a | Qwen3-30B-A3B | 2 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 20.160000 | 19.872000 | 1.014493 | 21.536000 | 19.872000 | 1.083736 |
| sm_103a | Qwen3-30B-A3B | 3 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 23.969000 | 24.224000 | 0.989473 | 25.216000 | 24.224000 | 1.040951 |
| sm_103a | Qwen3-30B-A3B | 4 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 27.792500 | 27.360000 | 1.015808 | 28.673000 | 27.360000 | 1.047990 |
| sm_103a | Qwen3-30B-A3B | 5 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 30.881000 | 30.977000 | 0.996901 | 30.592000 | 30.977000 | 0.987571 |
| sm_103a | Qwen3-30B-A3B | 6 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 33.472000 | 33.440000 | 1.000957 | 33.952000 | 33.440000 | 1.015311 |
| sm_103a | Qwen3-30B-A3B | 7 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 37.377000 | 37.153000 | 1.006029 | 36.993000 | 37.153000 | 0.995693 |
| sm_103a | Qwen3-30B-A3B | 8 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 40.193000 | 40.033000 | 1.003997 | 39.233000 | 40.033000 | 0.980016 |
| sm_103a | Qwen3-30B-A3B | 9 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 42.368000 | 42.496000 | 0.996988 | 41.216000 | 42.496000 | 0.969880 |
| sm_103a | Qwen3-30B-A3B | 10 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 44.544000 | 44.528500 | 1.000348 | 42.721000 | 44.528500 | 0.959408 |
| sm_103a | Qwen3-30B-A3B | 11 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 47.168000 | 47.104000 | 1.001359 | 44.321000 | 47.104000 | 0.940918 |
| sm_103a | Qwen3-30B-A3B | 12 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 49.505000 | 49.664000 | 0.996798 | 45.472000 | 49.664000 | 0.915593 |
| sm_103a | Qwen3-30B-A3B | 13 | sm_103a_c01 | sm_103a_gpu01 | sm_103a_source01 | PASS | 51.745000 | 51.777000 | 0.999382 | 47.457000 | 51.777000 | 0.916565 |
| sm_103a | Qwen3-30B-A3B | 14 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 53.793000 | 53.888000 | 0.998237 | 48.480000 | 53.888000 | 0.899644 |
| sm_103a | Qwen3-30B-A3B | 15 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 55.968000 | 56.160000 | 0.996581 | 50.528000 | 56.160000 | 0.899715 |
| sm_103a | Qwen3-30B-A3B | 16 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 58.112000 | 58.240000 | 0.997802 | 51.553000 | 58.240000 | 0.885182 |
| sm_103a | Qwen3-30B-A3B | 17 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 60.512000 | 60.449000 | 1.001042 | 52.960000 | 60.449000 | 0.876110 |
| sm_103a | Qwen3-30B-A3B | 18 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 63.201000 | 63.104000 | 1.001537 | 54.401000 | 63.104000 | 0.862085 |
| sm_103a | Qwen3-30B-A3B | 19 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 65.281000 | 65.185000 | 1.001473 | 54.913000 | 65.185000 | 0.842418 |
| sm_103a | Qwen3-30B-A3B | 20 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 67.072000 | 66.880000 | 1.002871 | 55.488000 | 66.880000 | 0.829665 |
| sm_103a | Qwen3-30B-A3B | 21 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 68.640000 | 68.544000 | 1.001401 | 56.480000 | 68.544000 | 0.823996 |
| sm_103a | Qwen3-30B-A3B | 22 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 70.113000 | 69.952000 | 1.002302 | 57.920000 | 69.952000 | 0.827996 |
| sm_103a | Qwen3-30B-A3B | 23 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 72.000000 | 72.065000 | 0.999098 | 58.304000 | 72.065000 | 0.809047 |
| sm_103a | Qwen3-30B-A3B | 24 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 74.017000 | 74.112000 | 0.998718 | 58.593000 | 74.112000 | 0.790601 |
| sm_103a | Qwen3-30B-A3B | 25 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 75.809000 | 75.809000 | 1.000000 | 60.544000 | 75.809000 | 0.798639 |
| sm_103a | Qwen3-30B-A3B | 26 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 77.888000 | 78.049000 | 0.997937 | 62.112000 | 78.049000 | 0.795808 |
| sm_103a | Qwen3-30B-A3B | 27 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 79.872000 | 80.065000 | 0.997589 | 62.593000 | 80.065000 | 0.781777 |
| sm_103a | Qwen3-30B-A3B | 28 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 81.473000 | 81.632000 | 0.998052 | 63.040000 | 81.632000 | 0.772246 |
| sm_103a | Qwen3-30B-A3B | 29 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 83.744500 | 83.937000 | 0.997707 | 64.257000 | 83.937000 | 0.765538 |
| sm_103a | Qwen3-30B-A3B | 30 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 85.665000 | 85.888500 | 0.997398 | 64.704000 | 85.888500 | 0.753349 |
| sm_103a | Qwen3-30B-A3B | 31 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 88.001000 | 88.129000 | 0.998548 | 65.280000 | 88.129000 | 0.740732 |
| sm_103a | Qwen3-30B-A3B | 32 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 89.984000 | 90.049000 | 0.999278 | 65.312000 | 90.049000 | 0.725294 |
| sm_103a | Qwen3-235B-A22B | 1 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 32.960000 | 32.928000 | 1.000972 | 34.944000 | 32.928000 | 1.061224 |
| sm_103a | Qwen3-235B-A22B | 2 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 43.456000 | 43.552000 | 0.997796 | 52.449000 | 43.552000 | 1.204285 |
| sm_103a | Qwen3-235B-A22B | 3 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 58.433000 | 58.496000 | 0.998923 | 67.040000 | 58.496000 | 1.146061 |
| sm_103a | Qwen3-235B-A22B | 4 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 69.472000 | 69.505000 | 0.999525 | 79.424000 | 69.505000 | 1.142709 |
| sm_103a | Qwen3-235B-A22B | 5 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 78.880000 | 78.624000 | 1.003256 | 90.144000 | 78.624000 | 1.146520 |
| sm_103a | Qwen3-235B-A22B | 6 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 89.505000 | 89.505000 | 1.000000 | 102.625000 | 89.505000 | 1.146584 |
| sm_103a | Qwen3-235B-A22B | 7 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 102.305000 | 101.664000 | 1.006305 | 116.864000 | 101.664000 | 1.149512 |
| sm_103a | Qwen3-235B-A22B | 8 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 113.601000 | 113.632000 | 0.999727 | 124.865000 | 113.632000 | 1.098854 |
| sm_103a | Qwen3-235B-A22B | 9 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 123.873000 | 123.969000 | 0.999226 | 132.577000 | 123.969000 | 1.069437 |
| sm_103a | Qwen3-235B-A22B | 10 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 133.601500 | 134.017000 | 0.996900 | 138.464000 | 134.017000 | 1.033182 |
| sm_103a | Qwen3-235B-A22B | 11 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 145.984000 | 145.313000 | 1.004618 | 144.289000 | 145.313000 | 0.992953 |
| sm_103a | Qwen3-235B-A22B | 12 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 155.905000 | 155.169000 | 1.004743 | 150.081000 | 155.169000 | 0.967210 |
| sm_103a | Qwen3-235B-A22B | 13 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 165.537000 | 165.473000 | 1.000387 | 157.409000 | 165.473000 | 0.951267 |
| sm_103a | Qwen3-235B-A22B | 14 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 177.281000 | 177.345000 | 0.999639 | 159.425000 | 177.345000 | 0.898954 |
| sm_103a | Qwen3-235B-A22B | 15 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 188.673000 | 188.642000 | 1.000164 | 166.562000 | 188.642000 | 0.882953 |
| sm_103a | Qwen3-235B-A22B | 16 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 198.337000 | 198.338000 | 0.999995 | 170.402000 | 198.338000 | 0.859150 |
| sm_103a | Qwen3-235B-A22B | 17 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 209.378000 | 209.634000 | 0.998779 | 176.193000 | 209.634000 | 0.840479 |
| sm_103a | Qwen3-235B-A22B | 18 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 220.546000 | 221.217000 | 0.996967 | 181.345000 | 221.217000 | 0.819761 |
| sm_103a | Qwen3-235B-A22B | 19 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 228.898000 | 229.153000 | 0.998887 | 183.313500 | 229.153000 | 0.799961 |
| sm_103a | Qwen3-235B-A22B | 20 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 238.306000 | 238.722000 | 0.998257 | 185.089000 | 238.722000 | 0.775333 |
| sm_103a | Qwen3-235B-A22B | 21 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 248.322000 | 247.874000 | 1.001807 | 188.802000 | 247.874000 | 0.761685 |
| sm_103a | Qwen3-235B-A22B | 22 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 258.690000 | 258.754000 | 0.999753 | 194.305000 | 258.754000 | 0.750926 |
| sm_103a | Qwen3-235B-A22B | 23 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 270.018000 | 269.091000 | 1.003445 | 194.401000 | 269.091000 | 0.722436 |
| sm_103a | Qwen3-235B-A22B | 24 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 279.778000 | 279.394000 | 1.001374 | 196.081500 | 279.394000 | 0.701810 |
| sm_103a | Qwen3-235B-A22B | 25 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 288.578000 | 288.674000 | 0.999667 | 203.554000 | 288.674000 | 0.705135 |
| sm_103a | Qwen3-235B-A22B | 26 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 300.259000 | 300.354000 | 0.999684 | 211.650000 | 300.354000 | 0.704668 |
| sm_103a | Qwen3-235B-A22B | 27 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 312.099000 | 312.098000 | 1.000003 | 213.345000 | 312.098000 | 0.683583 |
| sm_103a | Qwen3-235B-A22B | 28 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 324.050500 | 324.067000 | 0.999949 | 215.138000 | 324.067000 | 0.663869 |
| sm_103a | Qwen3-235B-A22B | 29 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 334.210000 | 334.275000 | 0.999806 | 219.073000 | 334.275000 | 0.655368 |
| sm_103a | Qwen3-235B-A22B | 30 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 344.643000 | 344.035000 | 1.001767 | 220.881500 | 344.035000 | 0.642032 |
| sm_103a | Qwen3-235B-A22B | 31 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 354.659000 | 354.435000 | 1.000632 | 224.481000 | 354.435000 | 0.633349 |
| sm_103a | Qwen3-235B-A22B | 32 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 365.635000 | 364.995000 | 1.001753 | 224.610000 | 364.995000 | 0.615378 |
| sm_103a | Qwen3.5-35B-A3B | 1 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 14.400000 | 14.113000 | 1.020336 | 15.584000 | 14.113000 | 1.104230 |
| sm_103a | Qwen3.5-35B-A3B | 2 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 17.024000 | 16.929000 | 1.005612 | 19.296000 | 16.929000 | 1.139819 |
| sm_103a | Qwen3.5-35B-A3B | 3 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 20.352000 | 20.545000 | 0.990606 | 22.240000 | 20.545000 | 1.082502 |
| sm_103a | Qwen3.5-35B-A3B | 4 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 23.233000 | 23.297000 | 0.997253 | 24.928000 | 23.297000 | 1.070009 |
| sm_103a | Qwen3.5-35B-A3B | 5 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 26.528000 | 26.208000 | 1.012210 | 27.424000 | 26.208000 | 1.046398 |
| sm_103a | Qwen3.5-35B-A3B | 6 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 28.704000 | 28.672000 | 1.001116 | 30.144000 | 28.672000 | 1.051339 |
| sm_103a | Qwen3.5-35B-A3B | 7 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 31.201000 | 31.040000 | 1.005187 | 32.960000 | 31.040000 | 1.061856 |
| sm_103a | Qwen3.5-35B-A3B | 8 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 33.664000 | 34.048000 | 0.988722 | 35.616000 | 34.048000 | 1.046053 |
| sm_103a | Qwen3.5-35B-A3B | 9 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 36.096000 | 36.384000 | 0.992084 | 37.472000 | 36.384000 | 1.029903 |
| sm_103a | Qwen3.5-35B-A3B | 10 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 37.761000 | 38.208000 | 0.988301 | 38.912000 | 38.208000 | 1.018425 |
| sm_103a | Qwen3.5-35B-A3B | 11 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 39.744000 | 39.968000 | 0.994396 | 41.088000 | 39.968000 | 1.028022 |
| sm_103a | Qwen3.5-35B-A3B | 12 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 41.472000 | 41.409000 | 1.001521 | 42.016000 | 41.409000 | 1.014659 |
| sm_103a | Qwen3.5-35B-A3B | 13 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 42.976000 | 43.105000 | 0.997007 | 44.064000 | 43.105000 | 1.022248 |
| sm_103a | Qwen3.5-35B-A3B | 14 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 44.673000 | 44.833000 | 0.996431 | 45.152000 | 44.833000 | 1.007115 |
| sm_103a | Qwen3.5-35B-A3B | 15 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 46.464000 | 46.529000 | 0.998603 | 46.560000 | 46.529000 | 1.000666 |
| sm_103a | Qwen3.5-35B-A3B | 16 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 48.417000 | 48.513000 | 0.998021 | 47.936000 | 48.513000 | 0.988106 |
| sm_103a | Qwen3.5-35B-A3B | 17 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 50.176000 | 50.112000 | 1.001277 | 49.536000 | 50.112000 | 0.988506 |
| sm_103a | Qwen3.5-35B-A3B | 18 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 51.552000 | 51.648000 | 0.998141 | 50.496000 | 51.648000 | 0.977695 |
| sm_103a | Qwen3.5-35B-A3B | 19 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 53.408000 | 53.440000 | 0.999401 | 51.776000 | 53.440000 | 0.968862 |
| sm_103a | Qwen3.5-35B-A3B | 20 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 55.232000 | 55.168000 | 1.001160 | 52.960000 | 55.168000 | 0.959977 |
| sm_103a | Qwen3.5-35B-A3B | 21 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 56.609000 | 56.641000 | 0.999435 | 53.792000 | 56.641000 | 0.949701 |
| sm_103a | Qwen3.5-35B-A3B | 22 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 57.824000 | 57.985000 | 0.997223 | 55.072000 | 57.985000 | 0.949763 |
| sm_103a | Qwen3.5-35B-A3B | 23 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 59.329000 | 59.456000 | 0.997864 | 56.064000 | 59.456000 | 0.942949 |
| sm_103a | Qwen3.5-35B-A3B | 24 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 61.184000 | 60.961000 | 1.003658 | 56.577000 | 60.961000 | 0.928085 |
| sm_103a | Qwen3.5-35B-A3B | 25 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 62.657000 | 62.849000 | 0.996945 | 58.209000 | 62.849000 | 0.926172 |
| sm_103a | Qwen3.5-35B-A3B | 26 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 64.384000 | 64.417000 | 0.999488 | 59.552000 | 64.417000 | 0.924476 |
| sm_103a | Qwen3.5-35B-A3B | 27 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 66.049000 | 66.049000 | 1.000000 | 60.480000 | 66.049000 | 0.915684 |
| sm_103a | Qwen3.5-35B-A3B | 28 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 68.097000 | 68.032000 | 1.000955 | 62.529000 | 68.032000 | 0.919112 |
| sm_103a | Qwen3.5-35B-A3B | 29 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 69.793000 | 70.016000 | 0.996815 | 64.448000 | 70.016000 | 0.920475 |
| sm_103a | Qwen3.5-35B-A3B | 30 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 71.585000 | 71.585000 | 1.000000 | 65.856000 | 71.585000 | 0.919969 |
| sm_103a | Qwen3.5-35B-A3B | 31 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 73.408000 | 73.345000 | 1.000859 | 67.553000 | 73.345000 | 0.921031 |
| sm_103a | Qwen3.5-35B-A3B | 32 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 74.816000 | 75.520000 | 0.990678 | 68.321000 | 75.520000 | 0.904674 |
| sm_103a | Qwen3.5-397B-A17B | 1 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 31.232000 | 30.848000 | 1.012448 | 28.384000 | 30.848000 | 0.920124 |
| sm_103a | Qwen3.5-397B-A17B | 2 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 39.520000 | 39.552000 | 0.999191 | 41.568000 | 39.552000 | 1.050971 |
| sm_103a | Qwen3.5-397B-A17B | 3 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 52.225000 | 51.873000 | 1.006786 | 54.145000 | 51.873000 | 1.043799 |
| sm_103a | Qwen3.5-397B-A17B | 4 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 63.969000 | 63.712000 | 1.004034 | 65.664000 | 63.712000 | 1.030638 |
| sm_103a | Qwen3.5-397B-A17B | 5 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 75.233000 | 75.168000 | 1.000865 | 77.216000 | 75.168000 | 1.027246 |
| sm_103a | Qwen3.5-397B-A17B | 6 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 86.433000 | 86.401000 | 1.000370 | 88.256000 | 86.401000 | 1.021470 |
| sm_103a | Qwen3.5-397B-A17B | 7 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 94.657000 | 94.721000 | 0.999324 | 95.009000 | 94.721000 | 1.003041 |
| sm_103a | Qwen3.5-397B-A17B | 8 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 103.905000 | 103.935000 | 0.999711 | 103.233000 | 103.935000 | 0.993246 |
| sm_103a | Qwen3.5-397B-A17B | 9 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 113.728000 | 113.441000 | 1.002530 | 112.608000 | 113.441000 | 0.992657 |
| sm_103a | Qwen3.5-397B-A17B | 10 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 123.745000 | 123.777000 | 0.999741 | 122.465000 | 123.777000 | 0.989400 |
| sm_103a | Qwen3.5-397B-A17B | 11 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 131.296000 | 131.680000 | 0.997084 | 128.513000 | 131.680000 | 0.975949 |
| sm_103a | Qwen3.5-397B-A17B | 12 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 141.025000 | 141.121000 | 0.999320 | 134.400500 | 141.121000 | 0.952378 |
| sm_103a | Qwen3.5-397B-A17B | 13 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 151.105000 | 150.944500 | 1.001063 | 142.466000 | 150.944500 | 0.943830 |
| sm_103a | Qwen3.5-397B-A17B | 14 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 160.513000 | 160.705000 | 0.998805 | 150.305000 | 160.705000 | 0.935285 |
| sm_103a | Qwen3.5-397B-A17B | 15 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 170.946000 | 170.721000 | 1.001318 | 159.553000 | 170.721000 | 0.934583 |
| sm_103a | Qwen3.5-397B-A17B | 16 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 179.585000 | 179.554000 | 1.000173 | 167.457000 | 179.554000 | 0.932628 |
| sm_103a | Qwen3.5-397B-A17B | 17 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 188.290000 | 188.753500 | 0.997544 | 173.985000 | 188.753500 | 0.921758 |
| sm_103a | Qwen3.5-397B-A17B | 18 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 198.018000 | 198.178000 | 0.999193 | 182.881000 | 198.178000 | 0.922812 |
| sm_103a | Qwen3.5-397B-A17B | 19 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 208.161000 | 207.553000 | 1.002929 | 190.561000 | 207.553000 | 0.918132 |
| sm_103a | Qwen3.5-397B-A17B | 20 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 217.794000 | 217.826000 | 0.999853 | 196.386000 | 217.826000 | 0.901573 |
| sm_103a | Qwen3.5-397B-A17B | 21 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 226.818000 | 226.497000 | 1.001417 | 202.433000 | 226.497000 | 0.893756 |
| sm_103a | Qwen3.5-397B-A17B | 22 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 235.586000 | 235.170000 | 1.001769 | 206.466000 | 235.170000 | 0.877944 |
| sm_103a | Qwen3.5-397B-A17B | 23 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 245.602000 | 245.441500 | 1.000654 | 211.809000 | 245.441500 | 0.862971 |
| sm_103a | Qwen3.5-397B-A17B | 24 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 254.178000 | 254.466000 | 0.998868 | 217.634000 | 254.466000 | 0.855258 |
| sm_103a | Qwen3.5-397B-A17B | 25 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 264.034000 | 264.130000 | 0.999637 | 223.601500 | 264.130000 | 0.846559 |
| sm_103a | Qwen3.5-397B-A17B | 26 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 272.866000 | 273.346000 | 0.998244 | 228.770000 | 273.346000 | 0.836925 |
| sm_103a | Qwen3.5-397B-A17B | 27 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 282.242000 | 282.530000 | 0.998981 | 233.698000 | 282.530000 | 0.827162 |
| sm_103a | Qwen3.5-397B-A17B | 28 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 292.466500 | 292.626500 | 0.999453 | 236.642000 | 292.626500 | 0.808683 |
| sm_103a | Qwen3.5-397B-A17B | 29 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 302.307000 | 302.499000 | 0.999365 | 244.290000 | 302.499000 | 0.807573 |
| sm_103a | Qwen3.5-397B-A17B | 30 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 311.490000 | 310.979000 | 1.001643 | 247.394000 | 310.979000 | 0.795533 |
| sm_103a | Qwen3.5-397B-A17B | 31 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 321.603000 | 321.412000 | 1.000594 | 253.250500 | 321.412000 | 0.787931 |
| sm_103a | Qwen3.5-397B-A17B | 32 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 331.011000 | 331.203000 | 0.999420 | 256.131000 | 331.203000 | 0.773335 |
| sm_103a | MiniMax-M2 | 1 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 29.248000 | 29.280000 | 0.998907 | 31.136000 | 29.280000 | 1.063388 |
| sm_103a | MiniMax-M2 | 2 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 35.679500 | 35.521000 | 1.004462 | 44.032000 | 35.521000 | 1.239605 |
| sm_103a | MiniMax-M2 | 3 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 47.329000 | 47.392000 | 0.998671 | 57.152000 | 47.392000 | 1.205942 |
| sm_103a | MiniMax-M2 | 4 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 58.881000 | 58.657000 | 1.003819 | 69.537000 | 58.657000 | 1.185485 |
| sm_103a | MiniMax-M2 | 5 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 67.904000 | 67.584000 | 1.004735 | 79.393000 | 67.584000 | 1.174731 |
| sm_103a | MiniMax-M2 | 6 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 77.953000 | 77.665000 | 1.003708 | 92.385000 | 77.665000 | 1.189532 |
| sm_103a | MiniMax-M2 | 7 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 87.424000 | 87.104000 | 1.003674 | 104.897000 | 87.104000 | 1.204273 |
| sm_103a | MiniMax-M2 | 8 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 96.865000 | 96.769000 | 1.000992 | 115.584000 | 96.769000 | 1.194432 |
| sm_103a | MiniMax-M2 | 9 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 105.761000 | 105.824000 | 0.999405 | 125.408000 | 105.824000 | 1.185062 |
| sm_103a | MiniMax-M2 | 10 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 112.225000 | 112.416000 | 0.998301 | 131.137000 | 112.416000 | 1.166533 |
| sm_103a | MiniMax-M2 | 11 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 119.361000 | 119.456000 | 0.999205 | 139.553000 | 119.456000 | 1.168238 |
| sm_103a | MiniMax-M2 | 12 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 127.713000 | 127.872000 | 0.998757 | 143.681500 | 127.872000 | 1.123635 |
| sm_103a | MiniMax-M2 | 13 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 135.521000 | 135.874000 | 0.997402 | 152.033000 | 135.874000 | 1.118926 |
| sm_103a | MiniMax-M2 | 14 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 143.169000 | 143.649000 | 0.996659 | 156.481000 | 143.649000 | 1.089329 |
| sm_103a | MiniMax-M2 | 15 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 151.905000 | 152.193000 | 0.998108 | 162.081000 | 152.193000 | 1.064970 |
| sm_103a | MiniMax-M2 | 16 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 160.961000 | 160.497500 | 1.002888 | 167.969000 | 160.497500 | 1.046552 |
| sm_103a | MiniMax-M2 | 17 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 169.666000 | 169.377000 | 1.001706 | 174.753000 | 169.377000 | 1.031740 |
| sm_103a | MiniMax-M2 | 18 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 176.577000 | 176.353000 | 1.001270 | 178.881000 | 176.353000 | 1.014335 |
| sm_103a | MiniMax-M2 | 19 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 184.545000 | 184.897000 | 0.998096 | 184.385000 | 184.897000 | 0.997231 |
| sm_103a | MiniMax-M2 | 20 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 192.162000 | 192.098000 | 1.000333 | 189.730000 | 192.098000 | 0.987673 |
| sm_103a | MiniMax-M2 | 21 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 200.226000 | 199.937000 | 1.001445 | 193.794000 | 199.937000 | 0.969275 |
| sm_103a | MiniMax-M2 | 22 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 208.386000 | 208.033000 | 1.001697 | 199.137000 | 208.033000 | 0.957238 |
| sm_103a | MiniMax-M2 | 23 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 216.225000 | 216.162000 | 1.000291 | 203.217500 | 216.162000 | 0.940117 |
| sm_103a | MiniMax-M2 | 24 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 224.610000 | 224.289000 | 1.001431 | 205.889000 | 224.289000 | 0.917963 |
| sm_103a | MiniMax-M2 | 25 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 233.186000 | 233.121000 | 1.000279 | 213.089000 | 233.121000 | 0.914070 |
| sm_103a | MiniMax-M2 | 26 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 241.409000 | 241.602000 | 0.999201 | 219.618000 | 241.602000 | 0.909007 |
| sm_103a | MiniMax-M2 | 27 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 250.242000 | 250.594000 | 0.998595 | 223.650000 | 250.594000 | 0.892479 |
| sm_103a | MiniMax-M2 | 28 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 259.138000 | 259.650000 | 0.998028 | 231.746000 | 259.650000 | 0.892532 |
| sm_103a | MiniMax-M2 | 29 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 267.970000 | 268.162000 | 0.999284 | 238.818000 | 268.162000 | 0.890574 |
| sm_103a | MiniMax-M2 | 30 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 276.482000 | 276.802000 | 0.998844 | 245.666000 | 276.802000 | 0.887515 |
| sm_103a | MiniMax-M2 | 31 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 284.098000 | 284.258000 | 0.999437 | 254.178000 | 284.258000 | 0.894181 |
| sm_103a | MiniMax-M2 | 32 | sm_103a_c02 | sm_103a_gpu02 | sm_103a_source02 | PASS | 291.714000 | 292.066000 | 0.998795 | 258.210000 | 292.066000 | 0.884081 |
| sm_103a | MiniMax-M3 | 1 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 42.784000 | 42.784000 | 1.000000 | 38.976000 | 42.784000 | 0.910995 |
| sm_103a | MiniMax-M3 | 2 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 57.888000 | 57.824000 | 1.001107 | 59.489000 | 57.824000 | 1.028794 |
| sm_103a | MiniMax-M3 | 3 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 78.657000 | 78.368000 | 1.003688 | 80.768000 | 78.368000 | 1.030625 |
| sm_103a | MiniMax-M3 | 4 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 97.953000 | 98.081000 | 0.998695 | 99.521000 | 98.081000 | 1.014682 |
| sm_103a | MiniMax-M3 | 5 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 111.169000 | 111.105000 | 1.000576 | 109.152000 | 111.105000 | 0.982422 |
| sm_103a | MiniMax-M3 | 6 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 129.249000 | 129.281000 | 0.999752 | 127.297000 | 129.281000 | 0.984654 |
| sm_103a | MiniMax-M3 | 7 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 147.585000 | 147.521000 | 1.000434 | 144.674000 | 147.521000 | 0.980701 |
| sm_103a | MiniMax-M3 | 8 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 165.025000 | 164.897000 | 1.000776 | 157.858000 | 164.897000 | 0.957313 |
| sm_103a | MiniMax-M3 | 9 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 183.842000 | 183.649000 | 1.001051 | 176.545000 | 183.649000 | 0.961318 |
| sm_103a | MiniMax-M3 | 10 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 196.802000 | 196.674000 | 1.000651 | 181.121000 | 196.674000 | 0.920920 |
| sm_103a | MiniMax-M3 | 11 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 214.306000 | 214.530000 | 0.998956 | 189.826000 | 214.530000 | 0.884846 |
| sm_103a | MiniMax-M3 | 12 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 232.418000 | 232.481000 | 0.999729 | 198.658000 | 232.481000 | 0.854513 |
| sm_103a | MiniMax-M3 | 13 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 250.370000 | 250.114000 | 1.001024 | 216.545000 | 250.114000 | 0.865785 |
| sm_103a | MiniMax-M3 | 14 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 268.066000 | 267.938000 | 1.000478 | 216.642000 | 267.938000 | 0.808553 |
| sm_103a | MiniMax-M3 | 15 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 285.794000 | 285.826000 | 0.999888 | 234.466000 | 285.826000 | 0.820310 |
| sm_103a | MiniMax-M3 | 16 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 303.810000 | 303.874000 | 0.999789 | 243.298000 | 303.874000 | 0.800654 |
| sm_103a | MiniMax-M3 | 17 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 322.339000 | 321.315000 | 1.003187 | 256.450000 | 321.315000 | 0.798126 |
| sm_103a | MiniMax-M3 | 18 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 339.331000 | 339.011500 | 1.000942 | 269.762000 | 339.011500 | 0.795731 |
| sm_103a | MiniMax-M3 | 19 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 354.307000 | 354.083000 | 1.000633 | 278.498000 | 354.083000 | 0.786533 |
| sm_103a | MiniMax-M3 | 20 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 368.995000 | 369.091000 | 0.999740 | 278.531000 | 369.091000 | 0.754640 |
| sm_103a | MiniMax-M3 | 21 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 386.564000 | 386.563000 | 1.000003 | 291.842000 | 386.563000 | 0.754966 |
| sm_103a | MiniMax-M3 | 22 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 403.972000 | 403.843000 | 1.000319 | 300.626500 | 403.843000 | 0.744414 |
| sm_103a | MiniMax-M3 | 23 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 422.083000 | 421.731000 | 1.000835 | 305.202500 | 421.731000 | 0.723690 |
| sm_103a | MiniMax-M3 | 24 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 439.779000 | 439.811000 | 0.999927 | 313.699000 | 439.811000 | 0.713259 |
| sm_103a | MiniMax-M3 | 25 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 457.348000 | 457.316000 | 1.000070 | 331.427000 | 457.316000 | 0.724722 |
| sm_103a | MiniMax-M3 | 26 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 474.948000 | 474.756000 | 1.000404 | 340.098500 | 474.756000 | 0.716365 |
| sm_103a | MiniMax-M3 | 27 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 492.484000 | 492.484000 | 1.000000 | 344.771000 | 492.484000 | 0.700065 |
| sm_103a | MiniMax-M3 | 28 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 510.212000 | 510.148000 | 1.000125 | 353.474500 | 510.148000 | 0.692886 |
| sm_103a | MiniMax-M3 | 29 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 528.100000 | 527.909000 | 1.000362 | 362.403000 | 527.909000 | 0.686488 |
| sm_103a | MiniMax-M3 | 30 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 545.669000 | 545.669000 | 1.000000 | 375.427000 | 545.669000 | 0.688012 |
| sm_103a | MiniMax-M3 | 31 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 563.237000 | 563.253000 | 0.999972 | 388.611000 | 563.253000 | 0.689940 |
| sm_103a | MiniMax-M3 | 32 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 581.157000 | 581.317000 | 0.999725 | 388.579000 | 581.317000 | 0.668446 |
| sm_103a | Kimi-K3 | 1 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 77.889000 | 77.568000 | 1.004138 | 121.185000 | 77.568000 | 1.562307 |
| sm_103a | Kimi-K3 | 2 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 122.432000 | 122.753000 | 0.997385 | 212.642000 | 122.753000 | 1.732275 |
| sm_103a | Kimi-K3 | 3 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 168.193000 | 167.936500 | 1.001527 | 310.529000 | 167.936500 | 1.849086 |
| sm_103a | Kimi-K3 | 4 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 213.633000 | 213.506000 | 1.000595 | 402.563000 | 213.506000 | 1.885488 |
| sm_103a | Kimi-K3 | 5 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 258.306000 | 258.914000 | 0.997652 | 486.692000 | 258.914000 | 1.879744 |
| sm_103a | Kimi-K3 | 6 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 303.235000 | 303.442500 | 0.999316 | 579.749000 | 303.442500 | 1.910573 |
| sm_103a | Kimi-K3 | 7 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 348.803000 | 348.898500 | 0.999726 | 669.351000 | 348.898500 | 1.918469 |
| sm_103a | Kimi-K3 | 8 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 392.931000 | 393.347000 | 0.998942 | 756.070000 | 393.347000 | 1.922145 |
| sm_103a | Kimi-K3 | 9 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 437.187000 | 437.475000 | 0.999342 | 840.615000 | 437.475000 | 1.921516 |
| sm_103a | Kimi-K3 | 10 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 482.916000 | 482.676000 | 1.000497 | 925.191500 | 482.676000 | 1.916796 |
| sm_103a | Kimi-K3 | 11 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 527.109000 | 527.956000 | 0.998396 | 1017.384000 | 527.956000 | 1.927024 |
| sm_103a | Kimi-K3 | 12 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 572.276500 | 574.341000 | 0.996405 | 1107.689000 | 574.341000 | 1.928626 |
| sm_103a | Kimi-K3 | 13 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 616.822000 | 618.053000 | 0.998008 | 1173.578000 | 618.053000 | 1.898831 |
| sm_103a | Kimi-K3 | 14 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 661.157500 | 662.278000 | 0.998308 | 1264.443000 | 662.278000 | 1.909233 |
| sm_103a | Kimi-K3 | 15 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 706.086000 | 707.430500 | 0.998099 | 1349.659000 | 707.430500 | 1.907833 |
| sm_103a | Kimi-K3 | 16 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 751.142500 | 751.670000 | 0.999298 | 1427.741000 | 751.670000 | 1.899425 |
| sm_103a | Kimi-K3 | 17 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 794.919000 | 795.959000 | 0.998693 | 1488.973000 | 795.959000 | 1.870665 |
| sm_103a | Kimi-K3 | 18 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 840.279000 | 840.503500 | 0.999733 | 1543.277500 | 840.503500 | 1.836135 |
| sm_103a | Kimi-K3 | 19 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 883.560000 | 887.224000 | 0.995870 | 1603.438500 | 887.224000 | 1.807253 |
| sm_103a | Kimi-K3 | 20 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 929.017000 | 929.112000 | 0.999898 | 1658.511000 | 929.112000 | 1.785050 |
| sm_103a | Kimi-K3 | 21 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 973.112500 | 973.688500 | 0.999408 | 1718.656000 | 973.688500 | 1.765098 |
| sm_103a | Kimi-K3 | 22 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1018.776500 | 1018.937000 | 0.999842 | 1791.007000 | 1018.937000 | 1.757721 |
| sm_103a | Kimi-K3 | 23 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1063.449000 | 1063.641000 | 0.999819 | 1862.896000 | 1063.641000 | 1.751433 |
| sm_103a | Kimi-K3 | 24 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1107.481500 | 1108.090000 | 0.999451 | 1923.986000 | 1108.090000 | 1.736308 |
| sm_103a | Kimi-K3 | 25 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1150.874000 | 1152.906000 | 0.998237 | 1953.265500 | 1152.906000 | 1.694211 |
| sm_103a | Kimi-K3 | 26 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1195.482500 | 1195.483000 | 1.000000 | 2025.233500 | 1195.483000 | 1.694071 |
| sm_103a | Kimi-K3 | 27 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1240.207500 | 1239.807000 | 1.000323 | 2080.681000 | 1239.807000 | 1.678230 |
| sm_103a | Kimi-K3 | 28 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1283.499000 | 1286.827000 | 0.997414 | 2141.138500 | 1286.827000 | 1.663890 |
| sm_103a | Kimi-K3 | 29 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1327.289000 | 1329.961000 | 0.997991 | 2190.655500 | 1329.961000 | 1.647158 |
| sm_103a | Kimi-K3 | 30 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1373.691000 | 1373.305000 | 1.000281 | 2238.814500 | 1373.305000 | 1.630238 |
| sm_103a | Kimi-K3 | 31 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1417.819500 | 1418.779500 | 0.999323 | 2281.713500 | 1418.779500 | 1.608223 |
| sm_103a | Kimi-K3 | 32 | sm_103a_c03 | sm_103a_gpu02 | sm_103a_source03 | PASS | 1462.572000 | 1462.812500 | 0.999836 | 2347.186500 | 1462.812500 | 1.604571 |
