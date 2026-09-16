# Ulysses exported-kernel validation results

**40/40 paired cases pass** on B200, covering world sizes 2, 4, 6 and 8.

The original/exported runtime ratio is at least 0.97 for every case. Directional disagreement and endpoint drift are each at most 2%. The observed maxima are 1.841119% and 1.719404%, respectively.

## Large BF16 cases

| World size | Original (ms) | Exported (ms) | Original/exported |
| --- | ---: | ---: | ---: |
| 2 | 1.4595230 | 1.4595515 | 0.999980473× |
| 4 | 0.7714230 | 0.7710370 | 1.000500624× |
| 6 | 0.6361580 | 0.6367815 | 0.999020857× |
| 8 | 0.4333095 | 0.4325420 | 1.001774394× |

## All ten cases per world size

These are geometric means across the nine small cases and one large case in each world.

| World size | Cases | Original (ms) | Exported (ms) | Original/exported |
| --- | ---: | ---: | ---: | ---: |
| 2 | 10/10 | 0.1695465 | 0.0904708 | 1.8740454× |
| 4 | 10/10 | 0.1611024 | 0.0886523 | 1.8172392× |
| 6 | 10/10 | 0.1616249 | 0.0918189 | 1.7602577× |
| 8 | 10/10 | 0.1565181 | 0.0901457 | 1.7362795× |

## Measurement and correctness

Every measured call includes three head scatters, one independently supplied gather, and all four staging-to-output copies. Both arms use the same inputs and caller-owned output buffers. No attention operation is timed.

Measurements use direct launches, CUPTI activity timing, cold L2, and rank maximum before aggregation. One untimed call to the upcoming arm equalizes immediate call history. Each case has three paired groups. Every case passes bit-exact correctness before and after measurement, zero device allocations/frees during prepared export submission, activity-count checks and clock checks.

The first 15 cases retain their original 1000 ms measurement and 100 ms warmup budgets per arm/group. The remaining 25 use 150 ms and 20 ms, respectively. The original 15 measurements were reassessed under the 0.97 acceptance floor without changing their observations or repeating them.

## Full portfolio

S is local sequence length unless marked global. All rows pass.

| World | Dtype | Case | B | H | D | S | Original (ms) | Exported (ms) | Ratio | Order (%) | Drift (%) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | float16 | aligned | 1 | 24 | 128 | 8 | 0.1355205 | 0.0673600 | 2.011883907× | 0.166688 | 0.800754 |
| 2 | float16 | batch | 2 | 24 | 64 | 16 | 0.1327680 | 0.0660160 | 2.011148812× | 0.514917 | 0.412076 |
| 2 | float16 | scalar | 1 | 24 | 3 | 5 | 0.1324480 | 0.0658240 | 2.012153622× | 0.546860 | 0.774443 |
| 2 | bfloat16 | aligned | 1 | 24 | 128 | 8 | 0.1326070 | 0.0669440 | 1.980864603× | 0.487804 | 0.229746 |
| 2 | bfloat16 | batch | 2 | 24 | 64 | 16 | 0.1317440 | 0.0660160 | 1.995637421× | 0.596612 | 0.581488 |
| 2 | bfloat16 | scalar | 1 | 24 | 3 | 5 | 0.1318080 | 0.0661440 | 1.992743106× | 0.281162 | 0.291898 |
| 2 | float32 | aligned | 1 | 24 | 128 | 8 | 0.1320960 | 0.0668160 | 1.977011494× | 0.032840 | 0.193424 |
| 2 | float32 | batch | 2 | 24 | 64 | 16 | 0.1364480 | 0.0664960 | 2.051973051× | 0.585516 | 0.192585 |
| 2 | float32 | scalar | 1 | 22 | 3 | 5 | 0.1359680 | 0.0662070 | 2.053680124× | 0.250809 | 0.482160 |
| 2 | bfloat16 | wan | 1 | 40 | 128 | 32760 global | 1.4595230 | 1.4595515 | 0.999980473× | 0.015281 | 0.050329 |
| 4 | float16 | aligned | 1 | 24 | 128 | 8 | 0.1355520 | 0.0702090 | 1.930692646× | 0.821024 | 0.708215 |
| 4 | float16 | batch | 2 | 24 | 64 | 16 | 0.1349760 | 0.0694400 | 1.943778802× | 0.882503 | 0.118051 |
| 4 | float16 | scalar | 1 | 24 | 3 | 5 | 0.1344960 | 0.0681290 | 1.974137298× | 0.607306 | 0.429287 |
| 4 | bfloat16 | aligned | 1 | 24 | 128 | 8 | 0.1361600 | 0.0704000 | 1.934090909× | 0.966722 | 0.907441 |
| 4 | bfloat16 | batch | 2 | 24 | 64 | 16 | 0.1346880 | 0.0695680 | 1.936062557× | 0.730864 | 0.191495 |
| 4 | bfloat16 | scalar | 1 | 24 | 3 | 5 | 0.1382080 | 0.0708800 | 1.949887133× | 1.441370 | 0.880560 |
| 4 | float32 | aligned | 1 | 24 | 128 | 8 | 0.1346075 | 0.0697920 | 1.928695266× | 1.006070 | 0.366468 |
| 4 | float32 | batch | 2 | 24 | 64 | 16 | 0.1342710 | 0.0707840 | 1.896911731× | 1.256388 | 0.720576 |
| 4 | float32 | scalar | 1 | 24 | 3 | 5 | 0.1354230 | 0.0682720 | 1.983580384× | 0.704793 | 0.470628 |
| 4 | bfloat16 | wan | 1 | 40 | 128 | 32760 global | 0.7714230 | 0.7710370 | 1.000500624× | 0.020552 | 0.058170 |
| 6 | float16 | aligned | 1 | 24 | 128 | 8 | 0.1398075 | 0.0737270 | 1.896286299× | 0.914526 | 0.678887 |
| 6 | float16 | batch | 2 | 24 | 64 | 16 | 0.1380800 | 0.0740150 | 1.865567790× | 1.228300 | 1.178676 |
| 6 | float16 | scalar | 1 | 24 | 3 | 5 | 0.1387840 | 0.0744640 | 1.863773098× | 1.280807 | 1.719404 |
| 6 | bfloat16 | aligned | 1 | 24 | 128 | 8 | 0.1390390 | 0.0743360 | 1.870412721× | 1.493874 | 0.647017 |
| 6 | bfloat16 | batch | 2 | 24 | 64 | 16 | 0.1388470 | 0.0749435 | 1.852689026× | 1.365902 | 0.531915 |
| 6 | bfloat16 | scalar | 1 | 24 | 3 | 5 | 0.1391030 | 0.0734400 | 1.894104031× | 1.623787 | 0.301551 |
| 6 | float32 | aligned | 1 | 24 | 128 | 8 | 0.1385280 | 0.0736640 | 1.880538662× | 1.579281 | 0.478469 |
| 6 | float32 | batch | 2 | 24 | 64 | 16 | 0.1387520 | 0.0744000 | 1.864946237× | 1.413380 | 0.348150 |
| 6 | float32 | scalar | 1 | 30 | 3 | 5 | 0.1382710 | 0.0734080 | 1.883595793× | 0.791403 | 0.390455 |
| 6 | bfloat16 | divisible | 1 | 48 | 128 | 32760 global | 0.6361580 | 0.6367815 | 0.999020857× | 0.007661 | 0.102821 |
| 8 | float16 | aligned | 1 | 24 | 128 | 8 | 0.1398070 | 0.0752320 | 1.858344853× | 1.134097 | 0.297497 |
| 8 | float16 | batch | 2 | 24 | 64 | 16 | 0.1402550 | 0.0762230 | 1.840061399× | 1.091496 | 0.275527 |
| 8 | float16 | scalar | 1 | 24 | 3 | 5 | 0.1405750 | 0.0770550 | 1.824346246× | 1.019870 | 0.601785 |
| 8 | bfloat16 | aligned | 1 | 24 | 128 | 8 | 0.1398710 | 0.0755520 | 1.851320945× | 1.322953 | 0.783235 |
| 8 | bfloat16 | batch | 2 | 24 | 64 | 16 | 0.1388800 | 0.0757750 | 1.832794457× | 1.841119 | 0.394162 |
| 8 | bfloat16 | scalar | 1 | 24 | 3 | 5 | 0.1396150 | 0.0750710 | 1.859772748× | 1.487460 | 0.252613 |
| 8 | float32 | aligned | 1 | 24 | 128 | 8 | 0.1398710 | 0.0759030 | 1.842759838× | 1.794603 | 0.721256 |
| 8 | float32 | batch | 2 | 24 | 64 | 16 | 0.1394870 | 0.0758390 | 1.839251572× | 1.451809 | 0.505902 |
| 8 | float32 | scalar | 1 | 24 | 3 | 5 | 0.1396150 | 0.0749440 | 1.862924317× | 1.244819 | 0.344511 |
| 8 | bfloat16 | wan | 1 | 40 | 128 | 32760 global | 0.4333095 | 0.4325420 | 1.001774394× | 0.108474 | 0.133592 |

## Hardware and architecture coverage

Each campaign used eight B200 GPUs with NV18 connectivity and peer access across all 56 directed pairs. Every original/exported pair ran on the same campaign hardware. The first campaign used driver 610.57.04; the second used 580.82.07. Both used CUDA compiler 13.3.33, GCC 13.3.0, PyTorch 2.13.0 and TVM FFI 0.1.13.dev57+gd1fd51222. Generated sources, physical routes and compiler/linker flags match between campaigns; original row, binary and hardware identities are retained separately.

The second campaign's temporary topology files were removed during allocation cleanup. Its original successful hardware-check logs and the completed independent metadata validation retain the checked device identities and topology verdict; no raw topology matrix or peer observation was reconstructed.

| Architecture | Compilation | GPU correctness and performance |
| --- | --- | --- |
| SM100 / B200 | Passed | All 40 paired cases pass |
| SM103 / B300 | 62 translation units compiled and linked in 61.80 s | Not measured |

SM103 compilation does not establish B300 runtime correctness or performance.

- Compute Sanitizer synccheck: skipped after its hard 20-second timeout.
- Compute Sanitizer racecheck: skipped after its separate hard 20-second timeout.

## Elapsed time

The 40 shape-measurement stages total **13372.1 s**, including **2317.7 s** for the final 25 shorter-budget cases. These stage durations include the paired measurement harness; the GPU latencies are in the tables above.

The final 25-case campaign took **2912.905 s** from its first successful trial submission to the last GPU step completion. The full validation interval was **99968.022 s** from the first qualifying campaign submission to the final GPU completion, including inter-job gaps and resource waiting. Initial preparation and later publication are outside that interval.

## Public reproduction

```bash
python -m pytest tests/comm/test_cake_ulysses_a2a.py -v
torchrun --standalone --nproc-per-node=8 benchmarks/comm/bench_cake_ulysses_a2a.py --all-shapes --json results.json
```

Use 2, 4 or 6 processes for the other worlds. The public benchmark compares the generated implementation with NCCL; the original/exported comparison above comes from the paired export-validation campaign.
