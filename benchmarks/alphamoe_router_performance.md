# Router performance and validation report

**Overall performance acceptance: MET.** The completed prespecified precision matrix passes all original numeric performance gates for the frozen source and actual v5 public export. Source/public paired medians are `1`, `1`, `1`, `1`, `1.004149377593361`, and `1` in the original six-shape order. Original correctness, public tests, separate sanitizers, and actual public-model accuracy/input audits pass. Four negative source/public rounds remain disclosed; the requirement is evaluated on the original shape-level paired medians. Final PR-head CI is pending.

Completed evidence includes the frozen source, actual v1/v3/v4 and v5 measurements, failed v5 identity attempt, historical original grouped v5 matrix, completed precision grouped v5 matrix, and audited source/v4/v5 real-model results. Every historical failure and negative comparison remains disclosed. Final PR-head CI is pending.

Frozen source SHA256: `835ac0c1b8e7914b9bafed91c2575fecdcd6259b8efc9dd24a04c71b4543c956`.
Validated implementation commit: `a077c72a79f7330403607dbe9f3bce1d668c1537`. This identifies the code checked below. The PR description will record the final head and its terminal CI results after the documentation commit.

## Validated v5 code identity

| Public code path | SHA256 |
|---|---|
| `csrc/alphamoe_router/large.cu` | `44ad6d15ef38ee170a92c4ba3f09ff5e2902fbaa272fa68c7a9fd6c38e3fc024` |
| `csrc/alphamoe_router/large_routed.cu` | `0c90734a0da06b904801a2df6fe5df04ecf0419d66023c7210cb658f46043452` |
| `csrc/alphamoe_router/medium.cu` | `c49324220978396b5a23ed3fe825adbaf810f5d056603885015684826f66f4a7` |
| `csrc/alphamoe_router/medium_routed.cu` | `6e8c0b043af2b9de40a20ca3879f111567a9129e088ef8a9f3d978ae7db01417` |
| `csrc/alphamoe_router/small.cu` | `af161691b7cacb39014cc2815124e5c58348a3f698046b87276ae3b54065a27f` |
| `csrc/alphamoe_router/small_routed.cu` | `a9ca4df6733d9b814fbc8929cedcd4bc66754c4141547460564a44c9959c9685` |
| `csrc/alphamoe_router/tiny.cu` | `1aa5ce5d92499e3ad91e2f07adeaf449e278d6e57d922391f8e4664bf65ec752` |
| `csrc/alphamoe_router/tiny_routed.cu` | `cdaf1eda86b49fc2e53a597b897d1c149f9816e2a967a8368e488f834a6d41ce` |
| `csrc/alphamoe_router/large_tail.cu` | `05f157f63590043ffa8f4fc2567a1c163e6bd5123b7fbfac6c815d05e05bfecc` |
| `csrc/alphamoe_fused_router.cu` | `e02bcc6e8c126855cfd51df35fb640431fac0077afd68588ac9d0ac9fbd553e0` |
| `flashinfer/jit/alphamoe_nvrtc.py` | `4b0a0a0817d4f8490bdd4bad18e2ce8f6643f38aeb0c823bd0ea9cddc9a29458` |
| `flashinfer/jit/fused_moe.py` | `cd49928867fac78d0b7d3346ae3a0b4db510034de450c7f24f1b7c3822279e6c` |

Actual libraries are distinct and individually bound:

| Scope | Native library SHA256 |
|---|---|
| Initially tested v5 public artifact | `63b7ec5c53045c5c8d09ed1ed4a15b7994eccea932368f49d7a09d8a3c1d8409` |
| Original recovered public DSO used directly by the grouped performance matrix | `5c09623d3022f8804c84f963b10c8ba4c70987a57311d3343f0aaea4134c165e` |
| Isolated model-interpreter public DSO used by the actual v5 model run | `1edf1d81604ce75faf8534bb937885459b8aa2aecb7e2050e93d292d48401be4` |

The initial model-stage rebuild replaced the first library at its live cache path, causing the earlier v5 pair's final identity check to fail. That failure remains history123. Recovery used a separate cache; the grouped performance matrix loaded the recovered original library directly without a hidden clone or rebuild. The isolated model interpreter built its own host DSO while preserving the recovered public cache. Both completed runs bind implementation commit `a077c72a79f7330403607dbe9f3bce1d668c1537`, the exact twelve-file set above, the public API/import/JIT identities, and nine source-identical embedded cubins. Host-library bytes are reported separately for the measurement and model environments.

## Measurement boundary

GB300, SM103a, 512 experts, no shared expert, and all six original shapes. Times are cold-L2 CUPTI GPU kernel-duration sums over the complete operator; the large shape includes producer and tail. CPU gaps and model/API throughput are excluded. Each completed matrix retains six rounds, three buffer placements, all five source/public comparison arms where applicable, and every raw sample. Ratios are medians of within-round ratios, not divisions of the displayed marginal time medians.

## Frozen source result

| M/k/BM | Stock µs | v53 source µs | Frozen source µs | Stock/source | v53/source |
|---|---:|---:|---:|---:|---:|
| 32/8/16 | 12.040 | 8.440 | 7.224 | 1.669621 | 1.170921 |
| 128/8/16 | 12.320 | 10.528 | 9.336 | 1.319016 | 1.131409 |
| 8/10/8 | 11.368 | 6.320 | 5.200 | 2.184615 | 1.216593 |
| 128/10/8 | 14.064 | 9.312 | 7.928 | 1.757328 | 1.175498 |
| 512/10/8 | 16.480 | 12.576 | 7.744 | 2.130236 | 1.647280 |
| 16384/10/8 | 124.368 | 78.240 | 28.352 | 4.385722 | 2.766447 |

All six shape medians and all 36 rounds exceed 1 against both original Stock and v53 source. The broader 21-family source matrix retains 56 negative rounds, seven negative medians, 12 equal rounds, and three equal medians across its historical comparisons. These comparisons remain distinct from the Stock/v53 acceptance families.

## Actual export v1: nonqualifying

| M/k/BM | Stock µs | v53 source µs | v53 public µs | Source µs | Public µs | Stock/public | v53 public/public | Source/public |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/8/16 | 11.672 | 8.456 | 8.368 | 7.208 | 6.976 | 1.681258 | 1.202982 | 1.034403669725 |
| 128/8/16 | 12.256 | 10.528 | 10.504 | 9.360 | 9.416 | 1.297592 | 1.115612 | 0.990626903881 |
| 8/10/8 | 11.384 | 6.272 | 6.264 | 5.200 | 5.152 | 2.201860 | 1.212401 | 1.004658385093 |
| 128/10/8 | 14.256 | 9.456 | 9.248 | 7.992 | 8.016 | 1.772011 | 1.153998 | 0.997041048830 |
| 512/10/8 | 16.488 | 12.600 | 12.584 | 7.768 | 7.744 | 2.139463 | 1.644185 | 1.002087910172 |
| 16384/10/8 | 124.304 | 78.112 | 69.704 | 28.456 | 28.488 | 4.364510 | 2.448531 | 0.997479455254 |

Source/public: **3 negative shape medians, 15 negative rounds, 3 equal rounds**. Across all ten comparison families: 17 negative rounds and 7 equal rounds. Every original Stock and v53 acceptance comparison exceeds 1 for every shape and round, but the source/public nonregression gate is not met.

## Actual export v3: nonqualifying

| M/k/BM | Stock µs | v53 source µs | v53 public µs | Source µs | Public µs | Stock/public | v53 public/public | Source/public |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/8/16 | 9.696 | 8.544 | 8.512 | 7.296 | 7.328 | 1.327886 | 1.161572 | 0.997851086160 |
| 128/8/16 | 11.528 | 10.752 | 10.744 | 9.568 | 9.520 | 1.216695 | 1.127591 | 0.999143835616 |
| 8/10/8 | 11.960 | 6.016 | 5.992 | 5.008 | 4.960 | 2.403989 | 1.204839 | 1.003196022727 |
| 128/10/8 | 13.968 | 9.504 | 9.480 | 8.112 | 8.112 | 1.718931 | 1.159340 | 1.001949317739 |
| 512/10/8 | 16.480 | 12.600 | 12.592 | 7.704 | 7.672 | 2.162163 | 1.662874 | 1.007280901602 |
| 16384/10/8 | 123.560 | 76.480 | 68.808 | 28.152 | 28.160 | 4.389070 | 2.450568 | 1.000567214974 |

Source/public: **2 negative shape medians, 12 negative rounds, 6 equal rounds**. Across all ten comparison families: 17 negative rounds and 8 equal rounds. Every original Stock and v53 acceptance comparison exceeds 1 for every shape and round, but the source/public nonregression gate is not met.

### Source/public round disclosure

| M/k/BM | v1 below 1 / equal 1 | v3 below 1 / equal 1 |
|---|---:|---:|
| 32/8/16 | 0 / 0 | 3 / 0 |
| 128/8/16 | 5 / 1 | 3 / 1 |
| 8/10/8 | 0 / 2 | 1 / 2 |
| 128/10/8 | 4 / 0 | 2 / 1 |
| 512/10/8 | 2 / 0 | 1 / 1 |
| 16384/10/8 | 4 / 0 | 2 / 1 |

Each export matrix retains 5,400 samples, 7,920 kernel activities, and 12 large-shape component sidecars. History through v3 contains 121 matrices, 718 shape rows, 31,906 arm records, 84,904 ratios, 5,495 below-one records, and 4,278 rounds. Negative and equal ratios remain reported as measured. No negative result is relabeled as noise or rounded into a pass.

V1 compiled the generated kernels with NVCC. Attribution found different lowering from the source NVRTC path. V3 embeds source-identical NVRTC cubins in the actual public library; all nine embedded cubins and the seven selected source checks covering five measured kernels match. Its AOT module-name correction changes only lookup identity. These implementation checks do not replace the failed v3 performance gate.

## Actual export v4: nonqualifying; preserved history122

| M/k/BM | Stock µs | v53 source µs | v53 public µs | Source µs | Public µs | Stock/public | v53 public/public | Source/public |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/8/16 | 11.896 | 8.880 | 8.992 | 7.552 | 7.904 | 1.512106869916 | 1.138666584711 | 0.953770278935 |
| 128/8/16 | 12.264 | 10.696 | 10.720 | 9.576 | 9.752 | 1.256827158069 | 1.094572368421 | 0.981155400757 |
| 8/10/8 | 11.600 | 6.440 | 6.424 | 5.456 | 5.400 | 2.148157252471 | 1.179261847488 | 1.002980398317 |
| 128/10/8 | 15.192 | 9.640 | 9.464 | 8.248 | 8.616 | 1.767885962976 | 1.106724349601 | 0.961027809198 |
| 512/10/8 | 16.776 | 13.472 | 13.632 | 8.520 | 8.016 | 2.093808764940 | 1.702431579127 | 1.064228914302 |
| 16384/10/8 | 124.977 | 78.968 | 70.608 | 29.576 | 29.624 | 4.218210922573 | 2.369290273675 | 0.996769521206 |

V4 also fails source/export nonregression: four negative shape medians and 23 negative rounds, with no equal rounds. Across all ten comparison families it retains 40 negative rounds and seven negative medians. These outcomes remain nonqualifying alongside the completed v5 results. All original Stock and v53 acceptance comparisons still exceed 1 for every shape and round. The source/export negative-round counts, in table order, are 6, 5, 3, 5, 0 and 4. The matrix preserves 5,400 samples, 7,920 kernel activities, 1,080 separate five-arm cycle receipts and 12 large-shape sidecars.

V4 interleaves all five arms in each cycle, alternating forward/reverse order with 15 of each per round. Each arm receives an independent cold-L2 flush. Thirty cycles form each original round median, then six within-round ratios form each reported ratio. This protocol changes collection order, preserving the complete-operator denominator and existing gates.

The sealed [full historical shape table](alphamoe_router_history.csv) now has **742 rows** and the [full historical below-one table](alphamoe_router_below_one.csv) has **5,615 rows**. They represent history125: **125 matrices, 32,746 arm records, 86,584 ratios and 4,422 rounds**. Every original row and surviving field is preserved; only the internal summary-path column is removed. The precision matrix adds six below-one round values and zero negative medians. History124 remains intact with 124 matrices, 736 shape rows, 32,536 arm records, 86,164 ratios, 5,609 below-one values and 4,386 rounds. Its original grouped matrix added 21 below-one values: 18 rounds and three medians across all comparisons. Equal ratios remain in the complete ratio records and each matrix's separate disclosure evidence.

Its history123 predecessor remains intact: 123 matrices, 730 shape rows, 32,326 arm records, 85,744 ratios, 5,588 below-one records and 4,350 rounds. The `C133_EXPORT_V5_FAILED_IDENTITY` matrix remains incomplete after the terminal original-library identity failure caused by the concurrent model-interpreter cache rebuild. Its independently audited records add 46 negative values: 38 rounds and eight medians across all comparisons; source/public alone retains 21 negative rounds and four negative medians. All six sealed rows, 1,080 cycle records and twelve large-component sidecars remain unchanged. The audit establishes the executed clone's identity and accounts for the actual original-library/Ninja changes; it does not convert the failed attempt into successful acceptance. The initial audit's relative-path resolution error and its corrected audit are both retained. Recovery and model cache isolation preserve this failure history.

## V5 host configuration repair

A launch-suppressed comparison found equal arguments, grid/block dimensions, dynamic shared memory actually used, stream and cooperative flags. The source used CUDA Runtime launches and a default maximum dynamic shared-memory attribute of 49,152 bytes. V4 used Driver API launches and caps of 20,608 bytes for tiny/small or 4,224 bytes for medium/large/tail. These observations establish configuration differences; they do not establish the cause of the measured performance gap.

V5 uses the source-equivalent CUDA Runtime launcher and default function shared-memory attributes. The nine device sources, all nine embedded cubins, occupancy queries and grid calculations are preserved. The Runtime backend requires CUDA 12.8 or newer. The completed native build verifies the exact loaded library and all nine unchanged cubins. All 24 original public tests passed; separate synccheck and racecheck runs passed with zero reported errors/hazards within their individual 20-second caps. The historical original 30-sample grouped v5 pair remains NOT_MET. The completed prespecified 3,000-sample precision pair passes the original gates with this unchanged implementation; actual public-model accuracy and input/execution audits pass as reported below.

## Two-shape diagnostics: crossover and ordering

These diagnostics preserve the cold-L2 CUPTI complete-operator kernel-duration boundary and do not replace the five-arm, six-shape acceptance matrix. Host submission and persistence intervals remain diagnostic fields, separate from GPU kernel duration.

The handle/launcher crossover measured source/public self arms at 7.728/7.640 µs for M32 and 8.176250/8.168 µs for M512. Their within-round paired ratio medians were 1.009325361030 and 0.999050245098. The original self-arm comparison has one negative M32 round and three negative M512 rounds. Across its five factor comparisons, the crossover retains six and twelve negative rounds, respectively. It changed arm positions, persistence frequency and the public host library together; its results alone do not identify which factor explains the earlier gap.

The subsequent position diagnostic used the same retained public pair library for both conditions, the same five-arm call and persistence code, and alternated condition order within each round. The original condition reverses the five-arm order; the balanced condition rotates it so every arm occupies each position six times per round.

| M/k/BM | Original source/public µs | Original source/public ratio | Balanced source/public µs | Balanced source/public ratio | Below-one rounds, original/balanced |
|---|---:|---:|---:|---:|---:|
| 32/8/16 | 7.576 / 8.048 | 0.945709551657 | 7.568 / 7.984 | 0.951949899800 | 6 / 6 |
| 512/10/8 | 8.360 / 8.104 | 1.037056355510 | 8.432250 / 8.064 | 1.042818165707 | 1 / 0 |

Balancing positions does not remove the M32 gap: all six source/public rounds remain below 1 under both conditions. Ordering alone therefore does not resolve the observed regression in this diagnostic. M512 remains above 1 in both condition medians. Across all ten comparisons, M32 retains nine/six negative rounds and M512 four/two for original/balanced order; no values are relabeled as a pass or discarded.

The crossover retains 360 four-arm cycles (1,440 samples); the position diagnostic retains 720 five-arm condition blocks (3,600 samples). Their harness runtimes were 15.941530 s and 73.141299 s; actual physical turnaround was 27.347577 s and 88.300905 s. Both are diagnostic results, separate from history123's incomplete v5 acceptance attempt.

### Public-library and persistence diagnostics

The DSO comparison changes only the C133 export callable between the retained hidden clone and recovered original public library. Both conditions use the same original five-arm order, pointers and per-block persistence; condition order alternates within each round. Both actual libraries contain the same nine source cubins.

| M/k/BM | Clone source/public µs | Clone source/public ratio | Original-library source/public µs | Original-library source/public ratio | Below-one rounds, clone/original |
|---|---:|---:|---:|---:|---:|
| 32/8/16 | 7.656250 / 8.096 | 0.943711428795 | 7.784 / 7.968 | 0.974858276644 | 6 / 6 |
| 512/10/8 | 8.192250 / 8.256 | 1.001956977141 | 8.160 / 8.272 | 0.975735609602 | 2 / 5 |

Switching to the actual original public library reduces the M32 gap but leaves all six source/public rounds below 1. M512's original-library condition also has a median below 1. The C133 export clone/original duration ratios are 1.020222963799 for M32 and 0.993227801304 for M512; these are medians of paired condition ratios. The source and incumbent controls remain recorded separately. Across both shapes and conditions, the ten within-condition comparisons retain 22 negative rounds; the five cross-condition arm comparisons retain 29.

The persistence comparison uses the same recovered original public library in both conditions. Every cycle writes its raw evidence. One condition also serializes the aggregate/checkpoint every block; the other does so at the end of thirty cycles. Condition order alternates by round, with both orders at each of three pointer placements.

| M/k/BM | Per-block source/public µs | Per-block source/public ratio | Per-round source/public µs | Per-round source/public ratio | Below-one rounds, block/round |
|---|---:|---:|---:|---:|---:|
| 32/8/16 | 7.528 / 8.000 | 0.947026277420 | 7.520 / 8.008 | 0.937286696192 | 6 / 6 |
| 512/10/8 | 8.184 / 8.160 | 1.002000000000 | 8.112 / 8.120 | 1.000906250000 | 2 / 3 |

Reducing aggregate persistence does not remove the M32 gap. M512 retains negative rounds in both conditions and one equal source/public round in the per-block condition. Across both shapes and conditions, the ten within-condition comparisons retain 22 negative rounds; cross-condition arm comparisons retain 30. These results do not establish a fix or qualify a final export.

Each diagnostic retains 720 five-arm cycle files and 3,600 measured complete-operator calls. DSO comparison runtime/physical turnaround is **72.144538/85.130187 s**; persistence comparison is **48.032243/57.978972 s**. The exact native bindings remained unchanged through each run. All timings continue to use cold-L2 strict CUPTI kernel-duration sums, with CPU persistence/submission measurements kept separate. History123's failed identity/acceptance status is unchanged.

### Runtime binding and surrounding-call context

The live, untimed GOT probe observes the C133 source, hidden public clone and recovered original public `cudaLaunchKernelExC`, `cudaLibraryLoadData` and `cudaLibraryGetKernel` slots resolving to the same PyTorch-packaged CUDA runtime. It performs no rebinding or timing. Unused lazy-binding slots remain distinct in the raw record. The preceding ELF inspection alone did not establish these live bindings; neither probe proves a performance cause.

The context comparison keeps the same five visit positions, modules, actual source/export callables, pointers and native bytes loaded in one process. It alternates complete condition blocks within each round. Only the other three calls change: Stock/v53 source/v53 public versus C133 source/public/source. The measured source/export slots retain their identity. The C133-only context supplies no Stock denominator.

| M | Context | Source µs | Public µs | Source/public paired median | Negative/equal rounds |
|---|---|---:|---:|---:|---:|
| 32 | Stock/v53 calls | 7.400 | 7.464250 | 0.991428869762 | 5/0 |
| 32 | C133-only calls | 7.368 | 7.360 | 0.998920086393 | 3/1 |
| 512 | Stock/v53 calls | 8.264 | 8.176 | 1.011141906874 | 1/0 |
| 512 | C133-only calls | 8.280 | 8.264250 | 1.002912621359 | 2/1 |

Within this run, the Stock/v53-context over C133-only-context duration ratios are 1.003291967791 (source) and 1.014164402174 (public) for M32, and 0.999056854019/0.995143874042 for M512. These cross-context comparisons retain seven negative rounds and two negative medians; the fixed source/public comparisons retain eleven negative rounds, two negative medians and two equal rounds. M32 remains below source parity in both contexts. This does not qualify a final export.

This context run used a **different GB300 GPU** from the preceding DSO/persistence diagnostics. The table supports within-run context comparisons; absolute-time changes across those GPU runs cannot establish a context effect. All 720 cycle records and 3,600 complete-operator measurements are retained. Runtime/physical turnaround is **71.114177/89.771033 s**. The ELF probe took **0.262861/3.160436 s** and live GOT probe **18.786175/30.811923 s**. Node preflight took 0.346707 s inside the context work unit and verified the existing read-only model revision with no weight copies or downloads. The completed grouped six-shape matrix and actual v5 model result follow. The diagnostic outcomes do not replace their acceptance evidence.

### Occupancy-query object diagnostic

This diagnostic changes the function object passed to the public initializer’s nine real CUDA occupancy queries. Condition A queries the actual runtime-derived function; condition B queries a separately loaded function from the exact same cubin bytes. Both conditions load and retain all nine duplicate driver modules in the same order. Each of eight fresh processes measures one shape using the original five arms, six rounds, three placements, 30 warmups immediately before each arm, and 30 cold-L2 strict-CUPTI complete-operator samples per arm. Process order is ABBA for M32 and BAAB for M8, totaling **7,200 samples**. This is diagnostic evidence; it does not append an acceptance matrix or replace history124.

All nine returned occupancies and the derived configuration match across the eight processes: large/shared and routed variants have 5 active blocks per SM, medium variants 9, small/tiny variants 4, and the large tail 8. The observed device has 152 SMs. The actual source/public first-use calls also match in Runtime API, grid, block, dynamic shared memory, stream, cooperative attribute, all nine tensor pointers and all five scalar arguments. The initializer’s mapped embedded bytes are bound to its actual library, kernel and function handles; the B query target is separately bound to those same cubin bytes.

Instrumentation temporarily replaces selected entries in the loaded libraries’ Global Offset Table (GOT), which stores resolved function pointers. It saves each actual pointer and page permission, forwards real initialization and untimed first-use launches to those saved targets, and restores every pointer and permission before any warmup or measurement. Pointer restoration is checked again after timing. No launch handle or argument is substituted, and no optional symbol-lookup result is installed as a target. The exact recovered public library and measured source copies remain unchanged.

| M | Process condition | Source µs | Public µs | Source/public paired median | Source/public negative rounds | All ten comparisons’ negative rounds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 1: A: actual function | 7.320000 | 7.296000 | 0.998903508772 | 3/6 | 4 |
| 32 | 2: B: duplicate function | 7.312000 | 7.312000 | 1.000028730128 | 3/6 | 5 |
| 32 | 3: B: duplicate function | 7.288000 | 7.296000 | 1.000000000000 | 2/6 | 3 |
| 32 | 4: A: actual function | 7.312250 | 7.296000 | 1.002200222685 | 1/6 | 1 |
| 8 | 5: B: duplicate function | 4.976000 | 5.000000 | 0.995176749459 | 5/6 | 6 |
| 8 | 6: A: actual function | 4.960000 | 4.968000 | 0.998397435897 | 3/6 | 4 |
| 8 | 7: A: actual function | 4.976000 | 4.960000 | 1.004833523493 | 1/6 | 3 |
| 8 | 8: B: duplicate function | 4.984000 | 4.960000 | 1.003236245955 | 1/6 | 2 |

The eight source/public process medians retain **three below 1, one equal to 1, and four above 1**; their 48 rounds retain **19 below-one records**. Across all ten comparison families, the diagnostic retains **28 below-one round records**: those 19 plus nine v53-source/v53-public records. All original ratios and samples remain retained. The table uses medians of paired round ratios, which need not equal the quotient of the separately summarized durations.

The intervention shows no consistent direction across repeated processes: M32’s two A ratios bracket the B ratios, and M8’s source/public ratios increase across its BAAB sequence. There is no demonstrated causal improvement or qualifying export result. These comparisons apply to the same-node context with nine duplicate modules loaded in both conditions. Fresh-process address/layout differences and between-process temporal variation remain; this experiment does not isolate occupancy-query effects in the ordinary production loading context and cannot turn cross-node changes into paired improvements. History124’s **NOT_MET** conclusion remains unchanged.

Two earlier attempts failed before any measured sample. Attempt `c9a88a3ce96548292cb37edc` found an unresolved AArch64 PLT resolver in a saved GOT slot. The next attempt first loaded the exact libraries with immediate symbol relocation in both conditions; attempt `a956548a4598f7afa4cbe3b7` then failed in optional handle-scoped symbol lookup despite the relocated source import. The completed v3 diagnostic records that optional lookup failure separately while continuing to validate, forward and restore the actual resolved GOT targets. Both failed receipts and logs are retained; neither is a performance sample or acceptance result.

| Diagnostic attempt | Measured samples | Harness runtime s | Managed command elapsed s | Physical turnaround s |
| --- | ---: | ---: | ---: | ---: |
| v1: unresolved binding, failed | 0 | — | 29.188243 | 29.289211 |
| v2: optional symbol lookup, failed | 0 | — | 23.878840 | 23.977805 |
| v3: eight completed processes | 7,200 | 120.997262 | 131.896797 | 131.999401 |

The successful process receipt is `c858d654f4f9b2caf8696ccb`. Runtime includes the eight-process diagnostic harness; physical turnaround covers submission through actual terminal completion. These intervals remain separate from the GPU kernel-duration denominator.

## Historical actual v5 original grouped matrix: complete, performance NOT_MET (history124)

This matrix uses the original grouped sampling loop: six rounds, three shared buffer placements, forward/reverse arm order on alternating rounds, 30 external warmups immediately before each arm, then one 30-sample strict cold-L2 CUPTI call. Each sample still contains the complete operator. The actual recovered public DSO is loaded directly; Stock, v53 source and the sealed v53 actual-public denominator remain unchanged.

| M/k/BM | Stock µs | v53 source µs | v53 public µs | Source µs | Public µs | Stock/public | v53 public/public | Source/public |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/8/16 | 9.872000 | 8.568000 | 8.536000 | 7.328000 | 7.320000 | 1.348302717955 | 1.162838141286 | 0.998910675381 |
| 128/8/16 | 11.496000 | 10.704000 | 10.680000 | 9.520000 | 9.464000 | 1.218547573878 | 1.127757163471 | 1.005916449444 |
| 8/10/8 | 11.984000 | 6.072000 | 6.080000 | 4.992000 | 5.000000 | 2.392954842398 | 1.208931987464 | 0.996794838869 |
| 128/10/8 | 13.952000 | 9.584000 | 9.472000 | 8.248000 | 8.224000 | 1.698422823338 | 1.150870970638 | 1.001932383385 |
| 512/10/8 | 16.400000 | 12.696000 | 12.688000 | 7.728000 | 7.704000 | 2.132931047869 | 1.663555352352 | 1.001039501040 |
| 16384/10/8 | 124.136250 | 75.752500 | 68.568000 | 28.440000 | 28.336000 | 4.379204936751 | 2.440317997642 | 1.002251028294 |

All six shape medians and all 36 rounds exceed 1 against the original Stock and v53 acceptance references. Source/public has **two negative medians, ten negative rounds and six equal rounds**; M32 is `0.9989106753812637` and M8 is `0.9967948388687425`. Across all ten comparison families, the matrix retains **18 negative rounds, three negative medians, nine equal rounds and zero equal medians**. It is complete and nonqualifying; negative results are not relabeled as noise, rounded into equality or hidden by the other speedups.

| M/k/BM | Source/public below-one rounds | Equal rounds |
|---|---:|---:|
| 32/8/16 | 3 | 1 |
| 128/8/16 | 1 | 0 |
| 8/10/8 | 4 | 1 |
| 128/10/8 | 1 | 1 |
| 512/10/8 | 0 | 3 |
| 16384/10/8 | 1 | 0 |

The first invocation sealed the first four shapes, then failed at M512 preparation because its private prebuilt cache omitted existing shared-softmax modules used by the original medium/large occupancy queries. The repair copied those two existing cache modules, preserved the four sealed rows exactly, and measured only M512 and M16384. Frozen source, public code, original timing loop and gates were unchanged. The failed raw snapshot, both actual receipts and both runner identities remain retained. The completed aggregate contains 5,400 samples, 7,920 kernel activities and twelve large-component sidecars. The initial design note used three descriptive label aliases; actual recorded fixtures retain the original labels.

Aggregate runner runtime is **63.031634 s** (initial **32.915938 s**, repair **30.115697 s**). Physical turnaround from initial submission to repair completion is **418.898923 s**: **90.808881 s** active managed execution, **0.197818 s** admission and **327.892224 s** between-invocation wait/preparation. The individual attempts took **49.516116 s** and **41.490583 s** physically. CUPTI call host duration is **3.963454 s**, separate from the GPU denominator.

## Actual v5 precision grouped matrix: complete, performance MET (history125)

This was one prespecified fresh measurement of the unchanged frozen source and actual v5 public library, with **3,000 samples in a single grouped CUPTI call per arm per round**. It retained all six original shapes, all five original arms, six rounds, three shared buffer placements, forward/reverse arm order, and 30 external warmups immediately before each arm. Every sample retains its own cold-L2 flush and the original complete-operator correlated GPU kernel-duration sum. The large source and export calls both include producer and tail. No threshold, denominator, code, host launch policy, or device cubin was changed for this run.

The source/public and v53 native libraries were the same retained artifacts identified above; the measured public library is the recovered original DSO, without a clone or rebuild. The completed run retains **540,000 samples, 792,000 kernel activities, 180 exact arm sidecars and twelve large-component sidecars**. This fresh precision run provides the passing result; earlier negative samples and nonqualifying matrices remain unchanged historical evidence. The result is not attributed to a new implementation repair or to a comparison between different nodes.

| M/k/BM | Stock µs | v53 source µs | v53 public µs | Source µs | Public µs | Stock/source | Stock/public | v53 source/source | v53 public/public | Source/public |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/8/16 | 9.888000 | 8.576000 | 8.512000 | 7.360000 | 7.360000 | 1.349351639969 | 1.349351639969 | 1.165217391304 | 1.160869565217 | 1 |
| 128/8/16 | 11.456000 | 10.704000 | 10.688000 | 9.504000 | 9.504000 | 1.211765304047 | 1.211765304047 | 1.126262626263 | 1.124579124579 | 1 |
| 8/10/8 | 11.888000 | 6.112000 | 6.096000 | 5.024000 | 5.024000 | 2.378179576741 | 2.378179576741 | 1.216560509554 | 1.212903225806 | 1 |
| 128/10/8 | 13.984000 | 9.472000 | 9.504000 | 8.160000 | 8.160000 | 1.713725490196 | 1.716815442130 | 1.179508732605 | 1.157285916810 | 1 |
| 512/10/8 | 16.448000 | 12.640000 | 12.624000 | 7.744000 | 7.712000 | 2.123966942149 | 2.132780082988 | 1.661157024793 | 1.668049792531 | 1.004149377593361 |
| 16384/10/8 | 124.264250 | 76.448500 | 68.960000 | 28.032000 | 28.032500 | 4.432095462329 | 4.432016380711 | 2.727186786530 | 2.456497175141 | 1 |

All original Stock and v53 acceptance comparisons exceed 1 for all six shape medians and all 36 rounds. Source/public has **zero negative medians, five equal medians and one positive median**; its rounds retain **four below 1, 24 equal to 1 and eight above 1**. Across all ten comparison families, there are **six negative rounds, 35 equal rounds and 319 positive rounds; zero negative medians, six equal medians and 54 positive medians**. Equal source/public medians are actual recorded values, not rounded negative values. The two additional negative rounds and one additional equal median belong to v53 source/v53 public.

All six negative round ratios in this precision matrix are disclosed here and retained in the full historical tables:

| Shape | Comparison | Round | Paired ratio |
|---|---|---:|---:|
| 8/10/8 | Frozen source / actual public | 2 | 0.9936708860759493 |
| 128/10/8 | v53 source / v53 public | 2 | 0.9966329966329966 |
| 128/10/8 | Frozen source / actual public | 4 | 0.9961240310077519 |
| 128/10/8 | v53 source / v53 public | 5 | 0.9966329966329966 |
| 16384/10/8 | Frozen source / actual public | 4 | 0.9988558352402745 |
| 16384/10/8 | Frozen source / actual public | 6 | 0.9999643277565727 |

A CPU-only archive transfer overlapped the final large-shape round, during Unix seconds **1789503030.542–1789503045.640**, despite the measurement requesting an isolated GPU step. No concurrent GPU step is claimed. The effect of that CPU/I/O overlap on kernel durations was not independently measured; all affected samples remain included.

Harness runtime was **507.309725 s**, managed command elapsed time **532.575777 s**, and physical turnaround **532.667990 s**. CUPTI-call host duration was **408.182385 s**, separate from the GPU kernel-duration denominator. The original correctness and real-request evidence below applies to the unchanged code; this performance run introduces no new numerical oracle.

## Correctness and real requests

- Frozen source: four original GPU checks and four original eager/capture/replay ABI fixtures passed. The registered regression measured 0.0070 ms against the original 0.0250 ms threshold. Source synccheck and racecheck each timed out at their separate hard 20-second limit and remain **SKIPPED**.
- Actual v3 and v4 public tests: **24 passed each**, with original inputs and precision thresholds. V4 embeds all nine cubins byte-identical to v3. Separate actual v2 synccheck and racecheck passed with zero reported errors/hazards; unchanged host execution and all nine identical cubins establish applicability through v4. No v3 or v4 sanitizer rerun is claimed.
- Actual v5 public tests: **24 passed**, zero failures/errors/skips. Separate actual v5 synccheck and racecheck passed with zero errors/hazards at 9.631 s and 10.270 s, respectively. The original source timeout outcomes remain SKIPPED.
- Source real-model pair: Qwen3-Next-80B-A3B-Instruct-FP8, revision `c5f5f263bdd5cc134092897864e8905d8fe7b928`, original 1,314 GSM8K requests. Source **1,265/1,314 (96.270928%)** versus Stock **1,260/1,314 (95.890411%)**; **+0.380518 percentage points**. There are 12 improved and seven regressed question IDs, 318 identical raw predictions, and no empty/invalid answer or request failure. Both accuracy thresholds pass: accuracy ≥0.95 and candidate-minus-baseline ≥−0.005.
- The source audit verified request/scoring identity, per-question output accounting, four-rank build/execution identity, and 1,668 runtime geometry records spanning M=8–16,384 across eager and graph replay. Four sampled CUDA profiles and 688 source-shape links establish their sampled scope. The reused baseline retains its failed outer orchestration receipt and successfully audited inner baseline seals.
- Historical v4 real-model pair: actual public export **1,260/1,314 (95.890411%)** versus its paired Stock baseline **1,258/1,314 (95.738204%)**, **+0.152207 percentage points**. The independent audit passes the original accuracy and sampled execution/profile gates; nine improved and seven regressed question IDs, 301 identical raw predictions, and no empty/invalid answer or request failure. It records 1,696 actual export runtime geometries across eager, prefill-graph and decode-graph execution. This remains the historical v4 result and uses its own paired Stock baseline.
- Actual v5 real-model pair: public export **1,264/1,314 (96.194825%)** versus its own Stock **1,260/1,314 (95.890411%)**, **+0.304414 percentage points**. Ten question IDs improve and six regress; 322 raw predictions are identical, 992 differ, and 25 extracted answers differ. There are no empty/invalid answers, request failures or sampler retries. Both original accuracy gates pass (accuracy ≥0.95 and candidate-minus-Stock ≥−0.005); the independent audit also passes its sampled kernel-profile gate.
- The source/public comparison verifies all 1,314 original inputs, final and attempted SDK/wire requests, sampler parameters and server protocol. Inputs match with zero recorded mismatches. Source **1,265/1,314** to actual public **1,264/1,314** is **−0.076103500761 percentage points**; five question IDs improve and six regress. Full text is identical for 397 questions and differs for **917**; extracted answers are identical for 1,295 and differ for **19**. Exact output equality is therefore not claimed. All per-question outputs and original scores are retained without rescoring or a new numerical threshold.
- Source and public model runs retain their actual SGLang revisions, `54495612f52364d3c87ad4659291e4f7dbb53b21` and `5407ec1a7dfee227a408702addcc15007ec7f126`, respectively. Their audited input/server protocols match; those runtime revisions are disclosed rather than assumed equal. The Stock and export arms within the v5 model pair use the same public-run SGLang revision. The public run records **1,644** actual runtime geometries spanning **M=8–16,384**, with eager, prefill-graph and decode-graph execution on four ranks. Four sampled CUDA profiles observe the actual exported producer, tail and small-kernel symbols. Live process-map/native identity evidence binds the isolated model DSO to the validated implementation. These profiles establish sampled execution, not every kernel on every request.
- Shared model weights at the original FP8 revision are read in place. Model cache isolation preserves the public measurement cache; the earlier library collision remains disclosed above.

## Runtime and physical turnaround

| Completed work | Runtime s | Physical turnaround s |
|---|---:|---:|
| Frozen source six-shape pair | 166.116 | 187.709 |
| Actual export v1 pair | 104.996 | 115.419 |
| Actual export v3 pair | 129.288 | 350.675 |
| Actual export v4 pair harness / enclosing native-gates-pair work unit | 324.684 | 409.214 |
| Actual v4 model harness | 1211.830 | 1228.401 |
| Historical actual v5 grouped pair, both invocations | 63.032 | 418.899 |
| Actual v5 precision grouped pair | 507.310 | 532.668 |
| Actual v5 model harness | 1201.370 | 1220.589 |
| V5 audit + source/public comparison CPU leaf | 31.464 | 31.553 |
| v5 generation/native managed command | 68.193 | 68.282 |
| v5 public-tests managed command | 38.161 | 38.268 |
| Actual v5 synccheck | 9.631 | 24.407 |
| Actual v5 racecheck | 10.270 | 32.599 |
| Source real-model harness | 913.175 | 952.679 |
| Independent source-model audit | 38.849 | 45.708 |
| v3 generation, AOT lookup and native-build command | 86.762 | 86.851 |
| v3 public-tests and sanitizer-applicability command | 37.730 | 37.834 |
| Actual v2 synccheck | 8.481 | 23.809 |
| Actual v2 racecheck | 8.863 | 32.895 |

Runtime is the stated harness/command or sanitizer interval. Physical turnaround is submission to actual terminal completion and includes startup, queueing, isolation, and cleanup. Within the v3 commands, native compilation took **36.310 s**; the complete native-build script took **42.403 s**, and public tests took 27.317 s. Source model API phases were 62.151 s versus baseline 62.039 s; these are separate from GPU kernel performance.

## Completion status

V4 formatting checks passed with clang-format 19.1.1, Ruff 0.12.8, and Mypy 1.17.1. The v4 actual native comparison passed for all nine cubins and all 24 original public tests passed; the measured v4 source/export gate is nonqualifying, while its separate model audit passes. V5 actual public-model accuracy, input, execution and sampled profile audits pass. Its historical original grouped matrix remains NOT_MET; the prespecified precision grouped matrix passes all original source/export performance gates with the same code. V4 native compilation took 36.688 s within a 45.124 s build script; public tests took 26.436 s. The completed v4 pair harness took 324.684 s within its 409.115 s native/gates/pair managed command, 409.214 s physical turnaround. V5 native compilation took 37.096 s within a 47.011 s build script; its tests took 18.650 s within a 27.452 s gate script. The table reports the enclosing generation/build and test command runtimes separately. The historical v4 model API phases took 55.190 s for export and 59.701 s for Stock; neither is the kernel-performance denominator. Formatting took 7.640 s within a 24.252 s managed command; physical turnaround was 24.342 s.

- The actual precision grouped six-shape five-arm matrix is complete and passes all original numeric gates. Source/public medians are **1, 1, 1, 1, 1.004149377593361, 1**. Its four negative source/public rounds and every earlier nonqualifying matrix remain disclosed.
- V5 native identity, 24 original public tests and both actual sanitizer checks are complete as described above. Any implementation change after this packet requires its actual applicable evidence to be reflected here.
- Actual v5 public-model metrics and original accuracy/profile gates are complete. Input equality is verified; the 19 extracted-answer and 917 full-text differences from frozen source remain explicitly disclosed.
- Historical original grouped runtime/physical turnaround: **63.031634/418.898923 s**. Precision grouped runtime/physical turnaround: **507.309725/532.667990 s**. Actual public-model runtime/physical turnaround: **1201.370468/1220.588914 s**. Independent model audit takes **23.189944 s** and source/public comparison **0.806953 s**, within the **31.464079 s** CPU leaf and **31.552985 s** physical turnaround. Public/Stock API phases take **54.672513/58.780787 s**, separate from GPU kernel performance.
- Final PR-head CI: record required/related run URLs, conclusions, and actual skip counts in the PR description after the documentation commit. Earlier-commit CI remains historical.

No hardware speed-of-light (SOL) or hardware-limit claim is made.
