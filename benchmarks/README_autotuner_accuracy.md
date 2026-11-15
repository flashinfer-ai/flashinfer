# Autotuner selection-quality experiments

`bench_autotuner_accuracy.py --graph-selection-study` compares actual fresh-cache
v2 searches under `auto`, `eager` and `cuda_graph` measurement policies. It then
measures the selected tactic in a separately captured graph, with untuned-default
graphs captured before and after tuning as controls. This is a single-operation
experiment, not a serving-throughput or full-model accuracy benchmark.

For an FP8 GEMM component on a supported GPU:

```bash
python benchmarks/bench_autotuner_accuracy.py \
  --graph-selection-study --ops bmm_fp8 --grid 144 --n 4096 --k 1024 \
  --graph-policies auto eager cuda_graph --no-cold-l2 \
  --profile-inputs retained --warmup 5 --tuner-repeat 10 \
  --seeds 6 --oracle-rounds 7 --validation-replays 16 \
  --atol 0.1 --rtol 0.02 \
  --out fp8-graph-selection.json
```

The GPU architecture is detected normally; select a GPU with
`CUDA_VISIBLE_DEVICES`. Build dependencies and optional native wheels must match
the tested source. The report includes the source commit, hashes of the harness
and tuner, library versions, GPU, input metadata, arguments and raw timings.
Keep the native build/runtime logs with the report when qualifying a backend.

## What changes between the comparison arms

The measurement objective changes; warmup, repeat, input policy and requested L2
setting stay fixed. The default `-1` competes through the actual v2 `choose_one`
path. A real cache hit is required before a choice is scored, so an all-failed
search cannot pass as a successfully tuned default. Per-candidate records show
the effective capture/L2 settings and profiling policy.

`--profile-inputs retained` fixes the profile to the captured concrete shape and
returns the fixture's retained tensors through `inputs_pre_hook`. This removes
input synthesis and bucketing as variables in the objective comparison. It does
not qualify the framework's production bucket mapping or synthesized probes.
`--profile-inputs synthesized` preserves the operation's original configuration
instead; record that distinction when comparing results.

Each trial records the actual `input_shapes` and mapped `cache_profile`;
each candidate measurement records these for its profiling tensors too. A
bucket label can exceed the measured shape (for example, an endpoint at 144
can map to bucket 256). Do not interpret the cache label as a measured shape,
or assume a backend's tactic index identifies the same native algorithm at
every shape in that bucket.

Each trial uses a fresh tuner/cache; trials share a process, runner objects and
retained inputs. Policy order rotates across trials. Repeat the command in fresh
processes as well when studying startup history, keeping fixture inputs and
sources fixed and rotating `--graph-policies` order between processes. Do not
interpret fresh-cache trials as fresh-process trials.

## Independent graph validation

Each validation graph captures one runner invocation. It contains no timing
events, delay or cache-flush operation. CUDA events outside the graph time replay;
an unmeasured `torch.cuda._sleep` before the start event lets the host queue the
short replay window. `--prequeue-cycles` records that delay (default 10 million
cycles). Use enough delay for the host to enqueue the window, and check sensitivity
to larger values before drawing conclusions about very small kernels.

Validation uses repeated hot-L2 replay. It does not simulate the full model's L2
working set, eager tail or graph partitioning. Default-before, selected and
fresh-default graphs are interleaved in rotating order. The report keeps all
samples and reports:

- `selected_over_default_after`: selected latency relative to the fresh default;
- `retained_over_fresh_default`: whether the pre-tuning control differs from the
  post-tuning control;
- actual winners, profile counts and effective profiling policies across trials.

Numerical checks run after explicitly replaying each graph, before another graph
can overwrite a shared output buffer. The built-in FP8 builder also checks the
untuned output against FP32 multiplication of its quantized inputs. Tolerances
are explicit (`--atol`, `--rtol`), and reference errors are recorded. The example
allows FP8 accumulation and BF16 output rounding, including small outputs where
a relative tolerance alone is insufficient. Other fixtures must choose suitable
tolerances and references. Capture or validation errors stop the study. Do not continue
ranking candidates in a process with a potentially damaged CUDA context.

A lower isolated-operator latency does not establish an overall model win.
Mixed eager/graph deployment, TP/EP and backend-internal fallback need separate
framework validation. Runner names and tactic representations identify the
autotuner's choice, not a verified native kernel trace. Do not introduce a global
winner margin from one batch.

## Framework-owned fixtures

`--fixture package.module:build` imports a function taking the parsed arguments
and returning a zero-argument call to the public API. The call must reach exactly
one `choose_one` operation. The fixture owns weights, activations, quantization,
scales, layouts and any routing inputs; the harness retains the resulting runner
inputs. It does not load checkpoints or invent framework metadata.

The returned callable can expose a zero-argument `.reference` function for a
numerical reference. Without one, the report explicitly says that only the
untuned default was checked. Composite multi-operation fixtures are rejected.

A fixture can also provide `.prepare_validation(output, inputs)` to refresh its
owned output before each final correctness replay, after all timing samples.
The built-in FP8 fixture fills its separately allocated output with NaNs, so a
no-write tactic cannot pass using another graph's stale output. Custom callbacks
must preserve read operands and workspace, including any aliases. Without a
callback, the report records `validation.output_refresh = "unavailable"` and
cannot rule out this stale-output case.

For example, a framework can load its captured layer inputs once in `build`,
return a call through its real quantizer/linear path, and supply a suitable
reference. Timing still covers the captured tunable component, not all work in
that wrapper.

The existing exhaustive-oracle mode remains available without
`--graph-selection-study`. Explicit measurements now use
`prepare_tactic_profile` / `profile_tactic` under the requested context, so graph,
eager and L2 overrides are applied. A missing/disabled CUPTI route raises instead
of mislabeling event fallback as CUPTI evidence. Exhaustive sweeps record these
unavailable measurements as error rows and continue to the next shape; those
rows must not be scored as valid results. Graph-study failures still stop the
study.
