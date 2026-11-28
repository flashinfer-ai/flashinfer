# Offline SM120 NVFP4 selector tuning

The selection order is explicit overrides, a matching user tuning entry,
qualified built-in model rules, then the existing general/compatibility selector.
A conflicting explicit override skips the tuned entry rather than silently
combining an unmeasured config with that override. Redundant overrides preserve
the entry. Nondefault JIT/build execution retains compatibility selection.

GLM5.3-flash is the first model targeted by the built-in qualification campaign.
DSV4-flash retains its existing calibrated defaults. The model names are labels;
selection uses H, combined I, top-k, expert count, token capacity and execution
scope. Model-specific changes do not change numerical permission.

The qualified GLM rule changes K2 from stage2 to stage3 at **T16384 only**,
with H4096/combined-I4096/top-k8/E288, same-NUMA EP4/TP1/DP1, 110 SMs,
`owner_bf16`, no supplied host counts and the original complete anchor config.
Two independent seed/order pairs measured P50 changes of -3.157% balanced,
-1.690% power-law and -1.315% zero-owner, with P90 improving in each scenario;
100 mixed-owner dynamic replays passed per-replay native and transport checks.
The campaign is `SM120-SELECTOR-TUNING-20261105`, based on `22c08eb/ABI2737`.

GLM T512 N64 is deliberately not a default: despite the prior balanced gain,
two zero-owner comparisons regressed P50 by a geometric mean of 45.631%.
Nearby balanced-only probes do not qualify an interval. Host-count-aware
selection retains its existing path rather than assuming balanced routing.

## Run

Use the same CUDA/CuTeDSL/NVSHMEM environment as `mega_runner`. The v1 tuner
supports SM120, same-NUMA EP4, TP1/DP1 and the standard complete Green Graph
pipeline. `--intermediate` is the combined gate/up width; 4096 means 2048 after
SwiGLU. It is not the framework's post-SwiGLU intermediate dimension.

```bash
python tune_selector.py \
  --num_topk 6 --hidden 4096 --intermediate 4096 --num_total_experts 256 \
  --ep 4 --devices 0,1,2,3 --tokens auto \
  --numerical_contract owner_bf16 \
  --budget_seconds 3600 --work_dir tuning_dsv4_flash \
  --output dsv4_flash_tuned.json
```

Omit `--numerical_contract owner_bf16` to retain ordinary top-k association.
Owner BF16 explicitly permits per-owner BF16 partial sums. The tuner preserves
the baseline's RLC choice, so changing reduction association is not part of v1
search. Existing byte-exact native validation is separate from FP64 arithmetic
accuracy; this command does not run a new FP64 sweep.

`--plan_only` prints the workload without acquiring GPUs. `--tokens 512,16384`
limits the capacities. `auto` visits the existing 13 sweep points first, followed
by neighboring N64/N128 tile boundaries derived from the model's density.
`--max_candidates` bounds the search at each capacity. A short search tests
local tile/stage changes as well as its incumbent; larger searches expand into
the full legal catalog with geometry/SM/stage diversity.

The time budget includes launches, compilation, reference checks and dynamic
validation. It may expire before all capacities complete. Only fully qualified
entries are exported; the campaign reports remaining/incomplete work and exits
with code 2. An empty entries list is a valid result: no measured improvement
passed the gates, so normal selection remains active. Existing output files and
campaign directories are preserved; use new paths for another campaign.

The tuner rejects foreign GPU compute processes and stops its own workers when
interference or a timeout occurs. `--allow-resident-pid PID` explicitly permits
an existing resident with a pinned process identity. Its sampled compute
activity invalidates the measurement. Raw process monitoring, PCLK and power
remain in the campaign; clocks are not locked or normalized.

## Search and qualification

Screen configurations with full K1+K2+K3 event timing, including one independent
native QMMA reference check per rank outside timing. Confirm the best screen
candidates on balanced, power-law and zero-owner routes. Each route uses two
independent holdout seeds and reversed candidate/baseline order. A qualified
entry requires at least 1% geometric-mean process-P50 improvement on every
route, no mean P90 regression, and no observed P50/P90 regression in either
confirmation pair. These are conservative adoption gates, not confidence bounds
or proof of a global optimum. Noisy results retain the incumbent.

Inputs and final output hashes must match within each pair. Native QMMA byte
equality and transport validation are recorded separately; an unavailable
direct-store transport oracle remains `NOT_CHECKED`. Winners then undergo 100
changing-route replays with a native reference and transport check after every
replay. These diagnostic timings never qualify performance.

`--routing_trace /absolute/path/routes.npz` adds the supplied routing workload
to the required confirmation routes. The NPZ keys are decimal token capacities,
each containing an int64 `[EP, T_per_rank, topk]` array of global expert IDs.
For a single capacity, an NPY array is also accepted. IDs must be in range and
distinct within each token. This is routing-only replay with synthetic weights
and activations; it is not a model-quality evaluation.

## Load

The runner accepts a tuning file:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.mega_runner \
  --num_tokens_per_rank 16384 --hidden 4096 --intermediate 4096 \
  --num_topk 8 --num_total_experts 288 \
  --enable_static_expert_shape --split_launch green_graph \
  --numerical_contract owner_bf16 --tuning_config glm_tuned.json
```

The runner prints selection/fallback reasons and checks rank agreement before
launch. A framework can load once during setup and pass the immutable object:

```python
from moe_sm120_nvfp4_split import load_tuning_config, hardware_identity
from moe_sm120_nvfp4_split.api import select_compile_spec

tuning = load_tuning_config("glm_tuned.json")
hardware = hardware_identity()  # after selecting this rank's CUDA device
spec = select_compile_spec(
    problem=problem,
    ep_same_numa_peer_count=3, ep_cross_numa_peer_count=0,
    num_sms=110, sm_min_partition=8, sm_partition_alignment=8,
    numerical_contract="owner_bf16",
    tuning_config=tuning, hardware=hardware,
)
```

Frameworks remain responsible for agreeing token capacity, execution options
and selected configuration across EP ranks. `explain_selection(...,
tuning_config=tuning, hardware=hardware)["profiles"]` reports the selected
source and reasons. The API itself does not initialize CUDA or read files when
these objects are passed; load and hardware detection happen during setup.

V1 matches exact capacities and complete structure/topology/contract values.
Unmeasured capacities, incompatible hardware/toolchain identity, stale source
fingerprints or ABI, invalid configs and conflicting overrides fall back.
No nearest-neighbor lookup, interval interpolation or GPU route readback occurs.
The file contents identify the lookup cache; reloading a replaced file creates
a new key. Hold the immutable object through graph capture and execution.

The fingerprint covers package executable Python/CUDA sources, excluding tests
and the built-in model-policy table: changing which complete config the default
chooses does not alter the execution of a separately tuned complete config.
ABI remains 2737 because this change does not modify kernels or workspace layout.
Ordinary compile keys already include the complete selected config.

The JSON and campaign logs are data artifacts; no Python source is rewritten,
no global installation is changed, and no commit/push is performed by the tuner.

## Cached setup and FlashInfer integration

Final selection and immutable compile specs use bounded process-local caches.
Configuration contents, hardware identity, numerical contract and explicit overrides
are included in the keys. Load a tuning file once before setup/capture.

The FlashInfer integration exposes `tuning_profile` on the NVFP4 backend config
and imports qualified offline documents through its TuningProtocol/v2 managed
store. See FlashInfer `docs/design_docs/sm120_nvfp4_tuning.md`. The legacy v1
JSON cache does not persist these records. Prepared layers retain their profile
snapshot; recreate the layer to adopt a changed profile or cache context.
