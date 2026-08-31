# Hopper SM90 MXFP4 x FP8 MegaMoE autotune

This document is the handoff for the MXFP4 fused and Green-Context split
autotune integration.  Candidate discovery and formal performance use only
`benchmarks/bench_moe_ep_sm90_mega.py`; the backend-local tuner reuses the
same deterministic input recipe and production preprocessing/staging, but it
does not define a second performance benchmark.

## Routing profiles and frozen tuning data

MXFP4 tuning is partitioned by an explicit routing-workload identity:

- `block_permutation_v1` is the historical padded/truncated benchmark
  workload.  It remains the default for public production configs and for
  every old call that omits `routing_profile`, so existing applications and
  frozen tables do not change silently.
- `published_exact_balanced_v1` is the published comparison workload.  It is
  the default for the `flashinfer.moe_ep.tune --dtype sm90_mxfp4` CLI and is
  selected by the official benchmark's
  `--routing-mode published_exact_balanced` spelling.

Candidate, heuristic, online-autotune, cache, session, and workspace-pool
resolution are all profile-scoped.  An exact request never consumes a legacy
candidate or a profile-less historical MXFP4 cache entry.  The legacy public
`MXFP4_TUNING_PROVENANCE[execution_mode]` mapping is unchanged; profile-aware
callers use `hopper_mxfp4_tuning_provenance(...)` or
`MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE`.

Both H200 searches cover tokens/rank `8, 32, 64, 128, 256, 512, 1024, 2048`
at world size 4, hidden 7168, post-SwiGLU intermediate 3072, 384 experts,
top-k 6, and clamp 10.  Discovery uses one fresh `torchrun` for each
token/tactic/repeat, benchmark cooldown 5 seconds, warmup 3, and 10 timed
iterations.  The official candidate score is

~~~text
MAX_over_ranks(MEDIAN_over_that_rank's_timed_iterations(latency))
~~~

The historical `block_permutation_v1` frozen tables live in
`flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/shim/mxfp4_tuner.py`:

- fused: eight per-token winners, deduplicated union of eight tactics,
  manifest SHA-256
  `455cf75bdd0c0011184ee5a3f48eab9ac80782b4824562cd796005887a19d1cf`;
- split: eight per-token winners, deduplicated union of eight tactics,
  manifest SHA-256
  `1c350f333d365ef6284b23e1604faaa3388ba00f3cb82c63f686515778700f93`.

The published-exact profile is a separate table in the same module:

- fused: seven-tactic deduplicated union, external artifact SHA-256
  `62733c7605f7233ac81c341084e0d589f4a91ca3f1aaaf1fac0660f7d1842a61`;
- split: eight-tactic deduplicated union, merged external artifact SHA-256
  `094d840c579a7331439d1acd50690909ad2c88e6085253326c6f2d98ddad248a`,
  candidate-union SHA-256
  `210adb840e66c0f44949ea866a8bbfa9f5b2b3835ee8992e57b5edb75f8b9321`.

The module validates every tactic and reconstructs/hash-checks compact runtime
manifests at import time.  Returned tactics are fresh dictionaries, so callers
cannot mutate the frozen source of truth.The historical 1024/2048 winners retain their telemetry warnings; clock
locking is only a target and the manifest telemetry remains authoritative when
a run observes `sw_power_cap`.

## Runtime behavior

`Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig.knobs` has three modes within
the config's selected `routing_profile`:

- an explicit dictionary bypasses lookup (formal candidates and cache winners
  are complete; split explicit dictionaries must be complete);
- `"auto"` performs one collective online sweep on the first forward;
- `None` performs only a persistent-cache lookup, then falls back to the
  manifest's ceil-token-bucket heuristic.

For the formal shape, online tuning tests the complete profile/mode-specific
offline union (legacy fused/split: eight/eight; published exact: seven/eight).
Custom shapes first filter that union for compatible MMA-K divisibility.
Online tuning does not sleep, spawn fresh
processes per candidate, or expand the legal search domain.  For each candidate
all ranks in the backend's exact EP process group synchronize, run three
warmups, time ten launches, take their own median, and all-reduce those medians
with `MAX`.  The process-group size is checked against the session's EP world
size before any candidate runs.  Each phase's success is reduced with `MIN` in
the same group, so a rank-local exception makes every EP rank reject or fail
the same phase without taking a different barrier branch.  The minimum MAX
score wins.  Every rank applies the same winner and only EP rank 0 writes the
cache.

The EP process group remains the authority when EP size is one inside a larger
distributed job; online tuning does not silently switch to the global default
group.  Cache lookup, manifest fallback, and online/offline tuning also fail
closed unless the active device is exactly a standard `NVIDIA H200`, compute
capability 9.0, with 132 SMs.  In particular, an H200 NVL or a different SM90
GPU cannot consume these frozen H200 tactics merely because it is Hopper.

Fused applies candidate fields through its lazy-compile frontend.  Split
cannot mutate a captured fixed-pointer session: every candidate therefore
owns fresh symmetric buffers, K1/K2 compiled roles, Green contexts, graph
executables, SM partition, counter bank, and graph variant.  After selection,
the final fresh winner session is transferred to the caller workspace; the
next backend launch captures it, and later forwards replay it.

The offline entry points are:

~~~bash
torchrun --standalone --nproc-per-node=4 -m flashinfer.moe_ep.tune \
  --dtype sm90_mxfp4 --execution-mode fused \
  --routing-profile published_exact_balanced_v1 \
  --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \
  --max-tokens 8 32 64 128 256 512 1024 2048

torchrun --standalone --nproc-per-node=4 -m flashinfer.moe_ep.tune \
  --dtype sm90_mxfp4 --execution-mode split \
  --routing-profile published_exact_balanced_v1 \
  --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \
  --max-tokens 8 32 64 128 256 512 1024 2048
~~~

The MXFP4 CLI defaults are
`--routing-profile published_exact_balanced_v1`,
`--fp8-scale-mode mxfp4_hybrid`, canonical `--gate-up-clamp 10`,
`--warmup-iters 3`, and `--timed-iters 10`.  Passing a different clamp is
explicit and produces a different cache key.  MXFP4 rejects
schedule/base-knob/skew/nondeterministic flags so the CLI cannot silently
create a different candidate or data domain.

## Persistent cache schema and isolation

The shared JSON file remains schema version 1.  One entry has these problem
key fields:

| Field | MXFP4 meaning |
| --- | --- |
| `device` | CUDA product name, for example `NVIDIA H200` |
| `dtype` | versioned numerical/layout/execution/SM identity below |
| `fp8_scale_mode` | fixed `mxfp4_hybrid` |
| `world_size` | EP world size |
| `hidden`, `intermediate` | logical model dimensions |
| `num_experts`, `topk` | global experts and routing width |
| `gate_up_clamp` | resolved gate/up clamp; distinct values never cross-match |
| `max_tokens` | compile-time tokens/rank bucket |
| `routing_profile` | `block_permutation_v1` or `published_exact_balanced_v1` |

GPU class is separated by `device`; SM architecture and the complete MXFP4
weight/activation/layout ABI are encoded in `dtype` and compile identity.  The
two exact cache dtype identities are:

~~~text
sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_humming_v1_fold_m64_k128_gateup8_packedk2_residual64_swapab_fused
sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_humming_v1_fold_m64_k128_gateup8_packedk2_residual64_swapab_green_split_v1
~~~

The `w_..._a_...` segments make the MXFP4 E2M1/K32 weight format and FP8 E4M3
per-token/full-hidden activation format explicit.  The suffix fixes fused
versus Green-split execution, while the rest fixes the Humming layout ABI.
These identities cannot match ordinary FP8, the superseded
implicit-activation identity, or each other; there is no cross-format or
cross-mode fallback.  Lookup requires every problem field above to match, then
uses the exact token bucket when present, the smallest recorded bucket above it
otherwise, or the largest recorded bucket below it.  Rank 0 upserts the exact
`max_tokens` entry.

The JSON schema version remains 1 and the routing field is append-only.  A
profile-less historical MXFP4 entry denotes `block_permutation_v1` and can
match only a legacy MXFP4 request.  It never matches
`published_exact_balanced_v1`.  Profile-less FP8 entries retain their original
behavior, and FP8 production config/backend semantics do not gain a routing
profile axis.

The cached `knobs` value is the complete winner tactic, not merely a tactic ID.
A fused value contains exactly tile, cluster, group hint, scheduling stages,
ping-pong, load balance, token-back, accumulator mode, swap-AB, and
standalone-reduce selection.  A split value contains exactly K1/K2 tiles,
clusters, group hints, scheduling stages, SM counts, counter-bank count, graph
variant, and IKET selection.  Tuples are serialized as JSON lists and restored
to tuples on lookup.  Unknown, partial, cross-mode, and stale cached
dictionaries fail closed.

Candidate resolution is shape-aware before JIT or session allocation.  Fused
MMA K must divide both hidden and intermediate dimensions; split K1 MMA K must
divide hidden and split K2 MMA K must divide intermediate.  An incompatible
cached winner is rejected instead of being replaced silently.  If a frozen
token-bucket heuristic winner is incompatible with a smaller custom shape, the
resolver takes the first legal tactic in the stable mode-specific candidate
union; an empty legal set is an error.

Cache writes are safe across concurrent tuner processes: the writer takes an
exclusive `flock` on the sibling `.lock` file, reloads while holding the lock,
replaces only the matching exact-bucket entry, `fsync`s a temporary file, and
publishes it with atomic `os.replace`.  Recording remains best-effort; a write
failure warns and does not invalidate the already selected in-process winner.
For schema-v1 compatibility, a legacy entry with no `gate_up_clamp` field is
interpreted as clamp `None`; it can match only an unclamped request and never
the canonical MXFP4 clamp-10 key.

The cache problem key chooses one winner; the full tactic value becomes the
compile/session identity.  Fused compile keys include all inherited frontend
configuration plus the Humming format/layout identity.  Split configuration
is immutable and contains every K1/K2 geometry, stage, SM partition, counter,
graph, clamp, and fixed-pointer axis.  Split and `knobs="auto"` workspaces are
not pooled; a `knobs=None` fused workspace pool key first resolves the cache or
heuristic so a cache-winner change cannot reuse an old compiled workspace.
