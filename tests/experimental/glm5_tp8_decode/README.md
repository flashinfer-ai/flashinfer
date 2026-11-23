# GLM-5 TP8 decode tests

These tests exercise the public `flashinfer.glm5_tp8_decode` API. The numerical
references and the deterministic synthetic data live in this directory; the
tests do not import a serving runtime, a model graph or a checkpoint.

## Requirements and CI coverage

- `test_cpu.py` needs no GPU; run it in a FlashInfer development
  environment. It checks the numerical references, the packed weight
  layouts, the checkpoint-layer slicing and input validation of the weight
  packers, and the workspace geometry.
- The seven other `test_*.py` files need one node with eight B300 GPUs
  (SM103, 148 SMs each) and eight processes started by `torchrun`. Without
  that platform they skip in the session-scoped `collective` fixture.

A lane that runs plain pytest, without an eight-rank `torchrun` on eight B300
GPUs, executes only `test_cpu.py`; when this directory is targeted, the seven
GPU files are collected and reported as skipped. GPU validation is run with
the commands below.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/experimental/glm5_tp8_decode/test_cpu.py
```

## GPU tests

Run the commands below from the repository root, one at a time, on a node with
eight B300 GPUs. The operators use persistent peer-waiting kernels, so no other
process may run these kernels on the same GPUs at the same time. The fixture
uses eight processes, Gloo metadata exchange and FlashInfer's public workspace
(TorchDistBackend and symmetric device memory). With eight visible devices, it
maps logical rank `r` to device `(r + 3) % 8` to catch code that confuses the
logical rank with the CUDA ordinal. If each process sees only one GPU, it uses
ordinal zero. The CUDA modules are compiled on first use, so the first run of
each operator family takes longer.

Smoke test:

```bash
torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
  tests/experimental/glm5_tp8_decode/test_moe.py -k 847223
```

`test_lazy_first_exchange.py` checks that the first collective exchange after
workspace warmup can be a graph replay: compilation and kernel-function
initialization happen before capture. It must run in a fresh set of processes
with lazy module loading, and it skips otherwise:

```bash
CUDA_MODULE_LOADING=LAZY \
  torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
  tests/experimental/glm5_tp8_decode/test_lazy_first_exchange.py
```

All other GPU test files, each in its own set of eight processes:

```bash
for name in moe moe_exceptional attention combined selector long_context; do
  torchrun --standalone --nproc-per-node=8 -m pytest -q -x -p no:cacheprovider \
    "tests/experimental/glm5_tp8_decode/test_${name}.py" || break
done
```

## Hangs and failures

Every GPU test arms a `faulthandler` watchdog that prints the stacks of all
threads after 240 seconds; it does not terminate the process. Gloo collectives
time out after 240 seconds, so a rank that blocks makes its peers fail.

`test_moe_exceptional.py` waits at most 60 seconds for its GPU work, polling a
CUDA event instead of synchronizing. If completion is not confirmed on every
rank, the test fails on all ranks and the remaining GPU tests in the same
process are skipped, because the device state is unknown. Session teardown then
skips `torch.cuda.synchronize`, which would block behind the unfinished work,
but still destroys the process groups, so pytest exits with a failure status
and `torchrun` stops any rank that is still running.

## Synthetic weights

`data.py` builds synthetic checkpoint layers under the Hugging Face GLM-5
tensor names and the tests pack them through the public API. Every rank draws
the same attention layer (all 64 heads) on the CPU and packs it with
`device` set to its GPU; the references read the rank's checkpoint-named
shard. For MoE, each rank fills only its own intermediate channels of the
full-shape expert tensors (the rest stay zero and are never read) and the
replicated norm, router and correction bias are broadcast from rank zero.

## MoE coverage

The MoE matrix uses four ordinary weight seeds and one cancellation fixture;
each covers both q1 and q4. q1 checks the normalized input, legal
approximate-sigmoid routing, BF16 intermediate values and Down from the actual
intermediates. q4 checks router-dot rounding bounds, enumerates legal ambiguous
top-eight sets and carries the analytical sigmoid error through the expert
contributions. Both use per-128 FP8 activation quantization. Outputs must be
within two BF16 ULPs, where the ULP is taken at `max(|reference|, 1/16)`. The
cancellation fixture distinguishes one final BF16 rounding from a second
rounding before the residual addition.

RMSNorm evaluates `x * (gamma * rstd)` before BF16 rounding. The q4 routed
normalization and shared-expert normalization assign input pairs to lanes
differently, so shared scratch cannot provide the router's exact normalized
input. The router reference independently reconstructs its FP32 square-sum
order and enumerates the BF16 rows produced by the documented `rsqrtf` accuracy
range. CUDA 13.2 lists two FP32 ULP from the correctly rounded reciprocal square
root; this is a [tested accuracy bound, not an exhaustive guarantee](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html).
One scalar reciprocal square root must explain **all 256 logits in a token
row**. Each candidate row is checked with the dot-product bound
`gamma_64 * sum(abs(normalized_x * router))`, with
`gamma_64 = 64 * 2^-24 / (1 - 64 * 2^-24)`; the bound is not enlarged. The
RMSNorm output is checked separately with the same two-ULP metric. CPU negative
controls reject BF16-rounded logits, mixed per-expert norm candidates and an
incorrect normalized input, and cover FP32 spacing at exponent edges.

`test_moe_exceptional.py` covers finite inputs whose intermediate arithmetic
overflows to NaN: zero input with a very large finite gamma, overflowing Gate
with zero Up, and negative router overflow giving zero selected sigmoid mass.
q1/q4 calls must complete, publish no reserved readiness markers, and produce
the expected NaN classification. The same captured input addresses then
produce the zero-expert residual, and separate plans with ordinary weights
reuse the workspace without a reset and pass all stage and wire checks.

## Attention coverage

Attention covers q4 full and shared indexers and q1 with an accepted selection,
dense/sparse boundaries at 0, 31, 63 and 2043 through 2049, and contexts up to
196608 tokens with ordinary and concentrated indexer-cache families. Each stage
is checked against a reference evaluated on the kernel's actual stage inputs,
with BF16 rounding at the same boundaries: projections within two BF16 ULPs,
attention within one BF16 ULP of each row's maximum, exact zero rows, and a
rotation-invariant pair metric for RoPE. Causal indexer logits must satisfy
`max|actual - reference| / max|reference| <= 1e-5`. Sparse selection must
contain the **exact** top-2048 set of the actual logits; threshold ties need no
particular order. Dense queries use their causal prefix.

Both repeated full-indexer replays run every numerical and communication check,
and their sorted selected-ID sets must match exactly. Publication slot order is
unspecified, so full-indexer outputs are not required to match bit for bit
across calls. Shared replays with the same fixed selection order are checked
bit for bit. The q4 output is also compared with a reference that reproduces
the kernel's FP32 `o_proj` accumulation order, within two BF16 ULPs with a
relative floor of 1/16; an end-to-end FP64 reference is not used as a
pass/fail criterion.

`test_selector.py` feeds finite adversarial FP32 logits to the selector in
`long_topk.cuh` through a small test launcher (`selector_binding.cu`). Its
reference ranks full unsigned key32 values and then lower IDs, keeping
signed-zero ordering without packing unsigned composites into signed int64. It
covers FP16-bin collisions, negative and wide scores, subnormals, high-ID ties,
all-equal pool overflow, group spikes and partial causal tiles. Sixteen cases
are distributed over the eight GPUs; both the direct full-row mode and the
grouped mode use the operator's 16*C-byte scratch size between guard bytes,
including graph replay with changed inputs and positions. The launcher calls
the library selector, not a copy of it; these cases complement the stage checks
in `test_attention.py` and `test_long_context.py`.

`test_long_context.py` covers the extended capacity with the same stage, exact
causal top-k, cache and rank-ordered wire checks at 512K and 1M, including odd
live pitches, the `202756 -> 202757 -> 202756` switch between the
default-length and long-context dispatch paths, the last legal q4 position
`1052668`, and a short query afterward. Two concentrated indexer-cache seeds
also reach each long allocation boundary. Graphs keep their bound addresses
while inputs and positions change. Fixed-input full replays compare sorted
selected sets; full outputs are not compared bit for bit across calls.
Explicit q1/q4 accepted selections exercise IDs above 18 bits and the final
logical cache row. Invalid logical positions are rejected despite the eight
extra storage-padding rows. The KV and indexer caches are synthetic: they are
filled with deterministic random rows instead of being produced by a prefill.
`test_cpu.py` checks the extended limit `1052672` and the default capacity
`202752`.

## Both families

For both families, independently decoded current-generation wire records are
summed in logical-rank order in FP32, followed by residual addition and one BF16
rounding. Every rank must match this result bit for bit. The tests also cover
input, weight, cache and output guards; q1/q4 alternation; changed inputs,
positions and selections without recapture; interleaved independent workspaces;
destroy/recreate; collective host-validation failures; and preservation of
generations across request reset. A side-stream lifetime test drops a temporary
MoE plan and its input/weight owners, churns their allocation sizes on the
launching stream, then checks the completed output. MoE scratch is poisoned with
finite values only once, and tests never refill its NaN sentinels or reset
state/inboxes between calls. `test_combined.py` also launches the q1 and q4
attention -> MoE chains with `enable_pdl=False`; with fixed selections, they
must pass the same stage checks and match the default launch bit for bit.

Attention-only `FFFF` and `FSSS` chains delay rank zero before the second layer
to exercise exit credits. These four launches run back to back. Afterward, all
observed outputs must agree across ranks and the final call passes the full
stage and exact-wire checks. The full indexer's actual ID order is copied and
replayed through independently validated shared mode; every observed full
output must match that reconstruction bit for bit. Shared outputs must match the
validated fixed-selection output, and full-indexer sorted sets must match their
validated selection.

These are standalone operator tests. They do not validate a complete serving
model and do not measure performance; operator timings come from
`benchmarks/experimental/bench_glm5_tp8_decode.py`.
