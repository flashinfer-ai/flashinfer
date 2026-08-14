# CAKE KDA K1 parallelism on B200 and B300

This optimization keeps the fused CAKE chunk-32 K1 and ordered K2 recurrence,
but gives under-filled B200/B300 grids additional CTAs for independent K1
chunk preparation. One owner CTA retains recurrent state and consumes chunks
in token order. Helper CTAs produce K1 operand packets out of order and publish
them through a release/acquire synchronized bounded ring mailbox. B200 packed
varlen inputs use the same protocol without a host synchronization.

## Dispatch policy

The route is intentionally conservative and falls back to the existing M64 or
M128 kernel outside measured profitable regions.

| Device | Layout | Batch-head tasks | Route | Minimum length |
| --- | --- | ---: | --- | ---: |
| B200/GB200 | fixed `B=1, H=1` | 1 | M64 C4, depth 10 | 4096 |
| B200/GB200 | fixed or packed | 1-8 | M128 C4, depth 15 | 2048 |
| B200/GB200 | fixed or packed | 9-32 | M128 C4, depth 30 | 2048 |
| B300/GB300 | fixed | 1-8 | M128 C4, depth 30 | 2048 |
| B300/GB300 | fixed | 9-32 | M128 C4, depth 45 | 2048 |
| either | other | other | exact baseline fallback | N/A |

The single-head M64 route takes precedence over the general M128 helper row.
Outside a profitable helper region, baseline M64 is selected while
`2 * batch * heads <= SM count`; otherwise M128 is the fixed-layout fallback.
Outside the enabled B200 helper region, packed varlen input falls back to M128.
For packed B200 input, the minimum length is the host-known integer average
`total_tokens / num_sequences`; dispatch never copies `cu_seqlens` to the CPU.
This deliberately conservative rule can leave a highly skewed, low-average
batch on M128, but it cannot introduce a device-to-host synchronization or
make graph capture depend on device data. Packed B300 remains on M128 until it
is independently tuned and validated. M128 helpers support fixed H=1, H=4, or
a head count of at least eight divisible by eight; packed H=1 remains on M128.
Four-head beta tiles are padded to the generated eight-head TMA box.

Each helper CTA contains five K1 preparation instances while the owner focuses
on mailbox ingress and ordered K2. M128 C4 therefore exposes 15 concurrent K1
instances instead of the baseline M128 CTA's five. The single-head M64 C4
variant has two recurrent owners and two helpers, exposing ten K1 instances
while splitting K2 across two 64-row state halves. M128 C8 can expose 35, but
cold-L2 B200 and B300 sweeps found that its larger grid costs more than the
extra K1 capacity saves. It remains a supported forced M128 benchmark
configuration, not a public dispatch choice; M64 is restricted to validated
C4 launches. Ordered K2 recurrence and the 31,520-byte packet handoff remain
serial or bandwidth limits. Mailbox depth is constrained to a multiple of the
helper instance count, so a ring slot cannot be reassigned to a different
producer generation before its current packet is consumed.

## Validation required before an upstream PR

Run on B200 and B300 with the exact pushed commit and record CUDA, PyTorch,
Python, GPU, and driver versions:

1. `pytest tests/jit/test_flash_kda_jit.py -q`
2. targeted recurrent-KDA prefill correctness, routing, non-default stream,
   and CUDA graph tests in `tests/kda/test_recurrent_kda_prefill.py`
3. focused C8, C4, M64-fallback, and M128-fallback cases under
   `compute-sanitizer --tool memcheck`
4. focused C8, C4, M64-fallback, and M128-fallback cases under
   `compute-sanitizer --tool synccheck`
5. `pre-commit run --all-files`
6. a cold-L2 CUPTI table against both M64 and M128 for every benchmark shape;
   report speedup against `min(M64, M128)`, not against only one baseline

The upstream pull request should state that the full repository test suite was
not run if validation remains targeted. External contributors also need a
FlashInfer `ci-users` member to add the `run-ci` label or comment
`@flashinfer-bot run`; draft PRs skip public CI.
