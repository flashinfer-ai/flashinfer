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
| B200/GB200 | fixed or packed | 1-8 | C4, depth 15 | 2048 |
| B200/GB200 | fixed or packed | 9-32 | C4, depth 30 | 2048 |
| B300/GB300 | fixed | 1-8 | C4, depth 30 | 2048 |
| B300/GB300 | fixed | 9-32 | C4, depth 45 | 2048 |
| either | other | other | exact baseline fallback | N/A |

M64 is selected while `2 * batch * heads <= SM count`; otherwise M128 is the
fixed-layout fallback. Outside the enabled B200 helper region, packed varlen
input falls back to M128. For packed B200 input, the minimum length is the
host-known integer average `total_tokens / num_sequences`; dispatch never
copies `cu_seqlens` to the CPU.
This deliberately conservative rule can leave a highly skewed, low-average
batch on M128, but it cannot introduce a device-to-host synchronization or
make graph capture depend on device data. Packed B300 remains on M128 until it
is independently tuned and validated. The helper implementation also requires
at least eight heads and a head count divisible by eight because of the
generated beta TMA layout.

Each helper CTA contains five K1 preparation instances. C4 therefore exposes
15 concurrent helper instances in addition to the owner's original five. C8
can expose 35 helper instances, but cold-L2 B200 and B300 sweeps found that its
larger grid costs more than the extra K1 capacity saves. C8 remains a supported
forced benchmark configuration, not a public dispatch choice. Ordered K2
recurrence and the 31,520-byte packet handoff remain serial or bandwidth
limits.

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
