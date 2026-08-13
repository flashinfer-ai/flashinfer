# CAKE KDA K1 parallelism on B200

This optimization keeps the fused CAKE chunk-32 K1 and ordered K2 recurrence,
but gives under-filled fixed-length B200 grids additional CTAs for independent
K1 chunk preparation. One owner CTA retains recurrent state and consumes
chunks in token order. Helper CTAs produce K1 operand packets out of order and
publish them through a generation-tagged bounded ring mailbox.

## Dispatch policy

The route is intentionally conservative and falls back to the existing M64 or
M128 kernel outside measured profitable regions.

| Batch-head tasks | Route | K1 parallelism | Minimum length |
| ---: | --- | ---: | ---: |
| 1-8 | C8 owner/helper | 1 owner + 7 helpers | 2048 |
| 9-32 | C4 owner/helper | 1 owner + 3 helpers | 2048 |
| other | exact M64/M128 oracle fallback | unchanged | N/A |

M64 is selected while `2 * batch * heads <= SM count`; otherwise M128 is the
fallback. Packed varlen input and CC 10.3 continue to use the existing M128
path. The helper implementation also requires at least eight heads and a head
count divisible by eight because of the generated beta TMA layout.

Each helper CTA contains five K1 preparation instances. C8 therefore exposes
35 concurrent instances, or about seven times the K1 producer capacity of the
original five-instance M128 CTA. This is not a sevenfold end-to-end speedup:
ordered K2 recurrence and the 31,520-byte packet handoff remain serial or
bandwidth limits.

## Validation required before an upstream PR

Run on a B200 with the exact pushed commit and record CUDA, PyTorch, Python,
GPU, and driver versions:

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
