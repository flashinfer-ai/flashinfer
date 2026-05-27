# moe_monokernel standalone device tests

Compilable, self-contained sanity checks for the low-level PTX wrappers that
back the TMA + WGMMA up-projection path (spec:
`.kiro/specs/tma-wgmma-weight-load/`).

These tests sit alongside the production kernel code but build **outside** the
main CMake target so you can iterate on a single wrapper without linking the
whole `moe_monokernel` translation unit.

## Requirements

- NVIDIA Hopper GPU (SM90+; H100 / H200).
- CUDA toolkit ≥ 12.0 (for SM90a PTX and `mbarrier.*` / `cp.async.bulk.tensor.*`).

## Tests

| Test                      | Covers                                          | Spec refs        |
| ------------------------- | ----------------------------------------------- | ---------------- |
| `mbarrier_sanity_test.cu` | `mbarrier_init`, `fence_mbarrier_init_release_cluster`, `mbarrier_arrive_expect_tx`, `mbarrier_try_wait_parity` with `tx=0` | R3.1, R3.2, R3.5 |

## Build + run

From the repository root:

```bash
nvcc -std=c++17 -arch=sm_90a -O2 \
     -I vllm/csrc/moe/moe_monokernel/src \
     vllm/csrc/moe/moe_monokernel/tests/mbarrier_sanity_test.cu \
     -o /tmp/mbarrier_sanity_test
/tmp/mbarrier_sanity_test
```

Expected output on an H100 / H200:

```
mbarrier_sanity_test OK: barrier completed after 1 poll(s) (flag=0xC0DE0001).
```

Exit codes:

- `0` — test passed.
- `1` — wrapper behaved incorrectly (barrier never completed).
- `2` — CUDA runtime error (printed to stderr).
- `77` — device is pre-SM90; the test is skipped rather than failed.
