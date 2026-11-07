# kda_ucache_flush

KDA (Kimi K3) ReplaySSM speculative-decode verify kernel: verify T draft
tokens, append (L2-normalized key, correction `u`, per-channel cumulative
log-decay) to a 32-slot per-request ring, and fold the ring into the
checkpoint state once it holds `flush_min` entries. The KDA analogue of
`flashinfer.gdn_kernels.gated_delta_rule_mtp_ucache_flush`.

- **Owner:** ameynaik-hub
- **Tracking:** PR [#4864](https://github.com/flashinfer-ai/flashinfer/pull/4864)

The public entry point is the thin
`flashinfer.kda_decode.kda_delta_rule_mtp_ucache_flush`, marked
`@flashinfer_experimental_api`. Calling it is the opt-in (it warns once) and
needs no environment variable. It is not exported from `flashinfer` or
`flashinfer.kda_kernels`.

Limitations: SM90+; bf16 q/k/v/g/b, state pool, rings and output (fp32
`g_cache`, `A_log`, `dt_bias`); `K == V == 128`; `H == HV`; draft length
T in {4, 8}; contiguous pools; `lower_bound` in `[-80 / T, 0)`. Kernels are
compiled with `cute.compile` per process (no on-disk cache). Range checks on
`initial_state_indices`, `hist_len` and `cache_base` cost one host sync per
eager call and are skipped under CUDA-graph capture.

Tests: `tests/experimental/test_kda_ucache_flush.py`. Benchmarks:
`benchmarks/experimental/bench_kda_ucache_flush.py` (fold-rate sweep) and
`benchmarks/experimental/bench_kda_ucache_vs_recurrent.py` (A/B against the
`recurrent_kda` spec-decode verify path). Example:
`examples/experimental/kda_ucache_flush.py`.
