# kda_output_only_umma

tcgen05 / UMMA (TMEM) kernel for the KDA (Kimi K3) output-only, frozen-state
decode: the outputs of T <= 16 speculative tokens from a read-only state pool,
plus the vLLM RecoverSSM verify caches. A backend of
`flashinfer.kda_kernels.kda_decode_wy_output_only`; it takes the WY kernel's
argument list and contract unchanged.

- **Owner:** ameynaik-hub
- **Status:** experimental

## Opt in

- `kda_wy_output_only(..., backend="umma")` (dense `[B, T, ...]` inputs, optional
  bf16 corrections + kg cache) and `kda_recoverssm_verify(..., backend="umma")`
  (the vLLM drop-in: packed tokens, ragged lengths, null slots, fp32 slot-indexed
  corrections, raw k|g cache, padded pools). Naming the backend is the opt-in and
  warns once.
- `backend="auto"` (also `recurrent_kda(disable_state_update=True)`) selects it
  only with `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`, at T <= 8 and where the
  WY kernel would otherwise run.

## Limitations

SM10x only (tcgen05); K == V == 128; bf16 inputs and state pool (as the WY
kernel); T in [1, 16]. Faster than the WY kernel at T <= 8 (B200, H = 12:
1.05-1.28x dense, 1.12-1.89x drop-in) and slower at T = 16 (0.84-0.97x). Kernels
are JIT-compiled per process.

Tests: `tests/experimental/test_kda_output_only_umma.py` (re-runs
`tests/kda/test_kda_output_only.py` on this backend). Benchmark:
`benchmarks/experimental/bench_kda_output_only_umma.py`. Example:
`examples/experimental/kda_output_only_umma.py`.
