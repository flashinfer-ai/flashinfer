# kda_output_only_umma

tcgen05 / UMMA (TMEM) kernel for the KDA (Kimi K3) output-only, frozen-state
decode: the outputs of T <= 16 speculative tokens from a read-only state pool,
plus the vLLM RecoverSSM verify caches. A backend of
`flashinfer.kda_kernels.kda_decode_wy_output_only`; it takes the WY kernel's
argument list and contract unchanged.

- **Owner:** ameynaik-hub
- **Status:** experimental
- **Tracking issue:** none yet (this PR is the tracker)

## Opt in

- `kda_wy_output_only(..., backend="umma")` (dense `[B, T, ...]` inputs, optional
  bf16 corrections + kg cache) and `kda_recoverssm_verify(..., backend="umma")`
  (the vLLM drop-in: packed tokens, ragged lengths, null slots, fp32 slot-indexed
  corrections, raw k|g cache, padded pools). Naming the backend is the opt-in. An
  `ExperimentalWarning` is emitted once per entry point (shared with automatic
  selection: an explicit call after an automatic one does not warn again).
- `backend="auto"` (also `recurrent_kda(disable_state_update=True)`) selects it
  only with `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`, at T <= 8 and where the
  WY kernel would otherwise run.

## Limitations

SM10x only (tcgen05); K == V == 128; bf16 inputs and state pool (as the WY
kernel); T in [1, 16]. B200, H = 12, ncu: faster than the WY kernel at T <= 8
(1.03-1.25x dense, 1.14-2.02x drop-in); at T = 16 about even (dense 0.96-1.16x,
0.96-0.98x at B >= 512; drop-in 0.94-1.06x), so automatic selection stays at
T <= 8. Kernels are JIT-compiled per process.

Shared with the WY kernel (not checked; same behaviour on both backends unless
noted):

- Per-channel decay: |cumulative log-gate| over the window must stay below ~88
  (`exp(-cum)` overflows fp32 beyond it and the output becomes NaN). The Kimi K3
  lower-bound gate with |lower_bound| * T <= 80 is always inside; the precomputed
  and softplus gates are not bounded.
- q / k elements above ~1.8e19 overflow the fp32 L2-norm sum of squares.
- Drop-in: every sequence length must be <= `spec_query_len` (the WY kernel writes
  past the sequence's cache rows; this kernel processes the first
  `spec_query_len` tokens). Non-null `state_indices` must be `< num_blocks` and
  unique; out-of-range slots write out of bounds and duplicate slots leave the
  caches undefined.
- Dense mode: a negative `initial_state_indices` entry reads a zero state (as the
  WY kernel; the grouped recurrence reads slot 0).

Tests: `tests/experimental/test_kda_output_only_umma.py` (re-runs
`tests/kda/test_kda_output_only.py` on this backend). Benchmark:
`benchmarks/experimental/bench_kda_output_only_umma.py`. Example:
`examples/experimental/kda_output_only_umma.py`.
