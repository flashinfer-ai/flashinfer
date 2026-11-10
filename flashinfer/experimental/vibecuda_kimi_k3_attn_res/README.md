# VibeCUDA Kimi-K3 AttnRes (experimental, unfused variant)

Experimental backend for the **original unfused** Kimi-K3 attention-residual
mixing operator (`_apply_attn_res`), extending the experimental
`flashinfer.kimi_k3_attn_res` API introduced with the CAKE backend.

- **Status:** experimental (opt-in via `backend="vibecuda"`; JIT-only; never
  selected by `backend="auto"`).
- **Owner / tracking:** FlashInfer experimental track; graduation target is
  four weeks (correctness + latency parity with the CAKE line on the fused
  contract is a prerequisite for any graduation discussion).

## Contract

Exactly the unfused operator, nothing else:

- `delta is None` (prefix never mutated),
- `block_write_idx == -1` (bank never written),
- `output_norm_weight` required (output RMSNorm always applied),
- `eps == output_norm_eps` (one epsilon for both RMSNorms; the Kimi-K3
  model uses `1e-6`),
- `qk_weight is None` (the per-channel score weight is `norm_weight` alone),
- `num_blocks` in `[1, 8]`, BF16 tensors, hidden size 7168, FP32
  accumulation, rows mixed bank-first-then-prefix with the exact reference
  epsilon placement.

Fused variants (residual delta, bank snapshot write, optional output norm,
two score weights) belong to the `cake` backend; passing them to this
backend raises `NotImplementedError` loudly.

## Layout

- `csrc/vibecuda_kimi_k3_attn_res/vibecuda_kimi_k3_attn_res_kernels_sm100.cu`
  — SM100 (GB100/GB200, CC 10.0) kernel payload + frozen production dispatch.
- `csrc/vibecuda_kimi_k3_attn_res/vibecuda_kimi_k3_attn_res_kernels_sm103.cu`
  — SM103 (B300, CC 10.3) kernel payload + frozen production dispatch.
- `csrc/vibecuda_kimi_k3_attn_res/vibecuda_kimi_k3_attn_res_binding.cu` —
  shared TVM-FFI binding (input checks, stream resolution, launch).
- `vibecuda_jit.py` — per-architecture JIT module registration.
- `vibecuda_backend.py` — validation, plan/runner, public glue.

The payloads carry persistent tcgen05/TMEM routes for the mid/large token
bands and lean CTA-per-token routes for the small bands; production route
selection is compiled into the payload and validated at `prepare` time.

## Tests and benchmark

```bash
pytest tests/experimental/test_vibecuda_kimi_k3_attn_res.py -v
python benchmarks/bench_vibecuda_kimi_k3_attn_res.py
```

Both run on SM100 (GB200) and SM103 (B300); other targets are skipped (tests)
or fail loudly (`arch_for_device` raises `NotImplementedError`).
