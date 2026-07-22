# vLLM: paged prefill via the unified API

Target: `vllm/v1/attention/backends/flashinfer.py` (clone @ 75698e60).
Scope: the **prefill half** only — decode routing, cascade, fp8/nvfp4
quantized paths keep their current code (they share `use_trtllm_attention`
with decode and wait for the decode follow-up).

## 1. Init: replace prefill routing with a pinned resolution

```python
# FlashInferMetadataBuilder.__init__ (near the q_data_type decisions, ~:672)
+from flashinfer.attention.unified import resolve_paged_prefill
+
+self._prefill_resolution = resolve_paged_prefill(
+    device=self.device,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.model_dtype,          # bf16/fp16 envelope of the POC
+    page_size=self.page_size,
+    kv_layout=get_kv_cache_layout(),   # "HND" on SM100, "NHD" elsewhere
+    causal=True,
+    need_lse=self.use_dcp,             # DCP consumes LSE
+)
```

Deletes on the prefill side: the `prefill_use_trtllm` predicate
(`use_trtllm_attention(...)` call at `:984-996`), the `page_size >= 128`
force (`:981-983`), and the init-time capability guesswork this resolution
replaces.  `resolve()` also reports exclusion reasons, replacing the logged
"reverting to FlashInfer" strings.

## 2. build(): one plan replaces both prefill metadata builds

```python
# in build(), prefill branch (replacing TRTLLMPrefill :1175-1207 AND the
# FIPrefill CSR construction :1208-1269 for the prefill slice)
+prefill_attn = UnifiedPagedPrefill(self.device)
+prefill_attn.plan(
+    qo_indptr=qo_indptr[prefill_start:] - qo_indptr[prefill_start],
+    kv_seq_lens=seq_lens[prefill_start:],
+    block_tables=block_table_tensor[prefill_start:],
+    page_size=page_size,
+    max_q_len=max_q_len,                     # already host ints
+    max_kv_len=max_seq_len,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type_prefill,
+    kv_layout=get_kv_cache_layout(),
+    causal=causal,
+    window_left=self.window_left,            # uniform per batch, as today
+    sm_scale=self.sm_scale,
+    return_lse=self.use_dcp,
+    qo_indptr_cpu=qo_indptr_prefill_cpu,     # mirrors vLLM already owns
+    kv_seq_lens_cpu=seq_lens_cpu[prefill_start:],
+    backend=self._prefill_resolution,
+)
```

Deletes: `_compute_flashinfer_kv_metadata` for the prefill slice (the
numpy-indptr + Triton `_copy_page_indices_kernel` CSR expansion, `:902-940`),
the trtllm `cum_seq_lens_kv` GPU cumsum (`:1180-1194`), the per-build
`q_data_type` mutation (`:1036-1038` — dtype is pinned by the resolution),
and both prefill dataclasses' metadata duplication.

## 3. forward(): one run replaces the trtllm/FI fork

```python
-# trtllm_batch_context_with_kv_cache(...)  (:1745-1763)
-# / prefill_wrapper.run(...)               (:1653-1660)
+out, lse = attn_metadata.prefill_attn.run(
+    prefill_query, (kv_cache_k, kv_cache_v), out=output[num_decode_tokens:]
+)
```

LSE is base-2 packed `(tokens, h)` fp32 from every backend, so the DCP
merge path needs no per-backend shim — and the "Trtllm does not support
returning LSE" DCP fork (`vllm/utils/flashinfer.py:431-437`) becomes
unnecessary *for prefill* (it was already stale: trtllm prefill exposes LSE).

## What stays (v1 scoping, honest)

- decode routing + `use_trtllm_attention` (shared with decode),
- the artifactory HTTP probe (backs decode gates),
- cascade (fa2 MultiLevelCascade hardwired),
- fp8-Q / fp8-KV / nvfp4 paths (outside the POC dtype envelope),
- spec-decode reorder policy.
