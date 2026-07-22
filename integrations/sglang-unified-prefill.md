# sglang: paged prefill via the unified API

Target: `python/sglang/srt/layers/attention/flashinfer_backend.py`.
Scope: the paged prefill wrappers (no-prefix / multimodal / target-verify
paths, and the **paged half** of the radix-extend cascade).  The ragged half
of the extend cascade keeps `BatchPrefillWithRaggedKVCacheWrapper` until the
ragged follow-up; `merge_state` keeps working because the unified LSE is the
same base-2 packed contract fa2 produces today.

## 1. Init: unified wrapper + one resolution (token-CSR form)

```python
# FlashInferAttnBackend.__init__ (replacing the fa2-pinned wrapper, :303/:443-457)
+from flashinfer.attention.unified import (
+    UnifiedPagedPrefill, resolve_paged_prefill,
+)
+
+self._prefill_resolution = resolve_paged_prefill(
+    device=self.device,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type,
+    page_size=1,                      # token-granular slots, as today
+    kv_layout="NHD",
+    causal=True,
+    need_lse=True,                    # merge_state consumes LSE
+    kv_input_form="page_indices",     # the flat kv_indices sglang builds
+)
+self.prefill_attn = UnifiedPagedPrefill(self.device)
```

`backend="fa2"` pinning becomes unnecessary: at page_size=1 the
dense-needing backends are capability-excluded automatically and the
resolution explains why; on arches where more CSR-native backends appear,
sglang inherits them with zero code change.

## 2. Per-batch: plan directly from what the indices updater already builds

```python
# FlashInferIndicesUpdaterPrefill.call_begin_forward
 kv_indices = ...  # UNCHANGED: triton gather from req_to_token (pool-owned)
-kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)   # DELETED
-kv_last_page_len = ...                                           # DELETED
-wrapper.begin_forward(qo_indptr, kv_indptr, kv_indices,
-                      self.kv_last_page_len[:bs], ...)
+self.prefill_attn.plan(
+    qo_indptr=qo_indptr,                       # same preallocated buffer slice
+    kv_seq_lens=paged_kernel_lens,             # the masking truth, directly
+    kv_page_indices=kv_indices,                # over-allocated tail is fine
+    page_size=1,
+    max_q_len=max_extend_len,                  # host ints sglang carries
+    max_kv_len=max_kv_len,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type,
+    kv_layout="NHD",
+    causal=True,
+    return_lse=True,
+    qo_indptr_cpu=qo_indptr_cpu,               # from seq_lens_cpu, host origin
+    kv_seq_lens_cpu=paged_kernel_lens_cpu,
+    backend=self._prefill_resolution,
+)
```

Deleted wholesale:
- the `kv_indptr` page-unit cumsum and `kv_last_page_len` construction
  (derived internally from `kv_seq_lens` — one truth, not two);
- **the `fast_prefill_plan` monkeypatch (`:178-289`)**: its entire reason to
  exist — a sync-free, replay-safe plan fed from host arrays — is the
  native contract now (machine-checked by the zero-sync guard in
  `test_unified_prefill_engine_shapes.py::test_sglang_shaped_prefill_token_csr`);
- `global_override_indptr_cpu` for the prefill path (the `*_cpu` mirror
  parameters are the official version of that hack).

## 3. forward_extend: run

```python
-o = prefill_wrapper_paged.forward(q, kv_pool, ...)
+o, lse = self.prefill_attn.run(q, (k_pool, v_pool))
```

For the prefix-cache cascade, the paged half's `(o, lse)` feed `merge_state`
against the ragged half's output exactly as today (same LSE convention).

## What stays (v1 scoping, honest)

- the ragged wrapper for extend's new-token self-attention,
- decode wrappers + `fast_decode_plan` (decode follow-up),
- custom-mask paths (target-verify / multi-item) — capability axis not in
  the POC; they pin fa2 exactly as today,
- the SWA dual page table (windowed models at page_size=1 keep fa2 via the
  window capability; the translated-table trick is engine policy).
