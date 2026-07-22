# sglang: paged prefill via the unified API

Target: `python/sglang/srt/layers/attention/flashinfer_backend.py`
(clone @ 977ea336c).
Scope: the **full-sequence paged prefill paths** (no radix-cache hit /
multimodal / deterministic mode).  Explicitly OUT of the v1 diff:
- the radix-extend cascade — BOTH halves.  The paged (prefix) half runs
  `causal=False` over `kv = prefix_lens` (`:1390`), and no-prefix requests
  have prefix_len == 0, i.e. zero-length KV rows — outside the v1 envelope
  (kv_len >= 1) and with backend-divergent fully-masked-LSE semantics.
  Migrating it needs either envelope support for zero rows or engine-side
  filtering; it waits for the ragged follow-up anyway.
- custom-mask paths (target-verify / multi-item) — capability axis absent
  from the POC; they pin fa2 exactly as today.

## 1. Init: unified wrapper + one resolution (token-CSR form)

```python
# FlashInferAttnBackend.__init__ (replacing the fa2 pin at :312 and the
# paged-prefill wrapper construction at :492-510)
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

## 2. Per-batch: plan directly from what the indices updater builds

One piece of NEW plumbing is required (audited: no host copy of
`paged_kernel_lens` exists at the call site today — only the GPU tensor and
`paged_kernel_lens_sum`): the scheduler carries the per-request lens on host
already, so this is a copy-forward of an existing host array, not a sync.

```python
# FlashInferIndicesUpdaterPrefill.call_begin_forward (:2034-2213)
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
+    qo_indptr_cpu=qo_indptr_cpu,               # NEW plumbing: host lens
+    kv_seq_lens_cpu=paged_kernel_lens_cpu,     # forwarded from the scheduler
+    backend=self._prefill_resolution,
+)
```

Deleted wholesale:
- the `kv_indptr` page-unit cumsum and `kv_last_page_len` construction
  (derived internally from `kv_seq_lens` — one truth, not two);
- the prefill fast path's inline host-array reconstruction (`:2172-2188`).

## 3. forward_extend: run

```python
-o = prefill_wrapper_paged.forward(q, kv_pool, ...)
+o, lse = self.prefill_attn.run(q, (k_pool, v_pool))
```

For the prefix-cache cascade, the paged half's `(o, lse)` feed `merge_state`
against the ragged half's output exactly as today (same LSE convention).

## What stays (v1 scoping, honest)

- the entire radix-extend cascade (both halves; see Scope above),
- decode wrappers + `fast_decode_plan` (decode follow-up),
- **`fast_prefill_plan` (`:187-297`)**: it is CUDA-graph *replay* machinery
  (asserted cuda-graph-only, writes the wrapper's pinned capture buffers for
  EAGLE draft-extend) — unified's capture mode is not wired, so it stays
  until that follow-up.  What unified DOES already give natively is the
  sync-free host-fed plan for the eager path (machine-checked by the
  zero-sync guard in the engine-shaped test).
- custom-mask paths (target-verify / multi-item),
- SWA: the paged-only SWA half maps to `window_left` (its own Resolution,
  since window is in the config key); the ragged-extend SWA half remains on
  the custom-mask fa2 path.
