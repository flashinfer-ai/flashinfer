# FlashInfer Attention API Fragmentation — Synthesis (post-PR #3921)

One verification note up front: two survey reports contradicted each other on CUTLASS/trtllm-gen LSE base. I re-read the tests: `tests/attention/test_blackwell_fmha.py:54` returns `lse_ref * math.log2(math.e)` and compares against `BatchPrefillWithRaggedKVCacheWrapper(backend="cutlass").run(..., return_lse=True)` (:156-171) — so the cutlass path is **base-2** at the Python surface, same as fa2/fa3 (`state.cuh:45`, `cascade.cuh:42`) and trtllm-gen (validated against the fa2 wrapper LSE, `test_trtllm_gen_attention_prefill.py:183`). The claim that cutlass/trtllm are base-e was wrong. **The only LSE-base outliers are cuDNN (raw natural-log Stats, unverified by any in-repo test) and fmha_v2 (raw `(max, sum_exp)` pairs).** This materially simplifies the consolidation story: base-2 `(tokens, h)` fp32 is already the de-facto standard everywhere except cuDNN/fmha_v2.

---

## 1. Fragmentation matrix (differences only)

| | **fa2/fa3 wrappers** (Batch{Prefill,Decode}) | **cudnn native** (prefill/decode fn) | **trtllm-gen** (context/decode/ragged one-shots) | **mla** (wrapper + one-shots) | **fmha_v2** | **xqa / cute-dsl** |
|---|---|---|---|---|---|---|
| **A1 seqlen/index** | token-unit `(b+1,)` qo_indptr + **page-unit** kv_indptr/indices/last_page_len; **consumed on CPU at plan** (hidden `.cpu()`, prefill.py:2285) | per-seq lens **`(b,1,1,1)` GPU** + `(b+1,)` **element-unit** offsets (post-#3921: opt-in token-unit) + host max ints; decode: offsets **dead** (WAR, decode.py:95-97) | **triple-redundant**: per-seq `(b,)` GPU + token `(b+1,)` cum_seq_lens GPU + dense block_tables + host maxes, all required (prefill.py:4440-4446) | wrapper: token qo_indptr + page kv_indptr + per-seq `kv_len_arr` (dual); one-shots: per-seq lens + block_tables + optional `cum_seq_lens_q` | seq_lens `(b,)` + cum `(b+1,)` + host ints; deepseek variant: fully static, no seqlen tensors | xqa: per-seq `(b,beam)` **uint32** + page table, **no indptr anywhere**; cute-dsl: token indptrs + host maxes (D2H sync if omitted, fmha.py:427) |
| **A2 KV layout** | paged 4/5-D, NHD **default**, combined-or-tuple; or ragged THD | ragged THD 3-D or paged **HND-per-page** 4-D; separate k/v only | paged only, **HND default** (NHD auto-transposed, NVFP4 forces copy); `[pages,1\|2,…]` | latent ckv/kpe split (wrapper) vs combined-576 (one-shots); no head-dim axis | 5 layouts via `input_layout` string incl. PACKED_QKV; deepseek: padded BSHD | xqa: paged NHD `[pages,ps,H,D]`; cute-dsl: ragged THD, out 32-B aligned |
| **A3 convention** | plan/run; caller float ws + internal 8 MB int ws + pinned mirror | one-shot + hidden graph-cache (key omits `scale`!); caller ws | one-shot; caller ws + zero-init counter buffer quirks | wrapper plan/run; one-shots stateless | one-shot; deepseek: `out` **required positional** | in-place `output` required, **returns None**; caller semaphores (xqa) |
| **A4 LSE** | `(tokens,h)` fp32 **base-2** | prefill: **`(b,max_s,h)` padded, natural-log Stats, untested**; decode: **none** | `(tokens,h)` base-2; `lse=` filled even when `return_lse=False` | wrapper: base-2 + `return_lse_base_on_e` opt-in (only API documenting base!); xqa/sparse: none | `(tokens,h,2)` raw `(max,Σexp)` | xqa: none; cute-dsl: log2-domain buffer |
| **A5 scales** | `sm_scale` at plan; q/k folded host-side; **v_scale = post-kernel multiply** (extra launch, prefill.py:2903); fp8 = per-head GPU tensors | bare `scale` float **baked into graph, not in cache key**; fp8 descales = `(1,1,1,1)` GPU tensors | pre-fused `bmm1/bmm2_scale`, float **or fp32 GPU tensor** (log2e folded host-side) | wrapper `sm_scale`+float `ckv/kpe_scale`; one-shots bmm1/bmm2 | host floats only; `scale_softmax` pinned 1.0 (workaround) | xqa float-or-tensor q/kv_scale; cute-dsl **floats only** |
| **A6 masking** | causal + packed custom mask + window_left + soft-cap + multi-item | causal (bottom-right) **only**; no window/mask/cap | causal+window+sinks+skip-softmax; no custom mask | causal at plan (wrapper); nothing in decode one-shots | `mask_mode` **string** incl. chunked (SM90-only) + alibi + cap | xqa: bit-packed uint16 spec-dec mask; window baked into module binary |
| **A7 CUDA graph** | `use_cuda_graph` ctor + caller-pinned buffers; plan not capturable | `is_cuda_graph_compatible` flag **ignored on FE path** (prefill.py:781); safety = pointer-stable replay | host-int maxes + GPU tensors; tensor scales exist for graph-dynamic quant | same split; autotuner captures with graphs | static host ints | variant flags select different `.so` → must be stable |
| **A9 backend select** | ctor `backend=` string; `auto` picks only fa2/fa3 | `backend="cubin"` vs import-availability | param or hardwired | `backend=` + AutoTuner.choose_one dispatch (SM100) | none (module keyed on layout/dtype) | none |

**A8 note:** constraints diverge everywhere but that's mostly kernel-imposed; standout gratuitous items: cuDNN decode **hardcodes bf16** (decode.py:118-123) and no attention entry point uses `@backend_requirement` (all manual asserts) despite the infra existing for GEMM/MoE.

---

## 2. What #3921 actually consolidates — precisely

**Consolidates:**
- **The prefill functional API only** (`cudnn_batch_prefill_with_kv_cache`): `batch_offsets_units="tokens"` accepts ordinary `(b+1,)` token-unit indptrs; `batch_offsets_o←q`, `v←k` defaults kill two of the five offset tensors.
- **Direct path** (non-paged, fp16/bf16 @ cuDNN≥9.24+FE≥1.25 / fp8 @ ≥9.25+FE≥1.27): the *same* token indptr buffer is double-bound as `cu_seq_len_q/kv` (ACTUAL_SEQ_LENS UID slots) and as ragged offsets with `set_ragged_offset_multiplier(h*d)`, `implementation=UNIFIED` pinned (both dtypes in the merged version). Zero conversion kernels. *(Merged 2026-07-16 as 81632eee; the env kill-switch was dropped before merge, `actual_seq_lens_q/kv` became omittable with tokens+indptr — derived from indptr diffs on the conversion path, ignored on the direct path.)*
- **Fallback path**: tokens still accepted; flashinfer multiplies by `h*d` on device — one extra elementwise kernel per offsets tensor per call, but the *caller-side* element-unit arithmetic disappears.
- For sglang's vision path (uniform stride), tokens mode would delete `compute_flashinfer_batch_offsets_packed` + the whole elem_per_token/pack/unpack pipeline (qwen3_vl.py:814-940). For vLLM's fp8 ViT path likewise.

**Does NOT consolidate:**
- **`actual_seq_lens_q/kv` still required, still `(b,1,1,1)`** — they drive `graph_b = actual_seq_lens_q.shape[0]` (prefill.py:161) even in tokens mode where the cu_seq_lens occupy their UID slots. The dual-representation wart survives intact.
- **Decode untouched.** `cudnn_batch_decode_with_kv_cache` has no units knob; its batch_offsets are dead code (perf WAR nulls them, decode.py:95-97), no LSE, bf16 hardcoded. Also: the generic `BatchDecodeWithPagedKVCacheWrapper` has **no cudnn backend at all** — cuDNN decode is only reachable via the standalone fn.
- **Generic wrapper migration pending.** Both `BatchPrefillWith{Paged,Ragged}KVCacheWrapper` cudnn branches still pass element-unit offsets and still enforce the poisonous split validation `q.numel()==qo_indptr[-1]` for cudnn vs `q.size(0)` for everyone else (prefill.py:2683-2696, 3727-3740). `backend="cudnn"` remains *not drop-in* with other backends of the same wrapper.
- **Paged excluded** from the direct path (asserted non-paged) — the paged cudnn branch stays on the legacy graph.
- **vLLM bf16 ViT unrepresentable in tokens mode**: its element offsets encode interleaved-QKV with a 3× V stride and TP-sharded hidden size (mm_encoder_attention.py:285-316); a single uniform `h*d` multiplier can't express it. Element mode must remain for non-contiguous layouts.
- **LSE untouched**: default alloc still padded `(b,max_s,h)`, still raw Stats, base still unverified (the new test compares cuDNN-vs-cuDNN only).

**Net:** #3921 fixes the single most alien piece of the cuDNN calling convention (element-unit offsets) at the functional layer, and — via the direct path — eliminates the conversion cost that motivated the convention in the first place. But it's step 1 of ~5: wrapper migration, per-seq-lens redundancy, paged, decode, and LSE all remain.

---

## 3. Remaining disunities, ranked by consumer pain

**#1 — Redundant simultaneous seqlen representations (partly structural, partly gratuitous).**
Every non-fa2 backend demands 2–3 forms at once: trtllm requires per-seq `seq_lens` AND `cum_seq_lens_q/kv` AND host maxes AND block_tables (prefill.py:4440-4446); cuDNN requires `actual_seq_lens` AND offsets (prefill.py:161); MLA wrapper requires `kv_indptr` AND `kv_len_arr` (_core.py:1560-1588). Consequence in the wild: vLLM adds a GPU `torch.cumsum` purely to synthesize the second form without a sync (vllm flashinfer.py:1184-1194); sglang builds page tables on-device via triton for the same reason (trtllm_mha_backend.py:72-75). *Structural core*: host `max_*_len` for graph/grid sizing genuinely can't come from device tensors; per-seq-lens-vs-indptr is a 1-kernel derivation the **library** could own instead of every consumer.

**#2 — CPU-at-plan vs GPU-at-run split (gratuitous in form, structural in cause).**
FA2/FA3 wrappers silently `.cpu()` the indptrs inside `plan()` (prefill.py:2285, decode plan 1481) because split-KV scheduling needs host values. Both frameworks independently built mirrored CPU+GPU indptr buffers plus hacks to dodge the sync: vLLM's extra pinned copy for a CUDA-graph race (flashinfer.py:725-728), sglang's `global_override_indptr_cpu` + `fast_decode_plan` monkeypatch (flashinfer_backend.py:173-175, 1361). The need for host metadata is structural; the *undocumented implicit D2H* and the absence of an official `seq_lens_cpu=`/`indptr_cpu=` parameter is gratuitous — sglang's monkeypatch is the missing API written by a consumer.

**#3 — cuDNN residuals post-#3921 (gratuitous, now cheaply fixable).**
(a) Wrapper cudnn branches still element-unit (see §2) — a one-line migration to `batch_offsets_units="tokens"` per branch, deleting the `q.numel()` validation split. (b) `(b,1,1,1)` lens shape — API could accept `(b,)` and view internally (wrappers already do the reshape at prefill.py:2792, 3830; it belongs one layer down). (c) In direct mode the lens tensor is consumed only for `shape[0]` — could be replaced by an int `batch_size`. (d) Decode: no units knob, dead offsets, no LSE, bf16 hardcode, and a graph-cache key that omits scale/block-tables-presence (decode.py:67-72) — the coarse-key + `scale`-not-keyed issue is a live correctness footgun in prefill too (prefill.py:114-129).

**#4 — LSE convention split (gratuitous at the Python layer).**
Verified de-facto standard: `(tokens,h)` fp32 base-2 (fa2/fa3/cutlass/trtllm-gen/mla-wrapper). Outliers: cuDNN `(b,max_s,h)` natural-log Stats — **and no in-repo test pins its base**; the trace template even claims base-2/(tokens,h), contradicting the API's own alloc (attention.py:2850 vs cudnn/prefill.py:694-707). Worse, the paged wrapper allocates `(tokens,h)` then hands it to a cudnn function asserting `(b,max_s,h)` (prefill.py:2740 vs cudnn/prefill.py:704) — works only by coincidence. fmha_v2's `(tokens,h,2)` raw stats is kernel-structural but post-processable. xqa/cudnn-decode/sparse-MLA have none (structural gap). No docstring anywhere except MLA's states the base. Pain: DCP/cascade-style consumers (vLLM passes `return_lse=True` for DCP) cannot swap backends without LSE shims.

**#5 — Scale/quant conventions (naming gratuitous, fusion structural).**
Three regimes: `sm_scale`+`q/k/v_scale` (wrappers), bare `scale` (cuDNN), pre-fused `bmm1/bmm2_scale` (trtllm/xqa). fp8 scale *types* split three ways: per-head GPU tensors (fa2/fa3), `(1,1,1,1)` GPU tensors (cuDNN), float-or-0-d-tensor (trtllm). The wrappers' `v_scale` post-kernel multiply (prefill.py:2903-2909) is an extra launch that cute-dsl/trtllm fuse. Consumers fold layer scales host-side differently per backend (vLLM flashinfer.py:1476-1484). Graph-dynamic (tensor) scales only exist on trtllm/xqa/cuDNN — a real capability gap for fp8-under-capture on fa-family.

**#6 — KV layout defaults + capability discovery (gratuitous).**
NHD default (wrappers, xqa) vs HND default (trtllm one-shots, cuDNN paged-per-page); silent transposes, NVFP4 contiguous-copy penalty (prefill.py:4625-4637). Zero attention APIs carry `@backend_requirement`, so vLLM probes support by reading `torch.backends.cudnn.version()` itself (vllm utils/flashinfer.py:964-993). The decorator infra exists and is used elsewhere in the repo — pure convention debt.

**#7 — Masking/window vocabulary (mostly structural, naming gratuitous).**
`causal: bool` vs `mask_mode: str` vs packed-bool custom masks vs bit-packed uint16 spec-decode masks; `window_left` frozen at plan (fa) vs run-arg (trtllm) vs `window_left+1` rename (xqa:3645) vs absent (cuDNN — genuinely lacks sliding window/soft-cap/custom masks in the SDPA graph: structural FE gap). At least bottom-right causal alignment is universal — the one axis already converged.

**#8 — CUDA-graph contract expression (gratuitous flag chaos).**
Same underlying requirement everywhere (GPU-resident varying tensors, host-int stable shapes, pointer stability), but expressed as: ctor flag + caller-pinned buffers (wrappers), an `is_cuda_graph_compatible` param that the cuDNN FE path *ignores* (prefill.py:781), no flag at all (trtllm), and compile-variant stability (cute-dsl). Nothing states the contract in one place.

---

## 4. Consolidation proposal sketch

### Unified seqlen/metadata contract (target state)
One canonical `AttnSeqInfo` accepted by every batch entry point:

```
qo_indptr:    (b+1,) int32 GPU, token units          # THE canonical form
kv_desc:      either kv_indptr(+indices,+last_page_len)  [page units, paged]
              or kv_indptr (token units, ragged)
              or block_tables (b, max_pages) dense    # backends accept ≥1, library converts
seq_lens_q/kv: (b,) GPU — OPTIONAL; derived by library (one fused diff kernel, cached
              per (buffer_ptr, version)) if a backend needs per-seq form
max_q_len, max_kv_len: host ints, REQUIRED            # the only host-side truth; kills implicit D2H
*_cpu:        optional host mirrors (indptr_cpu, seq_lens_cpu) — plan() uses them if given,
              else may sync (documented). Formalizes sglang's global_override_indptr_cpu.
```
Companion conventions: LSE = `(tokens,h)` fp32 **base-2, documented in every docstring** (already true for 4/6 families); scales = `sm_scale` float + optional `q/k/v_descale: float|Tensor` with backend-internal folding; layout via existing `kv_layout` but with a single default (NHD) and documented transpose costs; capability via `@backend_requirement` so `api.is_backend_supported("cudnn", cc)` replaces consumer version-sniffing.

### Migration order (each step independently shippable)
1. **Wrapper cudnn branches → tokens mode** (prefill.py:2814, 3853): pass `batch_offsets_units="tokens"` with the existing token `qo_indptr`; delete the element-unit exception and the `q.numel()` validation split. Makes `backend="cudnn"` drop-in for the first time. Pure Python, no cuDNN work. *Do the `(b,1,1,1)`→`(b,)` acceptance in the functional API at the same time.*
2. **LSE hygiene**: pin the cuDNN Stats base with a real reference test; convert to base-2 `(tokens,h)` at the Python layer using `batch_offsets_stats` in tokens mode (multiplier `h_qo` already wired in #3921) — this simultaneously fixes the wrapper↔functional shape mismatch. Add fmha_v2 stats→LSE post-process. Document base-2 everywhere.
3. **Consumer migration** (vLLM fp8 ViT, sglang vision): tokens mode; delete both hand-rolled element-offset pipelines. Keep element mode alive solely for vLLM's bf16 interleaved-QKV until (6c).
4. **Kill the redundancy**: accept `batch_size:int` in place of `actual_seq_lens_q` on the direct path; derive `seq_lens` from indptr inside trtllm/cuDNN entry points when omitted (cached GPU diff kernel). Add official `seq_lens_cpu=`/`indptr_cpu=` plan() params (upstream `fast_decode_plan` properly).
5. **cuDNN decode parity**: `batch_offsets_units` + cu_seq_len direct path for decode (s_q=1 under the unified engine), un-dead the ragged offsets (revisit the perf WAR), LSE (Stats output with `is_inference=False`), fp8, remove the bf16 hardcode, and **fix the graph-cache keys** (include scale, or bind scale as a device scalar) — the key fix is a correctness patch shippable today.
6. **cuDNN backend/FE work needed** (the structural tail):
   a. **Paged + cu_seq_len direct path** — unified engine consuming block_tables together with cu_seqlens, so the paged wrapper branch and the core-LLM prefill path (which is paged) get the same zero-conversion treatment; today #3921's direct mode is non-paged-only, which is exactly the part vision encoders use but LLM serving doesn't.
   b. **Decode via the unified engine** (enables step 5); plus `implementation=UNIFIED` pin exposure for `sdpa_fp8`.
   c. **Per-tensor ragged-offset stride multipliers** (non-uniform, e.g. 3×-stride V for interleaved QKV) — would retire element mode entirely, including vLLM's bf16 ViT case.
   d. FE version reach: verify which FE release actually ships `set_ragged_offset_multiplier` (the pip FE 1.25.0 in the dev venv lacks it, so the `(1,25)` gate in `_cudnn_supports_direct_seqlens` is optimistic); gate on capability-probe rather than version tuple if possible.
7. **Convention rollout**: `@backend_requirement` on attention entry points; one documented CUDA-graph contract section replacing the per-API flag zoo.

**Bottom line:** #3921 removes the *worst-in-class* outlier (element-unit offsets) at the functional prefill layer and proves the cu_seq_len direct path works, but consumers' per-backend metadata code is driven mainly by disunities #1–#3 (redundant representations, CPU/GPU split, un-migrated wrappers + decode). Steps 1–4 are Python-only and deletable-code-positive for vLLM/sglang; the durable structural asks on cuDNN are paged-direct (6a) and decode-unified (6b) — without those, the core LLM serving path never benefits from #3921 at all.
