# Paged-Prefill Unification Proposal

**Status**: draft v0.2 (2026-07-14; revised after a three-lens adversarial review: vLLM-maintainer, sglang-maintainer, flashinfer/cuDNN feasibility) · **Scope**: batch prefill with paged KV only — MLA, decode, and ragged prefill are explicit non-goals for v1 (they follow the same pattern later; §6 is explicit about what that scoping means for consumer adoption).

> **Working prototype (2026-07-16)**: branch `yanxu/unified-prefill-prototype` (worktree `.claude/worktrees/unified-prefill-proto`). `flashinfer/attention/unified.py` (contract + adapter layers, `resolve_paged_prefill()`, LSE base-2 contract incl. the cuDNN ln→base-2 fold — first empirical pin of the cuDNN LSE base), a **reject-or-correct fuzzer** (every call either raises a clean error or matches an fp32 oracle; 15 corruption classes, known gaps visible as xfail), conformance matrix green on H100 (fa2/fa3/cudnn/auto: 94 pass) and SM100 Blackwell (fa2/cudnn/trtllm-gen/auto: 93 pass), demo script `prototype_demo_unified_prefill.py`. The prototype's fuzzer found and fixed a live main bug: the cuDNN prefill/decode graph-cache keys omitted `attn_scale` → silent stale-scale replay on same-shape re-plans (fix commit `09c3df82` on the branch, cherry-pickable). Cold-user trial verdict: an engine's attention glue migrates in about a day; blank-file to green in ~30 min.
**Companion docs**: `ATTENTION_API_FRAGMENTATION.md` (full 6-family fragmentation survey), PR #3921 (cuDNN token-unit indptr / cu_seq_len direct path).

---

## 1. Motivation: FlashInfer is a router that doesn't route

FlashInfer ships (or wraps) at least five paged-prefill implementations — fa2, fa3, trtllm-gen, cuDNN, fmha_v2 — plus cutlass/cute-dsl on the ragged side. But there is no way to ask FlashInfer *"run the fastest one"*:

- `BatchPrefillWithPagedKVCacheWrapper(backend="auto")` resolves to **fa2 or fa3 only**, on an arch/feature gate with zero perf awareness (`flashinfer/utils.py:537-550`), and the choice is sticky after the first `plan()` (`prefill.py:2394`).
- trtllm-gen and cuDNN require the consumer to hardcode the backend string or call standalone functions with a different metadata dialect.

The consequence is that **backend selection lives in the serving engines, as static tables, and it is rotting**:

- **vLLM** picks trtllm-gen prefill on SM100 with *no perf heuristic at all*: the auto rule is literally "trtllm if the KV cache is unquantized" (`vllm/utils/flashinfer.py:474-486`), plus hard forces (page_size≥128 → trtllm, fp8-Q → trtllm, sinks → trtllm, DCP → FlashInfer-wrappers-because-"trtllm can't return LSE" `:431-437`). Selection even depends on an **HTTP probe to artifactory at process start** (`:352-359`).
- **sglang** selects backends in two static layers: an arch-default selector whose docstring says *"Auto select the fastest attention backend"* (`server_args.py:4557-4620`) — where fa3 is pinned over FlashInfer on Hopper as a workaround for a FlashInfer 0.6.1 perf regression (sglang#17411, `:4583-4587`) — and a ~50-branch per-model override cascade (`server_args.py:3627-4420`) containing outcomes like an outright ban: *"FlashInfer backend can significantly degrade the performance of Olmo3 models"* (`:4264-4277`).

External benchmarks then do the same thing: they pin one backend string and publish it as "FlashInfer". A 2026-07-13 B300 kernel-only chart shows *FlashInfer CUTLASS* at 0.79×/0.68× of FA4 (BF16/FP8) — **while cuDNN, a backend FlashInfer wraps, sits at 0.98×/1.05× in the same chart**. The gap between "FlashInfer's published number" and "the envelope of FlashInfer's backends" is a routing failure, not a kernel gap. The team mission is *near-SOL enabled by default*; for attention, default routing **is** the product.

Unification has two halves, and they are ordered:

1. **One metadata contract** so every backend is callable from the same `plan()`/`run()` — the precondition for any selection.
2. **`backend="auto"` v2** — capability matrix + perf heuristic + optional autotune — so the default is the envelope.

## 2. Why now: the metadata is already the same information

Restricted to paged prefill, the three dialects in circulation carry identical information:

| unified form | fa2/fa3 wrapper today | trtllm-gen one-shot today | cuDNN today |
|---|---|---|---|
| `qo_indptr (b+1,) i32 GPU` | `qo_indptr` (hidden `.cpu()` at plan, `prefill.py:2285`) | `cum_seq_lens_q` | post-#3921: `cu_seq_len_q` + ragged offset (one buffer); pre-#3921: element-unit `batch_offsets_q/o` |
| `kv_seq_lens (b,) i32 GPU` | derived: CSR `last_page_len` + indptr | `seq_lens` (uint32) | `actual_seq_lens_kv` as `(b,1,1,1)` |
| `block_tables (b,max_pages) i32 GPU` | CSR triple (`paged_kv_indptr` page-units + flat `indices` + `last_page_len`) | `block_tables` dense | `block_tables` dense (reshaped `(b,1,p,1)`) |
| `page_size, max_q_len, max_kv_len` host | page_size at plan; maxes derived on host | args | args |

Two load-bearing facts verified in sources:

- **cuDNN paged + cu_seq_len is already legal at the FE layer**: `sdpa_support_surface.h:304` requires, for paged, *either* `(seq_len_q + seq_len_kv)` *or* `(cu_seq_len_q + cu_seq_len_kv)`. PR #3921's non-paged restriction is FlashInfer plumbing, not a cuDNN limitation. Gate: effective version ≥ 9.24, where effective = **min(runtime backend, cuDNN the FE wheel was compiled against)** (`sdpa_support_surface.h:421`) — see §4 for the fallback consequence.
- **The wrapper's `run_args` already union both dialects** (`prefill.py:2823-2896` carries CSR *and* dense block_tables *and* kv_lens *and* host maxes), and `plan()` already accepts `seq_lens`, `block_tables`, `max_token_per_sequence`, `max_sequence_kv` as optional args (`:2110-2114`) — today consumed only by the cudnn/trtllm branches. The unified contract is mostly *promotion of existing optional args to the canonical form*, not a new API.

And the consumers have already voted with their feet — **both engines independently built the unified form and derivation code the library should own**:

- vLLM's trtllm path: dense `block_tables` + GPU `seq_lens` + GPU `cum_seq_lens_q/kv` + host maxes, no CPU mirrors (`backends/flashinfer.py:1175-1207`). For the FI-wrapper path it separately builds CSR metadata (numpy indptr in pinned buffers + a Triton page-index expansion kernel, `:902-940`) and a GPU cumsum for trtllm `cum_seq_lens_kv` (`:1184-1194`).
- sglang's `TRTLLMMHAMetadata` is field-for-field the unified schema (`trtllm_mha_backend.py:51-66`), built sync-free (`needs_cpu_seq_lens=False`).
- sglang's FlashInfer path, by contrast, is pinned to fa2 with page_size=1 token-CSR, plus **two monkeypatches into wrapper privates** (`fast_decode_plan`, a local `fast_prefill_plan` asserting `_backend=="fa2"`, `flashinfer_backend.py:178-289`) to get a sync-free plan path FlashInfer doesn't expose.

## 3. The unified contract

### 3.1 Metadata (plan-time)

```python
wrapper.plan(
    # canonical seqlen/paging metadata — all GPU, int32:
    qo_indptr,            # (b+1,) token-unit prefix sums (= cum_seq_lens_q)
    kv_seq_lens,          # (b,)   per-request valid KV lengths
    block_tables,         # (b, max_pages_per_seq) dense page table
    page_size,            # host int
    max_q_len, max_kv_len,          # host ints — REQUIRED
    qo_indptr_cpu=None, kv_seq_lens_cpu=None,  # optional host mirrors.
                                    # Formalizes sglang's global_override_indptr_cpu
                                    # and fast_prefill_plan.
    # unchanged: heads/dims/dtypes/causal/window_left/sm_scale/...
)
```

Rules:

- **Sync-free contract, stated per backend** (this is the pitch to sglang, so it must be precise): required host maxes remove the *max-derivation* syncs. That alone makes trtllm-gen and cuDNN-tokens plans sync-free. **fa2/fa3 additionally need the `*_cpu` mirrors** because their C++ split-KV scheduler consumes full host arrays (`prefill.py:2470-2472`) — without mirrors, fa2/fa3 plan() documents one D2H (today's hidden `.cpu()` at `:2285`, made explicit). Acceptance criterion for P1: with mirrors given, plan() performs **zero** D2H and is CUDA-graph-replay-safe — the property sglang's `fast_prefill_plan` monkeypatch exists to obtain.
- The CSR triple (`paged_kv_indptr`/`indices`/`last_page_len`) remains accepted, and at **page_size=1 it stays the canonical input**: dense block_tables at page_size=1 is `(b, max_context_len)` — a memory and derivation blowup — so CSR→dense derivation is forbidden there and dense-native backends (trtllm-gen: pages 16/32/64) are simply capability-filtered out. sglang's FlashInfer path runs page_size=1 by default (`environ.py:573`); it benefits from auto only after adopting page_size≥16 — an adoption prerequisite, not a library bug (§6).
- `kv_seq_lens` is the *single source of truth for masking*. Derived forms are produced from it by construction, so the direct-vs-conversion mask divergence flagged in the #3921 review **cannot arise at the wrapper layer** (paged cu_seq_len_kv is mask-only; addressing goes through block_tables).
- dtype normalized to int32 (trtllm's uint32 `seq_lens` handled in the adapter; xqa out of scope).
- **KV layout is an init-time contract, not a run-time cost**: the engine allocates one paged pool in one layout before any plan() (vLLM: process-global, `v1/attention/backends/utils.py:82-109`; sglang: NHD pools). Transposing a whole cache per call is not an option (the trtllm NVFP4 NHD path already forces a full `.contiguous()` copy, `prefill.py:4523-4528`). Therefore `kv_layout` is an input to the init-time backend resolution (§5.3) and a per-backend capability output: **auto never routes to a backend whose layout requirement mismatches the allocated pool**, and the recommended engine flow is *resolve first, then allocate the pool in the winner's preferred layout*.

### 3.2 Derivation layer (library-owned)

One fused CUDA/Triton kernel + tiny host mirror, run **unconditionally at each plan()** and memoized on the wrapper for that plan's lifetime. No cross-plan pointer-keyed caching: engines and the wrapper's own CUDA-graph mode rewrite the *same* persistent metadata buffers in place every step (vLLM persistent `paged_kv_indptr` buffers `backends/flashinfer.py:724-739`; wrapper `copy_()` into `_qo_indptr_buf`/`_paged_kv_indices_buf` `prefill.py:2338-2346`), so `data_ptr` identity says nothing about content. If cross-plan skip ever matters, it keys on an explicit engine-provided generation token.

| derived | needed by | from |
|---|---|---|
| `cum_seq_lens_kv (b+1,)` | trtllm-gen, cuDNN cu_seq_len_kv | cumsum(kv_seq_lens) |
| `q_seq_lens (b,)` | cuDNN `actual_seq_lens_q` (legacy graph), heuristics | diff(qo_indptr) |
| CSR triple (page-unit indptr, flat indices, last_page_len) | fa2/fa3 kernels | block_tables + kv_seq_lens + page_size; indices buffer **capacity-allocated from the block-table width** (`b * max_pages_per_seq`) so sizing needs no sync |
| dense block_tables | trtllm/cuDNN when caller gave CSR (page_size>1) | today a **host Python for-loop per plan** in the trtllm branch (`prefill.py:2443-2463`) — replaced by this kernel |

What this deletes and what it doesn't: it deletes vLLM's CSR materialization (vLLM's source form *is* the dense block table) and FlashInfer's own Python block-table loop. It does **not** delete sglang's Triton page-table builder — that kernel's input is the engine-owned `req_to_token` slot pool via `req_pool_indices` indirection with fused SWA translation (`triton_ops/trtllm_mha_page_table.py:1-13`), an engine-specific source form. Accepting `req_to_token`-style pool indirection as a third input form is possible but is explicitly *out of scope* here (large surface, one consumer).

### 3.3 Output contract

- **LSE**: `(total_q_tokens, num_qo_heads) fp32, base-2` for every backend. Verified de-facto standard already for fa2/fa3/cutlass/trtllm-gen (see `ATTENTION_API_FRAGMENTATION.md` §1 A4); cuDNN is the outlier (padded `(b,max_s,h)`, natural-log Stats, **no in-repo test pins its base**) — fixed at the adapter with `batch_offsets_stats` tokens-mode (multiplier `h_qo` already wired by #3921) plus a `log2(e)` fold, and pinned by a conformance test.
- LSE support becomes a queryable capability (§5.1). The motivating rot: vLLM's DCP fork exists solely on a *"Trtllm does not support returning LSE"* claim (`vllm/utils/flashinfer.py:431-437`) that is **stale for both phases on the trtllm-gen path** — `trtllm_batch_context_with_kv_cache` exposes `lse/return_lse` (`prefill.py:4462-4463`) and so does trtllm-gen decode (`decode.py:3030-3031`; only the xqa path refuses, `:3257-3259`), with LSE parity tested against the fa2 reference (`test_trtllm_gen_attention_decode.py:776-796`). Consumer-side capability tables rot; that is the argument for library-side capability queries, not better consumer tables.
- Scope note: sglang's dominant prefill-with-prefix path merges a *ragged* wrapper result with the paged result via `merge_state` (`flashinfer_backend.py:794-798, 1071`) — so the LSE convention must hold for the ragged wrapper too. That is cheap (ragged fa2/fa3/cutlass already comply) but must be stated, since a paged-only LSE guarantee would still break the merge.
- `out=` pre-allocation supported everywhere; scale convention `sm_scale` + optional `q/k/v_descale` (float or tensor), backend adapters fold to `bmm1/bmm2` internally (vLLM does this folding today, `backends/flashinfer.py:1476-1484`).

## 4. Backend adapters and work items

| backend | metadata it consumes | adapter work | kernel/library work |
|---|---|---|---|
| fa2/fa3 | CSR + full host arrays for split-KV planner | derive CSR (kernel above); official `*_cpu` mirrors for sync-free plan | none |
| trtllm-gen | unified form natively | dedupe (derive cum_seq_lens_kv); uint32; wrapper parity with the standalone: fp4 output, bmm scales, sinks, **and NVFP4 KV-cache input (`kv_cache_sf` + fp8-Q pairing, `prefill.py:4456-4458`)** — without the last one, auto can never carry vLLM's nvfp4 configuration | none for metadata; prefill/decode LSE already exposed (see §3.3) |
| cuDNN | post-#3921 tokens mode; paged still legacy | **paged + cu_seq_len direct path** (#3921 follow-up; FE-legal per `sdpa_support_surface.h:304`). The `(b,1,1,1)` lens/legacy graph is **kept as a version-conditional fallback**, not retired: COMPOSITE rejects `CU_SEQ_LEN_*` outright (`:402-405`), and the ≥9.24 gate is on min(runtime backend, FE-compiled-against) (`:421`) — a 9.24 runtime under an older-built FE wheel still rejects the graph. Gate by feature-probe + NOT_SUPPORTED→legacy fallback (see #3921 review). LSE fold to base-2 `(tokens,h)` | cuDNN backend: none for this step. Asks for later phases: decode-unified, `sdpa_fp8` cu_seq_len binding, non-uniform stride multipliers |
| fmha_v2 (**SM90 + SM12x**, `jit/attention/modules.py:2102-2103`) | per-seq + cum + host ints, paged-capable, wrapper-unreachable (`prefill.py:4897-4924`) | wire into wrapper as `backend="fmha_v2"` (explicit deliverable, §7 P2.5); stats→LSE post-process; dtype limits fp16/bf16/e4m3 (e4m3 rejected on SM120, `prefill.py:5132-5138`). Notably this adds a second Hopper candidate exactly where sglang pinned fa3 over FlashInfer | none |
| cutlass / cute-dsl | **no paged-prefill kernel exists** (`jit/attention/modules.py:994-995`; cute-dsl rejected in `__init__` `prefill.py:1692-1696`) | out of v1 scope; candidates for later paged enablement | kernel work — not blocking v1 |

## 5. `backend="auto"` v2: capability → heuristic → autotune

The capability/heuristic layers (`@backend_requirement` + `heuristic_func` + `suitable_auto_backends`, `flashinfer/utils.py:1053-1332`) are proven on GEMM/MoE and used by **zero** attention APIs; the autotune layer has exactly one attention precedent, MLA decode (`mla/_core.py:3063-3147`).

### 5.1 Capability matrix

Decorate the paged-prefill entry point with `@backend_requirement`: per-backend checkers encode dtype-in/out (incl. required Q dtype and **NVFP4 KV**), head ratio, page-size set, **kv_layout requirement**, sinks, LSE, non-causal, window, **custom mask** (packed masks + the multi-item `prefix_len_ptr`/`token_pos_in_items` family — today these silently pin sglang's spec-decode target-verify and multi-item paths to fa2/fa3, `flashinfer_backend.py:664-675, 785-791`), arch, cubin availability. Consumers get `is_backend_supported(backend, cc)` / `suitable_auto_backends` for free — the deletion target for vLLM's scattered gates (`utils/flashinfer.py:384-408`, `backends/flashinfer.py:338-356,680-719`) and sglang's whitelists. Cubin availability becomes a library answer instead of vLLM's 5-second HTTP probe.

### 5.2 Heuristic default

`heuristic_func` returns the ordered candidate list. **Feature set in the sync-free case** (the case the contract optimizes for): arch, dtypes, heads/dims, page_size, causal/window, batch_size, total_q_tokens, host maxes — *not* full length distributions, which exist only when the caller opts into `*_cpu` mirrors (the plan-time host arrays cited at `prefill.py:2245-2312` are materialized by exactly the syncs §3.1 eliminates). The v1 table is seeded from the flashinfer_benchmark suite across (arch × dtype × GQA ratio × max-len regime); it only has to beat static consumer tables with documented failures on both sides.

### 5.3 Two-level selection: init-time pinning, then bounded tuning

The adversarial review surfaced a real contradiction in v0.1 (shape-bucketed tuning vs. capture-stable selection). Resolution — a two-level contract:

1. **Init-time resolution** (`flashinfer.attention.resolve_paged_prefill(arch, heads, dims, dtypes, page_size, kv_layout, feature_flags) -> ResolvedConfig`): a **static, config-level query — no wrapper instance, no tensors** — because engines decide graph-affecting properties before any wrapper exists: vLLM sets its cudagraph support level in a `@classmethod` from `(vllm_config, kv_cache_spec)` alone (`backends/flashinfer.py:743-783`) and pins Q dtype at builder `__init__` (`:684-696`). The resolution pins the **candidate set**: all backends that are *observationally identical* to the engine (same required Q dtype, same LSE availability, same out dtype, graph-safe) and layout-compatible with the pool (§3.1). It also reports the winner's preferred `kv_layout` so the engine can allocate the pool accordingly.
2. **Plan-time choice within the pinned set**: heuristic (default) or autotuned (opt-in) selection over the pinned candidates, keyed on bucketed (total_q_tokens, max_kv_len). Because every candidate in the set is observationally identical, a bucket-dependent winner change is invisible to the engine — no Q-dtype flapping (vLLM mutates `q_data_type` back per-build today, `:1036-1038`), no graph invalidation.

### 5.4 Autotune (opt-in)

MLA decode is the precedent but does **not** transfer directly: MLA calls `choose_one` at run() with real q/kv/out tensors, and its synthetic profiling initializers read `num_pages`/`page_size` off the real kv_cache (`mla/_core.py:3140-3147, 2167-2192`). At plan() none of those tensors exist. Design:

- **Plan-time selection is a shape-descriptor key lookup** against the autotune cache (no tensor allocation on the hot metadata path).
- **Profiling runs where tensors exist**: offline / at engine warmup with an explicitly budgeted synthetic KV allocation, or lazily at the first eager (non-capturing) run(), MLA-style. Never inside plan(), never inside capture.
- Tuned choices persist via the autotune cache v2 mechanism (PR #3861 `persist=`) for warm-start routing; candidates whose JIT module isn't built return no tactics instead of blocking (CuteDsl-MLA precedent, `mla/_core.py:2449-2455`).

### 5.5 Determinism mode

A library-level `deterministic=True` restricts candidates to batch-invariant kernels **and pins the split policy** — `fixed_split_size`/`disable_split_kv` already exist as plan() args (`prefill.py:2115-2116`); vLLM pins them under `VLLM_BATCH_INVARIANT` (`backends/flashinfer.py:586-593`) and sglang additionally forces split tile sizes via env and keeps a deterministic backend whitelist (`server_args.py:211`) — one more consumer table the capability query should absorb.

## 6. Consumer end-state — scoped honestly to v1

**What v1 (paged prefill only) does *not* let consumers delete** — decode shares most of the routing surface, so full deletion waits on the decode follow-up:

- vLLM keeps `use_trtllm_attention` (its decode auto-rule `num_tokens<=256` and spec-decode force live there, `utils/flashinfer.py:458-486`), the artifactory probe (backs decode + cudagraph-support gates), and the CSR derivation for decode/cascade (`backends/flashinfer.py:1103-1113`). The cascade path is hardwired to the fa2 `MultiLevelCascadeAttentionWrapper` (`:878-883, 1118-1163`) and the spec-decode reorder policy is set from `can_use_trtllm` at init (`:564, 701`) — both survive v1.
- sglang: `fast_decode_plan` patches the *decode* wrapper — it retires with decode unification, not v1. The per-model cascade shrinks materially only after decode/MLA follow-ups; v1 absorbs the **CUDA MHA paged-prefill selection** (fa2/fa3/trtllm-gen/cuDNN/fmha_v2 inside the `"flashinfer"` registry backend), not MLA-family routing, not the prefill/decode hybrid split.

**What v1 does deliver:**

- **vLLM**: the prefill half of the fork disappears — `FlashInferBackend` builds unified metadata once (it already does, for the trtllm path), passes `backend="auto"`, deletes `prefill_use_trtllm` + the prefill-side metadata duplication; `resolve_paged_prefill()` replaces the init-time capability guesswork. The BF16-Q+fp8-KV per-layer Triton dequant mock-cache (`:98-210, 1714-1740`) is a *coverage* gap tracked separately — auto reduces it from "wrong kernel" to "capability-filtered".
- **sglang**: for the no-prefix / multimodal / target-verify / deterministic prefill subset, the `"flashinfer"` backend becomes unified-metadata + auto, and `fast_prefill_plan` retires (P1 acceptance criterion: mirrors ⇒ zero-D2H, replay-safe plan). **Limitation**: the dominant radix-cache extend path is ragged(new)+paged(prefix)+`merge_state` (`flashinfer_backend.py:794-798`) — the paged half can use auto (with the ragged-LSE compatibility guarantee of §3.3), but full-path benefit waits on the ragged follow-up. Adoption prerequisite: page_size≥16 (§3.1); `TRTLLMHAAttnBackend` no longer needs to subclass-and-ignore the fa2 wrappers.
- Both keep escape hatches: `backend="trtllm-gen"` etc. remain valid explicit pins; engine policy knobs (batch-invariance, force flags) become pass-throughs.

## 7. Migration phases (each independently shippable)

| phase | deliverable | depends on |
|---|---|---|
| **P0** | Conformance matrix: one backend-parametrized paged-prefill test (reference stack from `test_trtllm_gen_attention_prefill.py` + `sink_attention_unified`), pinning outputs *and* LSE base/shape per backend (incl. ragged wrappers for the merge_state guarantee) | — |
| **P1** | plan() canonical unified metadata + derivation kernel + official `*_cpu` mirrors. **Acceptance: with mirrors, plan() is zero-D2H and replay-safe** (retires sglang's `fast_prefill_plan`); CSR canonical at page_size=1 | P0 |
| **P2** | cuDNN adapter: wrapper→tokens mode (#3921), paged+cu_seq_len direct plumbing with feature-probe + legacy fallback, LSE base-2 fold | ✅ #3921 merged 2026-07-16 (81632eee) and #3784 merged 2026-07-15; #3801 still open. Remaining #3921 follow-ups tracked in §8 |
| **P2.5** | fmha_v2 wired into the wrapper (`backend="fmha_v2"`, SM90+SM12x, stats→LSE) | P0 |
| **P3** | `@backend_requirement` capability matrix (incl. kv_layout, custom-mask, nvfp4 axes) + heuristic table; `backend="auto"` spans fa2/fa3/trtllm-gen/cuDNN/fmha_v2; `resolve_paged_prefill()` static query | P1, P2, P2.5 |
| **P4** | AutoTuner integration (shape-descriptor lookup at plan, warmup/eager profiling, persist via cache v2) | P3, #3861 |
| **P5** | vLLM/sglang PRs consuming auto for the prefill half + deleting prefill-side routing/derivation | P3 |
| **P6+** | decode unification (unlocks the §6 "not deleted" list), ragged prefill (unlocks sglang extend path), MLA | — |

## 8. Risks / open questions

- **Heuristic quality**: a bad default is worse than the hand-tuned rules it replaces. Mitigation: seed from the benchmark suite; keep explicit pins first-class; autotune as the perf backstop; ship auto behind a flag for one release.
- **Feature asymmetries**: sinks (trtllm-only), SWA dual page tables (sglang builds a second translated table, `trtllm_mha_backend.py:136-138`), per-layer-uniform hyperparameter restriction of the FI wrapper path (`vllm backends/flashinfer.py:1023-1034`). Capability checkers must encode these or auto mis-routes.
- **Q-quantization policy**: vLLM casts Q to fp8 itself conditional on routing (`:672-696`). Under the two-level contract, required Q dtype is part of init-time pinning: the library declares it per candidate set; the engine casts once after `resolve_paged_prefill()`.
- **Page-size inversion is a P3 blocker for the sglang half, not a footnote**: engines currently choose page size *for* a backend (sglang forces 64 for trtllm_mha, `server_args.py:4788-4792`); under auto, page size is an input constraint that filters candidates. Auto cannot fix a page-size choice made for a backend that then loses the heuristic — document, and recommend page_size≥16 defaults for engines that want routing.
- **cuDNN specifics**: graph-cache key gaps (scale not keyed — pre-existing footgun, `cudnn/prefill.py:114-129`), feature-probe over version-compare, and the open #3921 findings (mask-semantics divergence, fp8 gate, int32 conversion overflow, #3784 conflict) must land first.
- **Chart-level accountability**: after P3, publish a "FlashInfer auto" row in the benchmark suite so external comparisons have a routed number to cite. The B300 chart's 0.79×/0.68× would have been 0.98×/1.05× (its own cuDNN row) under routing — that delta is the value of this proposal, measured by someone else.
