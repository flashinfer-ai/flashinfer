# Block-Extend (dLLM Block-Diffusion) Attention — Design Response to Review

Status: draft response plan for the upstream review of `feature/block-extend`.
Audience: FlashInfer maintainers (Zihao et al.).
Scope: the two design asks (shared-config isolation; API convergence) plus the
reviewer-flagged correctness items. Each section states the goal, what the code
does today, the proposed change, and the concrete edit list.

---

## 0. Summary of the two design asks

1. **Isolate the new mask mode from the shared compilation space.** The reviewer
   is concerned `kBlockExpanding` inflates JT-cache / AOT binary size for every
   existing prefill URI. Requested fix: a small, dedicated, fixed-mask-mode
   cartesian product for dLLM instead of a new value multiplied into the big
   one.
2. **Converge the API surface.** The PR ships a parallel `flashinfer/dllm/`
   package (4 APIs, own module cache, own CI/AOT lane). Requested fix: expose
   block-diffusion as a **mask option** on the existing prefill
   APIs/wrappers — `mask_mode="block_diffusion"` + `block_size` +
   `q_offset` / `kv_offset` — following the existing `kMultiItemScoring`
   precedent.

A precursor note on framing (worked out against the current branch, not a
rejection of the ask): AOT and the shared default mask list are **already
isolated**. `flashinfer/aot.py` is untouched and AOT still uses the
`[0,1,2,3]` default. The shared jinja changes are inert
`{% if 'maybe_q_block_expanding_offset' in additional_params_decl %} … return 0;`
guarded getters, not a new mask-mode axis value. The genuine leakage is
narrower (see §1). Both asks still stand and we adopt them.

---

## 1. Ask 1 — Isolate the mask mode from shared compilation

### 1.1 What the code does today

- Enum / literal additions (these are shared infra and are correct to keep):
  - `include/flashinfer/attention/mask.cuh:26` — `kBlockExpanding = 4U`.
  - `flashinfer/utils.py:43` — `MaskMode.BLOCK_EXPANDING = 4`.
  - `flashinfer/jit/utils.py:95` — `4: "MaskMode::kBlockExpanding"`.
- The JIT gentemplate functions gained an optional `mask_modes` arg
  (`flashinfer/jit/attention/modules.py`, `gen_customize_single_prefill_module`
  and `gen_customize_batch_prefill_module`); default is `[0,1,2,3]`, so existing
  URIs are unaffected.
- **Single dLLM path is already correct**: `flashinfer/dllm/block_extend.py:229`
  calls `gen_customize_single_prefill_module(..., mask_modes=[4])`.
- **Batch dLLM path leaks**: `flashinfer/dllm/batch_block_extend.py:330,468`
  builds the dedicated dLLM batch URI with `mask_modes=[0,1,2,3,4]` — it
  instantiates the four irrelevant mask modes into the dedicated URI's source
  set. This is the real cache inflation. It is confined to the dLLM URI, not the
  shared prefill URIs, but it is waste and a footgun (the dispatcher can pick a
  non-block-extend kernel behind the dedicated front-end).
- Jinja getters (4 files) are guarded and inert for existing URIs; they are
  kernel-side plumbing for the dLLM variant structs and should stay, but they
  should read from the dLLM variant params only.

### 1.2 Proposed change

Make the dedicated dLLM URIs a small, fixed, closed cartesian product — exactly
what the reviewer asked for. Concretely:

1. **Fix the batch mask-mode list to `[4]`** in
   `flashinfer/dllm/batch_block_extend.py:330,468`, matching the single path.
   This alone removes the leak from the dedicated URI.

2. **Restrict the dLLM URI axes to what dLLM needs**, as an explicit closed
   product rather than the full prefill matrix:
   - dtype: `{fp16, bf16}` (drop fp8 — see §3.3).
   - head_dim: `{64, 128}` (extend only if a production dLLM config needs more).
   - layout: ragged + paged (both kept — SGLang-dLLM uses both).
   - backend: FA2 ( Ampere/Hopper) + FA3/Hopper (`_fa3`).
   - mask_mode: fixed `{4}` — never enumerated, never a parameter.

3. **Stop parameterizing the shared gentemplate signature.** Remove the
   `mask_modes` arg from the shared `gen_customize_single_prefill_module` /
   `gen_customize_batch_prefill_module` (and from
   `get_customize_batch_prefill_module` in `prefill.py:191`). Replace it with a
   small, standalone dLLM gen entry that compiles **only** mask-mode 4 over the
   restricted axes above. The dLLM variant structs (`BatchBlockExtendOffsetAttention`
   / `BatchBlockExtendOffsetAttentionFA3`) are already passed inline as
   `variant_decl` strings, so a new standalone C++ variant class is not needed —
   only a new Python gen function.

   This is the structural change that turns "the new mode has a hook into the
   shared config" into "the new mode has its own tiny config and never touches
   the shared one."

4. **Keep the shared infra (enum value + literal + jinja getters).** These do
   not inflate any cartesian product on their own; the kernel needs the enum to
   dispatch at runtime. The jinja getters remain `{% if %}`-guarded so existing
   URIs see `return 0;`.

### 1.3 Concrete edit list (Ask 1)

- `flashinfer/dllm/batch_block_extend.py:330` and `:468`: `mask_modes=[0,1,2,3,4]` → `[4]`.
  **[DONE]** — both set to `[MaskMode.BLOCK_EXPANDING.value]`.
- `flashinfer/dllm/batch_block_extend.py` and `block_extend.py`: restrict the
  dtype/head_dim/backend loop to the closed dLLM product above; assert on
  anything else rather than silently generalizing.
  **[DONE]** — dtype was already gated (`_get_dtype_str`, `_get_batch_be_module_uri`);
  idtype (`{int32, int64}`) and head_dim (`{64, 128}`) are now gated too; backend
  remains limited to `auto/fa2/fa3` by `select_best_backend[_paged]`.
- `flashinfer/jit/attention/modules.py`: remove `mask_modes` param from the two
  shared `gen_customize_*` functions; add `gen_customize_block_extend_*` (or a
  dedicated `gen_dllm_*` in `flashinfer/dllm/`) that loops only mask-mode 4
  over the restricted axes. **[NOT YET DONE]** — see §1.4 note; the shared
  signature change is now low-risk because the only callers passing
  `mask_modes` are the dLLM paths, which already pass `[4]`.
- `flashinfer/prefill.py:191`: remove the `mask_modes` passthrough from
  `get_customize_batch_prefill_module`. **[NOT YET DONE]** — same.
- Keep: `mask.cuh:26`, `utils.py:43`, `jit/utils.py:95`, the four jinja getters.

> Note on §1.2 step 3 (standalone gen function): with steps 1–2 done, the
> remaining risk of leaving `mask_modes=` on the shared `gen_customize_*`
> functions is purely API-cleanliness — the shared default stays `[0,1,2,3]`,
> so no existing URI is affected, and the only override is the dLLM `[4]`. Doing
> the standalone-gen-function refactor is still the right structural move per
> the reviewer, but it can land as a follow-up commit without blocking the
> correctness/closed-product changes. It is listed as task #6 below.

### 1.4 Reviewer reassurance

- AOT: no change to `flashinfer/aot.py`; existing prefill AOT binary size
  unchanged. (Optional follow-up: register the dLLM URIs for AOT so dLLM users
  can ship prebuilt — but that is opt-in and out of scope for the isolation
  ask.)
- JIT cache: existing prefill URIs compile the same `[0,1,2,3]` set as before.
  dLLM URIs compile only `[4]`.

---

## 2. Ask 2 — Converge the API surface

### 2.1 What the code does today

`flashinfer/dllm/__init__.py` exports four user-facing APIs:

1. `block_extend_attention_with_offset` (`block_extend.py:237`) — single-request
   block-extend attention with explicit q/kv offsets. Has its **own global
   module cache** `_MODULE_CACHE_WITH_OFFSET = {}` (`block_extend.py:115`,
   used `:187-198`) and reimplements the AOT-vs-JIT branch by hand *without*
   the `JitSpec.build_and_load` file-lock guard (`jit/core.py:307`).
2. `block_extend_cascade` (`block_extend.py:317`) — single-request cascade:
   current chunk (block-extend) + prefix (non-causal) + `merge_state_in_place`.
3. `BatchBlockExtendRaggedOffsetWrapper` (`batch_block_extend.py:406`) — batched
   ragged; internally instantiates `BatchPrefillWithRaggedKVCacheWrapper`,
   passes `jit_args` (custom variant + extra tensors/scalars) and
   `mask_mode=BLOCK_EXPANDING`.
4. `BatchBlockExtendPagedOffsetWrapper` (`batch_block_extend.py:264`) — batched
   paged; same pattern over `BatchPrefillWithPagedKVCacheWrapper`.
   Plus composite helpers (`batch_block_extend_cascade`, `sglang_style_cascade_attention`).

Observations that make the fold-in cheap:
- APIs #3/#4 are **thin shims** that already call the existing
  `BatchPrefillWithRagged/PagedKVCacheWrapper` with `jit_args` + `mask_mode`
  (the PR already added `mask_mode=` to both wrappers' `plan`: `prefill.py:1804`
  paged, `prefill.py:2907` ragged). They exist only to (a) build the customized
  JIT module with the dLLM variant and (b) thread `q_offsets` / `kv_offsets` /
  `dllm_block_size` as extra run-args.
- **Precedent already in tree**: `kMultiItemScoring (=3)` is a structured mask
  exposed as ordinary plan args (`prefix_len_ptr`, `token_pos_in_items_ptr`,
  `max_item_len_ptr`, …) on the *existing* wrappers
  (`prefill.py:2901-2904` ragged, `:1793, :1882` paged), with an auto-flip to
  MULTIITEMSCORING at `prefill.py:2460-2461` (`if self._prefix_len_ptr is not
  None: mask_mode = MULTIITEMSCORING.value`). No new wrapper class, no separate
  cache, no separate CI lane. This is the exact template.
- Separate CI lane: `scripts/task_jit_run_tests_dllm.sh` + two CI jobs
  `gpu-tests-dllm-a10g` / `gpu-tests-dllm-h100` (`.github/workflows/pr-test.yml`).
  There is no separate AOT *builder*; only a parallel test/CI lane + URI
  namespace.

### 2.2 Proposed change — target shape

Expose block-diffusion as a **mask option** on existing prefill APIs, mirroring
the multi-item-scoring pattern.

#### 2.2.1 Batched path (replaces APIs #3, #4)

Add to `BatchPrefillWithRaggedKVCacheWrapper.plan` / `BatchPrefillWithPagedKVCacheWrapper.plan`:

```python
def plan(self, ...,
         mask_mode: Optional[Union[str, int]] = None,   # already partially present
         block_diffusion: Optional[BlockDiffusionConfig] = None, ...
         ) -> None:
```

where `BlockDiffusionConfig` carries `dllm_block_size`, `q_offsets`, `kv_offsets`
(tensors or None). Behavior mirroring `prefix_len_ptr`:

```python
# in run() dispatch (analog of prefill.py:2460-2461)
if self._block_diffusion is not None:
    mask_mode = MaskMode.BLOCK_EXPANDING.value
```

The dLLM variant (`BatchBlockExtendOffsetAttention[_FA3]`) is selected via the
existing `jit_args`/`jit_kwargs` mechanism when `block_diffusion` is set; the
offset tensors and `dllm_block_size` flow through as normal run-args alongside
`custom_mask_buf`, `alibi_slopes`, `sm_scale`, etc. — exactly as the current
dllm wrappers already do, minus the proxy class.

The user-facing call becomes:

```python
wrapper.plan(..., block_diffusion=BlockDiffusionConfig(64, q_off, kv_off))
out = wrapper.run(q, k, v, ...)
```

instead of:

```python
d = BatchBlockExtendPagedOffsetWrapper(...); d.plan(q_off=..., kv_off=...)
```

#### 2.2.2 Single-request path (replaces APIs #1, #2)

- `block_extend_attention_with_offset`: route through the existing
  `single_prefill_with_kv_cache` with `mask_mode=BLOCK_EXPANDING` (+ `dllm_block_size`,
  `q_offset`, `kv_offset` as extra scalars on the single-prefill JIT path,
  mirroring the batched `jit_args` shape). This **eliminates the dedicated
  `_MODULE_CACHE_WITH_OFFSET`** dict and the hand-rolled AOT/JIT branch — the
  single path reuses `JitSpec.build_and_load` (with its file lock) like the rest
  of the codebase.
- `block_extend_cascade`: keep as a thin convenience, but reimplement it as a
  composition of (single-prefill with `block_diffusion=True`) + (non-causal
  single-prefill) + `merge_state_in_place`, with no separate module cache.

#### 2.2.3 Optional thin convenience layer

If SGLang-dLLM wants a one-call entry point, keep a **thin** (≤30-line)
`flashinfer/dllm` shim that just calls into the existing APIs with
`block_diffusion=...`. The shim holds no cache, no AOT branch, no wrapper
subclass. This preserves the ergonomic SGLang entry point without fragmenting
the canonical API.

### 2.3 Concrete edit list (Ask 2)

- `flashinfer/prefill.py`: add `block_diffusion: Optional[BlockDiffusionConfig]`
  plan arg + run-time auto-flip (analog of `:2460`) on both batch wrappers;
  thread offset tensors / `dllm_block_size` through run-args.
- `flashinfer/dllm/batch_block_extend.py`: delete
  `BatchBlockExtendRaggedOffsetWrapper` (`:406`) and
  `BatchBlockExtendPagedOffsetWrapper` (`:264`); keep only the variant-decl
  strings + a thin plan/run convenience if SGLang needs one.
- `flashinfer/dllm/block_extend.py`: delete `_MODULE_CACHE_WITH_OFFSET`
  (`:115`), `get_block_extend_module_with_offset` (`:153`), and the hand-rolled
  AOT/JIT branch (`:194-205`); route `block_extend_attention_with_offset`
  through `single_prefill_with_kv_cache`; reimplement `block_extend_cascade` as
  a composition.
- `.github/workflows/pr-test.yml`: drop the `gpu-tests-dllm-a10g` /
  `gpu-tests-dllm-h100` dedicated CI lane; move
  `tests/attention/test_dllm_blockwise_mask_attention.py` into the existing
  prefill test part (parametrized by `block_diffusion=...`).

### 2.4 What we keep from the PR

- The kernel work (`block_expanding_prefill.cuh`, the Hopper mainloop edits,
  the mask.cuh branch) is sound and stays.
- The dedicated, restricted dLLM **URI namespace** stays (per Ask 1) — it is
  small and closed, not a parallel API family.

---

## 3. Correctness items (reviewer-flagged, pre-merge)

These are normal review iteration, listed here so they are not lost.

### 3.1 Zero-visible-KV TMA hang in the Hopper mainloop

- Location: `include/flashinfer/attention/hopper/prefill_sm90.cuh` (the shared
  template `PrefillWithKVCacheKernel`, used by single / ragged-batch /
  paged-batch FA3), with the producer/consumer handshake in `mainloop.cuh` /
  `mainloop_mma.cuh`. The block-expanding mask can make an entire CTA's tile set
  have **zero visible KV** (all tiles invisible), which the current code handles
  on the consumer side with a `store_zero` fast-path (`prefill_sm90.cuh:245`).
- Exact deadlock mechanism (verified against the code on this branch):
  - Producer and consumer advance `work_idx` in lockstep — both skip the
    `++work_idx` for a zero-tile (`prefill_sm90.cuh:176-180` producer
    `continue`; consumer `continue` at `:248` bypasses the `++work_idx` at
    `:286`). So `work_idx` parity stays consistent across both sides. ✓
  - The producer of every non-zero tile begins `load()` by waiting at
    `shared_storage.barrier_O.wait((work_idx + 1) % 2)` (`mainloop.cuh:264`).
    That arrival is owed by the *previous* work-tile's consumer, which makes it
    inside `mma_f16` (`mainloop_mma.cuh:82-90`, gated `work_idx != 0`).
  - **Bug:** when the previous tile was zero-visible, its consumer took
    `store_zero` and skipped `mma_f16`, so it **never arrived at `barrier_O`**.
    The next non-zero tile's producer waits forever → deadlock. FA2 is immune
    (no persistent TMA pipeline / cross-tile barrier).
- Fix: in the consumer's `store_zero` fast-path, emit the same `barrier_O.arrive`
  that `mma_f16` would have (gated `work_idx != 0`, last warp, `elect_one_sync`),
  so the next-tile producer's `barrier_O.wait` is released.
  - **Applied:** `prefill_sm90.cuh` consumer `num_kv_tiles <= 0` block now
    issues `barrier_O.arrive`, verbatim mirroring `mainloop_mma.cuh:82-90`.
  - The quantization twin (`hopper/quantization/prefill_sm90.cuh:222`) is NOT
    reachable for `kBlockExpanding` (FP8 is outside the dLLM closed product per
    §1.2; the FP8 dispatch does not set the block-expanding mask), so it needs
    no matching edit.
- Regression test: `tests/attention/test_dllm_blockwise_mask_attention.py ::
  test_zero_visible_kv_no_hang` — builds configs with `kv_offset` high enough that
  the first Q-tile's `kv_valid_end <= 0` (single-path all-invisible, partial, and
  baseline) plus a batch with a zero-visible request followed by a normal one
  (the cross-tile shape). Asserts FA3 completes and matches the FA2 reference
  (which writes the correct all-zero rows). A faulty kernel hangs on an H100;
  this is GPU-verifiable only on SM90.

### 3.2 Module URI missing idtype

- The dedicated dLLM **batch** URIs (`batch_prefill_block_expanding_hd{hd}_{dtype}`)
  embed `dtype` but previously not `idtype` (q input index dtype). The standard
  batch-prefill URI scheme embeds idtype as `dtype_idx_{...}`
  (`flashinfer/jit/attention/modules.py:390`), so omitting it lets an int32 and
  an int64 index build of the same `(head_dim, dtype)` alias the same generated
  source directory — the later build overwrites the earlier binary.
- The single-request path has no index-dtype axis (offsets are forwarded as
  scalars), so its URI/cache-key need no idtype.
- **Applied:** `_get_batch_be_module_uri(head_dim, dtype, idtype)` now embeds
  `idx{i32|i64}` and validates idtype; all six call sites thread the real
  `qo_indptr.dtype`. (The selector helpers `select_best_backend[_paged]` and
  both `_create_inner_wrapper` methods now take/pass `idtype`.) Single-path URI
  unchanged (no idtype dimension).

### 3.3 fp8 silently coerced to fp16

- **Verified on this branch: this is already a reject, not a coercion.** Both
  the batch dtype helper (`_get_batch_be_module_uri`) and the single dtype helper
  (`block_extend.py ::_get_dtype_str`) already raise `ValueError` for any dtype
  outside `{fp16, bf16}` — fp8 included. There is no `.to(torch.float16)`
  rewrite anywhere in `flashinfer/dllm/`.
- The reviewer's note most likely targets an earlier revision. The remaining
  good-of-correctness work here was folded into §1.2 step 2: make the closed
  product explicit (head_dim now also gated to `{64, 128}`; see §1.2) so the
  rejection is uniform across axes rather than ad-hoc per helper.
- Future fp8 support (if ever needed for dLLM) should be added as an explicit,
  tested axis of the closed product — not a silent coercion.

---

## 4. Sequencing / how this lands

Suggested order to minimize review churn:

1. **Correctness first (§3)** — smallest, highest-risk items; ship them as the
   next push so CI-testing unblocks.
2. **Ask 1 isolation (§1)** — fix the batch mask-mode list to `[4]`; introduce
   the dedicated restricted gen function; remove the shared `mask_modes`
   parameterization and the `prefill.py:191` passthrough. This is mostly
   mechanical and keeps the kernel work intact.
3. **Ask 2 convergence (§2)** — the larger refactor (fold wrappers into
   `block_diffusion=` options). Land after #2 is green so the diff is reviewable
   in isolation. Keep a thin `flashinfer/dllm` convenience shim if SGLang-dLLM
   wants a one-call API; this is the negotiating point with the reviewer.

Open question for the reviewer: whether they want the thin `flashinfer/dllm`
convenience shim (§2.2.3) or full deletion. Recommend keeping the shim — it
protects the SGLang-dLLM production ergonomics that motivated upstreaming, at
zero infra cost.

---

## 5. Implementation status (this session)

"Branch" = `fdz-1999/flashinfer:feature/block-extend` working tree at
`C:\Users\fengdaozhuo.fdz\Desktop\refine`.

### Done & in the working tree

Two local commits on `feature/block-extend` (not pushed):

- `cc6a23f3` — §3 correctness + §1.2 steps 1–2 (closed product, batch `mask_modes` fix).
- *(this commit, to be amended/added)* — §1.2 step 3 single-path dedicated gen
  function + §2.2.2 single-path convergence (drop own cache + hand-rolled AOT/JIT).

| Section | Change | File(s) |
|---|---|---|
| §3.1 | `barrier_O.arrive` added to consumer zero-visible-KV fast-path (mirrors `mma_f16`'s `mainloop_mma.cuh:82-90`), fixing the cross-tile TMA deadlock on Hopper | `include/flashinfer/attention/hopper/prefill_sm90.cuh` |
| §3.1 | Regression test `test_zero_visible_kv_no_hang` (single-path all-invisible/partial/baseline + batch zero-then-normal) with a `kv_offset`-aware reference helper | `tests/attention/test_dllm_blockwise_mask_attention.py` |
| §3.2 | `idtype` embedded in batch dLLM URI (`idx{i32\|i64}`), validated, threaded through `select_best_backend[_paged]` and both `_create_inner_wrapper`; single-path URI unchanged (no idtype axis) | `flashinfer/dllm/batch_block_extend.py` |
| §1.2 step 1 | Batch dLLM `mask_modes` fixed to `[MaskMode.BLOCK_EXPANDING.value]` at both call sites | `flashinfer/dllm/batch_block_extend.py` |
| §1.2 step 2 / §3.3 | Closed product enforced: dtype `{fp16,bf16}`, idtype `{int32,int64}`, head_dim `{64,128}` (gated in URI builders); backend stays `auto/fa2/fa3`. fp8 confirmed already rejected (not coerced). | `flashinfer/dllm/batch_block_extend.py`, `flashinfer/dllm/block_extend.py` |
| §1.2 step 3 (single) | Dedicated `gen_customize_block_extend_single_prefill_module` — compiles only `kBlockExpanding`, validates closed product, exported from `flashinfer.jit.attention`. Single-path now routes through it instead of the shared `gen_customize_single_prefill_module` with `mask_modes=`. | `flashinfer/jit/attention/modules.py`, `flashinfer/jit/attention/__init__.py`, `flashinfer/dllm/block_extend.py` |
| §1 literal revert (shared jinja/config) | Reverted the shared jinja/config edits the reviewer literally called out: removed the `get_q/kv_block_expanding_offset` getters from all 4 shared jinja config files AND the PR-added members/getters from the shared `default_prefill_params.cuh` (3 structs — verified dead code, never instantiated with `kBlockExpanding`). The FA2 kernel (`prefill.cuh`) now reads the offset fields directly (`params.maybe_q_block_expanding_offset[idx]` with nullptr-check for batch, `params.q_block_expanding_offset` scalar for single) inside the existing `if constexpr (MASK_MODE == kBlockExpanding)` branches — only compiled for the dLLM variant, never for existing URIs. The Hopper/FA3 path already read `additional_params` via SFINAE traits (no getter), unchanged. **Also removed the `mask_modes` axis from the public shared `gen_customize_single/batch_prefill_module`** (impl-factored to a private `_impl` that the public fn calls with `[0,1,2,3]` and the dedicated dLLM gen calls with `[kBlockExpanding]`); the dispatcher `get_customize_batch_prefill_module` retains `mask_modes` only to ROUTE `[4]`→dedicated. **Result: the shared jinja, shared C++ params header, AND shared public gen API now contain ZERO block-expanding-specific code/axis**; the dLLM dispatch is fully standalone. | `csrc/*.jinja` (4 files), `include/flashinfer/attention/default_prefill_params.cuh`, `include/flashinfer/attention/prefill.cuh`, `flashinfer/jit/attention/modules.py`, `flashinfer/prefill.py`, `flashinfer/dllm/block_extend.py` |
| §2.2.2 (single) | Deleted `_MODULE_CACHE_WITH_OFFSET`, the `_get_aot_path`/`_check_aot_available` helpers, the hand-rolled AOT-vs-JIT branch, and the `tvm_ffi`/`jit_env`/`Path` imports. Single-path module build now delegates to `JitSpec.build_and_load()` (with its file-lock) like every other single-prefill call. | `flashinfer/dllm/block_extend.py` |
| §1.2 step 3 (batch) | Delegation switch in `get_customize_batch_prefill_module`: when `mask_modes==[kBlockExpanding]`, route to `gen_customize_block_extend_batch_prefill_module` (dedicated gen, fixed mode 4, closed product). Shared default stays `[0,1,2,3]`. Exported dedicated gens from `flashinfer.jit`. | `flashinfer/prefill.py`, `flashinfer/jit/__init__.py` |
| §2.2.1 (batch) | Named `block_diffusion=` mask option on BOTH `BatchPrefillWithPagedKVCacheWrapper` and `BatchPrefillWithRaggedKVCacheWrapper` (`__init__(block_diffusion=, dllm_block_size=)`, `plan(q_offsets=, kv_offsets=)` with auto-flip to `kBlockExpanding`, `run` injects offsets+`sm_scale`+`dllm_block_size` from `self`). This is the reviewers' preferred shape — a mask option on the existing prefill API, not a new API family. All edits guarded by `self._block_diffusion` (default False → normal users untouched). | `flashinfer/prefill.py` |
| §2 shim | `flashinfer/dllm/batch_block_extend.py` factored to a shared `build_block_diffusion_jit_args` helper; the two `BatchBlockExtend*OffsetWrapper` classes are now thin shims over the existing wrappers (no own cache / no own AOT path — those were already gone; now also share the variant wiring). | `flashinfer/dllm/batch_block_extend.py` |
| §2.2.2 (single, complete convergence) | Native `block_diffusion=` mask option on `single_prefill_with_kv_cache` (`block_diffusion=`, `dllm_block_size=`, `q_offset=`, `kv_offset=`); when set, builds the variant via `get_block_extend_module_with_offset` (dedicated gen, fixed `kBlockExpanding`) and runs through `single_prefill_with_kv_cache_with_jit_module`. `flashinfer/dllm.block_extend_attention_with_offset` is now a thin shim that just calls the native API. **Completes the reviewers' design #2** — single-request block-diffusion is now a mask option on the existing single-prefill API, not a `flashinfer/dllm/` API family. | `flashinfer/prefill.py`, `flashinfer/dllm/block_extend.py` |
| §2 test | `test_block_diffusion_named_option` (batch wrappers) + `test_block_diffusion_single_native_option` (native `single_prefill_with_kv_cache(block_diffusion=True)`, cross-checked vs reference and vs the shim). | `tests/attention/test_dllm_blockwise_mask_attention.py` |
| §2 batch parity | Closed the functional gaps between the native `block_diffusion=` path and the dedicated `BatchBlockExtend*OffsetWrapper` shim: (a) cuda-graph offset buffers — `__init__(q_offsets_buf=, kv_offsets_buf=)` + plan-time `copy_` into them under `use_cuda_graph` (parity with the shim, needed for SGLang-dLLM cuda-graph capture); (b) shape-change rebuild — plan rebuilds the variant jit module when `(head_dim, dtype, idtype)` changes across plans (`_bd_built_key`), fixing a stale-module correctness bug. All guarded by `self._block_diffusion` (default False). Remaining minor difference: per-run `sm_scale` override (the shim exposed it as an unused knob; the native wrapper uses plan-time `sm_scale` — all real callers, incl. cascade/SGLang-dLLM, set sm_scale at plan time, so this is unused). | `flashinfer/prefill.py` |

Python files pass `ast.parse`; the kernel `.cuh` change mirrors an existing
in-tree primitive. **No GPU verification done — none possible from this host
(can't compile/import flashinfer or build CUDA here). User will verify on a
GPU server.**

### Not yet done (remaining tasks)

- **§3 (none remaining)** — all three reviewer-flagged correctness items are
  addressed. The TMA-hang fix is **GPU-verifiable only on SM90** (`test_zero_visible_kv_no_hang`
  will hang on a faulty kernel); needs the H100 CI lane to confirm green.
- **Task #9 (§2.3 CI consolidation)** — the dedicated `gpu-tests-dllm-a10g/h100`
  CI lanes are KEPT for now (they run the dLLM test suite that covers the blind
  block-diffusion code; useful for the user's GPU verification). Full lane
  deletion + folding the test into the standard prefill test part is deferred —
  it needs multi-spot YAML `needs`-graph surgery that can't be CI-verified from
  this host and is best done with the maintainers' test-suite wiring knowledge.

### Remaining risk / verification

- The local commits are `ast.parse`-clean but **unverified on GPU**. Before
  merge, the existing tests (including the new `test_zero_visible_kv_no_hang`
  and `test_block_diffusion_named_option`) must run on the H100/dLLM CI lane to
  confirm: (a) the TMA-hang fix is exercised and green, (b) the idtype-in-URI
  change recompiles cleanly (first-call JIT after the URI schema change,
  discarding any stale cached dLLM module), (c) the dedicated single-path gen
  and the `block_diffusion=` named option produce output identical to the
  dedicated shim path, (d) the `block_diffusion=` run-arg injection (offsets +
  `sm_scale` + `dllm_block_size`) matches the variant kernel's expected arg
  order — this is the one piece of blind hot-path wiring that most needs a GPU
  run to confirm.