# Deprecated-API deletion survey — eligibility at v0.7.0

> Point-in-time record, kept for bookkeeping. It documents which deprecated APIs were eligible for
> removal at the 0.7.0 boundary and why, so the next sweep does not repeat the archaeology. It is
> **not** a policy document — see "What this document is not" below. Not published to the docs site
> (`docs/conf.py` sets `source_suffix = [".rst"]`, so Markdown here is repo-only).

**Rule:** deprecated in **0.5.x or earlier** → eligible now (two minor versions). 0.6.x → wait for 0.8.
Every version below is **evidence**: `git log -S'<exact marker>' --reverse` for the introducing commit,
then earliest non-nightly tag via `git tag --contains`.

## 0. Why this document exists

**Produced:** 2026-09-13, during the v0.7.0 release cycle, requested by the release manager while
rc2 was being prepared. **Question it answers:** *which deprecated APIs are we allowed to delete in 0.7.0, and
which would break a downstream consumer if we did?*

**Why now, and not any earlier release.** FlashInfer guarantees a deprecated public API for two
minor releases. 0.7.0 is therefore the first release in which everything deprecated in 0.5.x or
earlier becomes removable — and that window had never been swept systematically, so the deprecated
surface had been accumulating across roughly five minor releases. Some of these symbols have carried
a deprecation notice since 0.1.6 (August 2024) and have simply never been collected.

**What this document is.** A one-time inventory plus an eligibility ruling per symbol, with the
evidence attached. It is deliberately evidence-first: every "deprecated in X.Y" below was
re-derived from git (`git log -S'<exact marker>' --reverse` for the introducing commit, then the
earliest non-nightly tag via `git tag --contains`) rather than read off a docstring. That distinction
is not pedantry — §3 documents a marker that claims 0.6.18 for a change that first shipped in
0.7.0rc1, which would have made two symbols look eligible two releases early.

### The rule, stated precisely

This is an internal document, so the convention is written out exactly as we apply it. (It is
deliberately **not** written this precisely in the public `CONTRIBUTING.md` — see the note below.)

1. **Window.** A deprecated public API is guaranteed for **two minor releases**, then may be removed.
   Deprecated in 0.5.x → removable at **0.7.0**. Deprecated in 0.6.x → removable at **0.8.0**.
   Deprecated in 0.7.x → removable at **0.9.0**.
2. **When the clock starts.** At the release the deprecation *shipped* in — not the release it was
   merged into, and not whatever a docstring claims. Establish it with
   `git log -S'<exact marker>' --reverse` for the introducing commit, then
   `git tag --contains <sha> | grep -v rc` for the earliest non-nightly tag. §3 is the case that
   proves why: a marker reading `.. deprecated:: 0.6.18` for a commit that first shipped in
   `v0.7.0rc1` would have made two symbols look eligible two full releases early.
3. **Downstream interception (a veto, not a tiebreak).** Age makes a symbol *eligible*; it does not
   authorize removal. Check current vLLM and SGLang source — live, not a local checkout, ours is
   months stale (§6). A hit on a **serving path** blocks removal until that usage is migrated
   upstream, no matter how old the deprecation is. A hit confined to a **benchmark, test, or
   example** does not block. This rule alone demoted items #6 and #7.
4. **Deletion is all-or-nothing per symbol group.** If removing X orphans Y — a `destroy_` with no
   `create_`, a helper with no caller — Y goes in the same PR or X does not go at all.

**Why the public guide stays vague.** FlashInfer aligns with vLLM's release and deprecation scheme,
and vLLM's own policy does not pin the number down. Matching their level of precision in
`CONTRIBUTING.md` ("a limited number of minor releases", plus a link) is intentional: a harder public
rule becomes a commitment we would be held to and a divergence from the thing we track. Precision
belongs here, in the working notes, not there.

**What this document is not.** Not a commitment to delete — §1 is a recommendation, and the release
manager decides what actually ships. Not a permanent artifact; once the deletions land it is history, and the next sweep (for
0.6.x deprecations, at 0.8.0) should be a fresh survey rather than an edit of this one.

**Why the vLLM / SGLang columns exist.** Eligibility by age is necessary but not sufficient. These
APIs are load-bearing for downstream serving stacks, and deleting a symbol that SGLang calls on a
serving path breaks it at import or at first token regardless of how long we have advertised the
deprecation. So every candidate was intercepted against both consumers before being recommended. That
check is what demoted items #6 and #7 — the largest and most tempting group, 16 symbols deprecated
since 0.1.6 — from "obviously delete" to "wait for 0.8, and send SGLang a migration PR first."
The caveats on how that check was performed are in §6, stated rather than hidden.

## 1. ELIGIBLE NOW — lowest risk first

| # | symbol | deprecated in | replacement | vLLM | SGLang | risk | suggested |
|---|--------|---------------|-------------|------|--------|------|-----------|
| 1 | `comm.trtllm_custom_all_reduce` (`trtllm_ar.py:890`) | **0.5.0** (`7d9d7aff`, PR #1991) | `trtllm_allreduce_fusion` | NOT FOUND | NOT FOUND | LOW | **DELETE NOW** |
| 2 | `comm.trtllm_create_ipc_workspace_for_all_reduce` (`trtllm_ar.py:440`) | **0.5.0** (same commit) | `..._fusion` | NOT FOUND | NOT FOUND | LOW | **DELETE NOW** (pair with #1) |
| 3 | 9 × no-op `end_forward` (`decode/prefill/sparse/cascade/pod`) | **0.1.6** (`d940d2e0`, PR #466) | n/a — body is `pass` | NOT FOUND | USED ×1 — *benchmark only* | LOW | **DELETE NOW** |
| 4 | `data_type` param on `plan()` (decode/prefill/pod) | **0.2.0** (`78e26e47`, PR #542) | `q_data_type`/`kv_data_type` | NOT FOUND | NOT FOUND | LOW | **DELETE NOW** |
| 5 | `decode.BatchDecodeMlaWithPagedKVCacheWrapper` (`decode.py:2772`) | **0.2.1** (`88fa03f3`, PR #818) | `mla.BatchMLAPagedAttentionWrapper` | NOT FOUND | NOT FOUND | MEDIUM — top-level export in `flashinfer/__init__.py:59` | **DELETE IN 0.7.0** (minor boundary is the right moment) |
| 6 | 9 × legacy `forward` / `forward_return_lse` | **0.1.6** (`d940d2e0`) | `run` / `run_return_lse` | NOT FOUND | **USED — SERVING PATH** | **HIGH** | **WAIT FOR 0.8** + upstream SGLang PR first |
| 7 | 7 × `begin_forward = plan` class aliases | **0.1.6** (`d940d2e0`) | `plan` | NOT FOUND (low confidence) | **USED — SERVING PATH** ×14 | **HIGH** | **WAIT FOR 0.8**; add a real `DeprecationWarning` in 0.7.x |

SGLang serving-path hits for #6/#7: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py:125,142,166,176,869`
and `flashinfer_backend.py:846`. Deleting these in 0.7.0 breaks SGLang.

## 2. NEEDS A DECISION THIS RELEASE

**`gated_delta_rule_mtp`** — deprecated 0.6.7 (`7cb016df`, PR #2730). Its marker promises the implicit
`intermediate_states_buffer=True` default flips **"in version 0.7.0"** — i.e. the release being cut —
and **SGLang imports it** (`srt/layers/attention/linear/kernels/gdn_flashinfer.py:44,49` + 2 more).
Either honour the promise now or amend the marker; leaving it is a documented-but-unkept commitment.

## 3. A DOCSTRING THAT CONTRADICTS GIT

`cute_dsl/sparse/bsa_attn_sm100_blk128.py:255` and `bsa_attn_sm100_blk64.py:476` declare
`.. deprecated:: 0.6.18`. But `git merge-base --is-ancestor 05e5d927 v0.6.18` → **not an ancestor**:
v0.6.18 was tagged 2026-08-28, the commit landed after the branch cut and first shipped in `v0.7.0rc1`.
Taking the docstring at face value would make these look eligible in 0.8 when they are not eligible
until **0.9**. **Trust git, not the docstring.**

## 4. TOOLING GAP — the largest deprecated surface is unmonitored

`scripts/pr_checks/inspect_sources.py:107` `is_function_deprecated` scans **`tree.body` only —
module-level functions**. Extending it to classes/methods yields zero *additional* detector hits, which
means:

- **All 21 legacy `forward`/`begin_forward`/`end_forward`/`forward_return_lse` methods are invisible**
  to the tooling — they use plain prose (`Warning: this function is deprecated...`), matching neither
  the decorator nor `_DEPRECATED_DOCSTRING_RE`.
- **Worse: `begin_forward = plan` (7 sites) and `forward = run` (`cascade.py:558`,
  `gemm/gemm_base.py:2996`) are bare class-attribute aliases** — no docstring, no decorator, and
  **no `DeprecationWarning` at runtime**. Deprecated ~5 minor versions; callers get zero signal. SGLang
  calls these on its serving path.
- Module-level `__getattr__` shims (`fused_moe/runners.py:6121`, `cute_dsl/tuner.py:1289`,
  `fused_moe/prepare.py:2523`) are invisible too — the symbol is not an AST node at all.

Fixes: extend `collect_deprecated_symbols` to methods/classes; add `Warning:` prose to
`_DEPRECATED_DOCSTRING_RE`; require a version in every marker (`.. deprecated:: X.Y`) so the next
survey is a lookup rather than archaeology.

## 5. Deprecated in 0.7.0 itself → earliest deletion 0.9

`cute_dsl_fused_moe_nvfp4`, `cute_dsl_fused_moe_mxfp8_mxfp4`, `CuteDslMxfp8Mxfp4MoEWrapper`,
`CuteDslNvfp4Runner`, `CuteDslFusedMoENvfp4Runner`, `prepare_cute_dsl_nvfp4_weights`,
`quant_mode='nvfp4'` (all `f7d4b167`, PR #4793); MLA positional args (`8b118e75`, PR #4697);
`bsa_attn_fwd`/`bsa_attn_blk64_fwd` (`05e5d927`, PR #4590).
**`cute_dsl_fused_moe_nvfp4` is USED by vLLM** (`vllm/utils/flashinfer.py:132,263`) — so vLLM starts
seeing a deprecation warning from 0.7.0.

## 6. Evidence caveats (stated, not hidden)

- Local vLLM/SGLang checkouts are **2026-04-06, ~5 months stale**; cross-checked against live GitHub
  code search.
- `forward_return_lse` hit the 10/min code-search rate limit on the first pass; re-run succeeded
  (vLLM zero files, SGLang two).
- vLLM's live `begin_forward` hits are in `vllm/v1/hisparse/`, absent from the snapshot and on
  inspection **vLLM's own API**, not a FlashInfer wrapper → rated NOT FOUND with **low confidence**.
- A pathspec-scoped `git log` initially mis-dated the whole legacy `forward` family to v0.2.0; that was
  a file-move artifact (`python/flashinfer/` → `flashinfer/`). Repo-wide re-run gives the true origin
  **d940d2e0, PR #466, 2024-08-25, first tag v0.1.6**.
