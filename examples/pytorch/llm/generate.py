"""Batch text generation with the FlashInfer LLM example model.

Usage:
    python generate.py --model-id Qwen/Qwen3-0.6B --max-tokens 32
    python generate.py --model-id Qwen/Qwen3-8B --prompt "The capital of France is" \
        --temperature 0.7 --top-k 50 --top-p 0.9

Besides generating text, this script emits machine-readable ``[smoke] key=value``
lines consumed by ``smoke_test.py``:

- ``jit_builds_total`` — number of FlashInfer JIT modules actually *compiled*
  in this process (the built artifact changed during ``JitSpec.build()``).
  On a warm cache this must be 0.
- ``jit_build_calls`` — informational: ``build()`` invocations, i.e. ninja
  dependency scans. By design this equals the number of JIT modules used
  even on a warm cache (``try_load`` defers freshness to ninja).
- ``jit_builds_steady`` — compilations triggered after the first decode step
  (must always be 0: steady-state decoding must not recompile anything).
- ``tokens_<i>`` — generated token ids per request, for determinism checks.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from typing import Dict, List, Optional

import torch

import flashinfer

from modeling import FlashInferLLM, PagedKVCache, resolve_checkpoint_dir

DEFAULT_PROMPTS = [
    "The capital of France is",
    "The three primary colors are",
    "To bake bread you need flour, water,",
]


def maybe_init_distributed() -> tuple:
    """Initialize NCCL for torchrun launches (TEP: TP=EP over all ranks).

    Returns (world_size, rank). Single-process runs return (1, 0) untouched.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return world_size, torch.distributed.get_rank()


def install_jit_build_counter() -> Dict:
    """Count actual JIT compilations (not mere cache probes) in this process.

    Nuance discovered on first deployment of this harness:
    ``JitSpecNvcc.try_load()`` intentionally returns ``None`` for JIT-path
    modules — artifact freshness is owned by ninja's dependency scan — so
    ``build()`` runs in *every* process and ninja no-ops when the cached
    ``.so`` is up to date (see flashinfer/jit/core.py). Counting ``build()``
    calls therefore counts ninja invocations, not recompiles. A module was
    actually (re)compiled iff its built artifact changed across the
    ``build()`` call, so that is what ``count``/``names`` track;
    ``build_calls`` keeps the raw invocation count as an informational
    warm-start metric.
    """
    import flashinfer.jit.core as jit_core

    # Registers the CuTe-DSL JitSpec subclass if the DSL is installed.
    with contextlib.suppress(Exception):
        import flashinfer.jit.cute_dsl_core  # noqa: F401

    counter: Dict = {"count": 0, "names": [], "build_calls": 0}

    def _artifact_stamp(spec):
        try:
            lib = spec.get_library_path()
            return lib.stat().st_mtime_ns if lib.exists() else None
        except Exception:
            return None

    def _wrap(cls):
        orig = cls.__dict__.get("build")
        if orig is None:
            return

        def build(self, *args, _orig=orig, **kwargs):
            counter["build_calls"] += 1
            before = _artifact_stamp(self)
            result = _orig(self, *args, **kwargs)
            if _artifact_stamp(self) != before:
                counter["count"] += 1
                counter["names"].append(self.name)
            return result

        cls.build = build

    seen = set()
    stack = [jit_core.JitSpec]
    while stack:
        c = stack.pop()
        for sub in c.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                _wrap(sub)
                stack.append(sub)
    return counter


def sampling_support(x: torch.Tensor, top_k: int, top_p: float):
    """Reference top-k ∩ nucleus support for temperature-scaled logits ``x``.

    Returns ``(allowed, p_top1)``: a bool mask over the vocabulary of tokens the
    sampler is permitted to return, and the greedy token's probability under the
    filtered+renormalized distribution the kernel actually draws from.

    Both boundaries close over ties (``>=`` against a *value*, not a rank), and
    the ``eps`` slack is applied on the permissive side only — the mask is a
    superset of what any correct implementation may keep, so a violation is
    unambiguous. This mirrors ``tests/utils/test_sampling.py``, which builds the
    same masks with sort/cumsum and asserts membership over many draws.

    The renormalization denominator is the *tie closure* of the top-k, not the
    k largest entries: if two logits collide exactly at the k-th position the
    kernel may legitimately keep both, and a narrower denominator would produce
    rare false failures.
    """
    eps = 1e-4
    vocab = x.shape[-1]
    k = min(top_k if top_k > 0 else vocab, vocab)
    values, ids = torch.topk(x, k, dim=-1)  # descending
    # Tie closure of the top-k, renormalized — the largest set the kernel may keep.
    probs = torch.softmax(x.masked_fill(x < values[:, -1:], float("-inf")), dim=-1)
    kept = probs.gather(1, ids)
    mass_above = torch.cumsum(kept, dim=-1) - kept
    n_keep = (mass_above < top_p + eps).sum(dim=-1, keepdim=True).clamp(min=1)
    cutoff = values.gather(1, n_keep - 1)
    allowed = x >= cutoff
    kept_mass = torch.cumsum(kept, dim=-1).gather(1, n_keep - 1).clamp(min=1e-20)
    return allowed, (kept[:, :1] / kept_mass).squeeze(-1)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
    audit: Optional[Dict] = None,
) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1).to(torch.int32)
    logits = logits / temperature
    vocab = logits.shape[-1]
    effective_top_k = top_k if top_k > 0 else vocab
    tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, effective_top_k, top_p, deterministic=True, generator=generator
    )
    if audit is not None:
        _audit_sample(audit, logits, tokens, top_k, top_p)
    return tokens


def _audit_sample(audit: Dict, x: torch.Tensor, tokens: torch.Tensor, top_k, top_p):
    """Accumulate support-membership evidence for one sampled step."""
    rows, vocab = x.shape
    tok = tokens.long()
    in_range = (tok >= 0) & (tok < vocab)
    audit["out_of_range"] += int((~in_range).sum())
    if not bool(in_range.all()):  # keep the gather below safe
        tok = tok.clamp(0, vocab - 1)
    allowed, p_top1 = sampling_support(x, top_k, top_p)
    audit["violations"] += int((~allowed.gather(1, tok[:, None]).squeeze(-1)).sum())
    audit["draws"] += rows
    audit["allowed_total"] += int(allowed.sum())
    # A sampler that has silently degenerated to argmax is still always inside
    # the support, so membership cannot see it; count divergences from greedy
    # against how many we should expect from the filtered distribution.
    audit["divergences"] += int((tok != x.argmax(dim=-1)).sum())
    audit["expected_divergences"] += float((1.0 - p_top1).sum())


class GenerationEngine:
    def __init__(
        self,
        model: FlashInferLLM,
        page_size: int = 16,
        prefill_backend: str = "auto",
        decode_backend: str = "auto",
    ):
        self.model = model
        self.page_size = page_size
        device = model.device
        # Separate 128 MB workspaces; the decode one must start zeroed.
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend=prefill_backend,
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            use_tensor_cores=True,
            backend=decode_backend,
        )

    def _make_cache(
        self, prompt_token_ids: List[List[int]], max_new_tokens: int
    ) -> PagedKVCache:
        cfg, device = self.model.config, self.model.device
        max_pages = sum(
            math.ceil((len(p) + max_new_tokens) / self.page_size)
            for p in prompt_token_ids
        )
        kv = PagedKVCache(
            cfg,
            max_pages,
            self.page_size,
            device,
            self.model.dtype,
            num_kv_heads=self.model.num_local_kv_heads,
        )
        for ids in prompt_token_ids:
            req = kv.add_request()
            kv.extend(req, len(ids))
        return kv

    def _prefill(
        self, prompt_token_ids: List[List[int]], kv: PagedKVCache
    ) -> torch.Tensor:
        """Run the batched causal prefill; returns last-token logits [batch, vocab]."""
        model, cfg, device = self.model, self.model.config, self.model.device
        input_ids = torch.tensor(
            [t for ids in prompt_token_ids for t in ids],
            dtype=torch.int64,
            device=device,
        )
        qo_indptr_list = [0]
        for ids in prompt_token_ids:
            qo_indptr_list.append(qo_indptr_list[-1] + len(ids))
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=device)
        nnz = input_ids.shape[0]
        kv_indptr, kv_indices, kv_last_page_len = kv.page_table_tensors()
        batch_indices, pos_ids = flashinfer.get_batch_indices_positions(
            qo_indptr, kv.seq_lens_tensor(), nnz
        )
        self.prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            model.num_local_qo_heads,
            model.num_local_kv_heads,
            cfg.head_dim,
            self.page_size,
            causal=True,
            q_data_type=model.dtype,
            kv_data_type=model.dtype,
        )
        return model.forward(
            input_ids,
            pos_ids,
            self.prefill_wrapper,
            kv,
            batch_indices,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            last_token_rows=(qo_indptr[1:].to(torch.int64) - 1),
        )

    @torch.inference_mode()
    def prefill_logits(self, prompt_token_ids: List[List[int]]) -> torch.Tensor:
        """Last-token prefill logits per prompt (no decode).

        Used by reference checks that compare against another implementation.
        """
        kv = self._make_cache(prompt_token_ids, max_new_tokens=1)
        return self._prefill(prompt_token_ids, kv)

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        jit_counter: Optional[Dict] = None,
        audit: Optional[Dict] = None,
    ) -> Dict:
        model, cfg, device = self.model, self.model.config, self.model.device
        batch = len(prompt_token_ids)
        page_size = self.page_size
        num_q, num_kv, hd = (
            model.num_local_qo_heads,
            model.num_local_kv_heads,
            cfg.head_dim,
        )
        generator = None
        if temperature > 0.0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        kv = self._make_cache(prompt_token_ids, max_new_tokens)

        # ---- Prefill: all prompts in one ragged batch ----
        t_start = time.perf_counter()
        logits = self._prefill(prompt_token_ids, kv)
        next_tokens = sample_tokens(logits, temperature, top_k, top_p, generator, audit)
        if audit is not None and temperature > 0.0:
            # Hermetic replay: same logits, a freshly seeded generator. Model
            # independent, so a mismatch is unambiguously the sampler (a
            # generator state that is not honored, or a deterministic=True that
            # does not actually pin the reduction order).
            replay_gen = torch.Generator(device=device)
            replay_gen.manual_seed(seed)
            replay = sample_tokens(logits, temperature, top_k, top_p, replay_gen)
            audit["replay_match"] = int(bool(torch.equal(replay, next_tokens)))
            # Per-request tensor params take the non-fast-path kernels
            # (top_k_mask_logits + full-vocab top_p_sampling_from_probs); a
            # scalar top_k on a large vocab only ever exercises the fast path.
            x = logits / temperature
            vocab = x.shape[-1]
            k_t = torch.full(
                (x.shape[0],), top_k if top_k > 0 else vocab, device=device
            )
            p_t = torch.full((x.shape[0],), top_p, device=device)
            per_req = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                x, k_t, p_t, deterministic=True, generator=replay_gen
            )
            allowed, _ = sampling_support(x, top_k, top_p)
            tok = per_req.long().clamp(0, vocab - 1)
            audit["perreq_violations"] = int(
                (~allowed.gather(1, tok[:, None]).squeeze(-1)).sum()
            )
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        eos = set(cfg.eos_token_ids)
        outputs: List[List[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        for i, tok in enumerate(next_tokens.tolist()):
            outputs[i].append(tok)
            finished[i] = tok in eos

        # ---- Decode: one token per request per step ----
        builds_after_first_decode: Optional[int] = None
        decode_rows = torch.arange(batch, dtype=torch.int64, device=device)
        append_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
        steps = 0
        for _ in range(max_new_tokens - 1):
            if all(finished):
                break
            for req in range(batch):
                kv.extend(req, 1)
            kv_indptr, kv_indices, kv_last_page_len = kv.page_table_tensors()
            self.decode_wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_q,
                num_kv,
                hd,
                page_size,
                q_data_type=model.dtype,
                kv_data_type=model.dtype,
            )
            batch_indices, pos_ids = flashinfer.get_batch_indices_positions(
                append_indptr, kv.seq_lens_tensor(), batch
            )
            logits = model.forward(
                next_tokens.to(torch.int64),
                pos_ids,
                self.decode_wrapper,
                kv,
                batch_indices,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                last_token_rows=decode_rows,
            )
            next_tokens = sample_tokens(
                logits, temperature, top_k, top_p, generator, audit
            )
            for i, tok in enumerate(next_tokens.tolist()):
                if not finished[i]:
                    outputs[i].append(tok)
                    finished[i] = tok in eos
            steps += 1
            if steps == 1 and jit_counter is not None:
                torch.cuda.synchronize()
                builds_after_first_decode = jit_counter["count"]
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        jit_builds_steady = 0
        if jit_counter is not None and builds_after_first_decode is not None:
            jit_builds_steady = jit_counter["count"] - builds_after_first_decode
        return {
            "outputs": outputs,
            "finished": finished,
            "prefill_seconds": t_prefill - t_start,
            "decode_seconds": t_end - t_prefill,
            "decode_steps": steps,
            "decode_tokens": steps * batch,
            "jit_builds_steady": jit_builds_steady,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt (repeatable). Defaults to a small built-in batch.",
    )
    parser.add_argument("--chat", action="store_true", help="Apply the chat template")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 = greedy (default)"
    )
    parser.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--check-sampling",
        action="store_true",
        help="Audit every sampled token against a torch reference support "
        "(top-k ∩ nucleus). Adds a topk+softmax per step; off by default.",
    )
    parser.add_argument("--prefill-backend", default="auto")
    parser.add_argument("--decode-backend", default="auto")
    args = parser.parse_args()

    audit = None
    if args.check_sampling and args.temperature > 0.0:
        audit = {
            "violations": 0,
            "out_of_range": 0,
            "draws": 0,
            "allowed_total": 0,
            "divergences": 0,
            "expected_divergences": 0.0,
            "replay_match": -1,
            "perreq_violations": -1,
        }

    jit_counter = install_jit_build_counter()
    world_size, rank = maybe_init_distributed()
    if rank != 0:  # keep stdout single-writer; stderr stays visible
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 — process-lifetime sink
    device = torch.device("cuda", torch.cuda.current_device())
    ckpt_dir = resolve_checkpoint_dir(args.model_id)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    t0 = time.perf_counter()
    model = FlashInferLLM.from_pretrained(
        str(ckpt_dir), device, tp_size=world_size, tp_rank=rank
    )
    tp_note = f", tep tp=ep={world_size}" if world_size > 1 else ""
    print(
        f"Loaded {args.model_id} ({model.config.model_type}, "
        f"{model.config.num_hidden_layers} layers{tp_note}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    prompts = args.prompt or DEFAULT_PROMPTS
    if args.chat:
        prompt_token_ids = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            # Render to text, then tokenize explicitly: apply_chat_template's
            # tokenized return type varies across transformers versions.
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:  # template without enable_thinking support
                text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            prompt_token_ids.append(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            )
        eos_extra = tokenizer.eos_token_id
        if eos_extra is not None and eos_extra not in model.config.eos_token_ids:
            model.config.eos_token_ids.append(eos_extra)
    else:
        prompt_token_ids = [tokenizer(p)["input_ids"] for p in prompts]

    engine = GenerationEngine(
        model,
        page_size=args.page_size,
        prefill_backend=args.prefill_backend,
        decode_backend=args.decode_backend,
    )
    result = engine.generate(
        prompt_token_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        audit=audit,
        jit_counter=jit_counter,
    )

    for i, (prompt, out_ids) in enumerate(zip(prompts, result["outputs"], strict=True)):
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(f"\n--- request {i} ({'eos' if result['finished'][i] else 'length'}) ---")
        print(f"prompt:     {prompt!r}")
        print(f"completion: {text!r}")

    decode_tps = result["decode_tokens"] / max(result["decode_seconds"], 1e-9)
    print(
        f"\nprefill {result['prefill_seconds']:.3f}s | "
        f"decode {result['decode_steps']} steps, {decode_tps:.1f} tok/s "
        f"(batch={len(prompts)}; includes first-call JIT warmup)"
    )

    if audit is not None:
        print(f"[smoke] sample_draws={audit['draws']}")
        print(f"[smoke] sample_violations={audit['violations']}")
        print(f"[smoke] sample_out_of_range={audit['out_of_range']}")
        # Mean admissible set size: guards against a vacuous membership check
        # (with --top-k 0 --top-p 1.0 the mask is the whole vocab).
        mean_allowed = audit["allowed_total"] / max(audit["draws"], 1)
        print(f"[smoke] sample_allowed_mean={mean_allowed:.1f}")
        print(f"[smoke] sample_divergences={audit['divergences']}")
        print(
            f"[smoke] sample_expected_divergences={audit['expected_divergences']:.2f}"
        )
        print(f"[smoke] sample_replay_match={audit['replay_match']}")
        print(f"[smoke] sample_perreq_violations={audit['perreq_violations']}")
    print(f"[smoke] jit_builds_total={jit_counter['count']}")
    print(f"[smoke] jit_builds_names={','.join(jit_counter['names'])}")
    print(f"[smoke] jit_build_calls={jit_counter['build_calls']}")
    print(f"[smoke] jit_builds_steady={result['jit_builds_steady']}")
    for i, out_ids in enumerate(result["outputs"]):
        print(f"[smoke] tokens_{i}={','.join(map(str, out_ids))}")
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
