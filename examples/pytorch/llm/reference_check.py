"""Numerical reference check for the FlashInfer LLM example.

Builds tiny random-weight models (dense Qwen3 and Qwen3-MoE) with
``transformers``, saves them as normal HF checkpoints, loads them with this
example's FlashInfer-based implementation, and compares last-token prefill
logits between the two — in BF16 — for a batch of random prompts of varying
lengths.

This validates the full numerical wiring (attention over the paged KV cache,
RoPE, norms, SwiGLU, MoE routing + ``cutlass_fused_moe`` expert dispatch)
without downloading real checkpoints: a tiny random MoE exercises exactly
the same kernel paths as Qwen3-30B-A3B. Kernel-vs-eager BF16 accumulation
differences are expected; the assertion is a relative-L2 bound per prompt
(the HF bf16-vs-fp32 noise floor for these models is ~0.007).

Single GPU:
    python reference_check.py                    # dense + moe tiny models
    python reference_check.py --arch moe --rel-tol 0.05

TEP (TP=EP) across N GPUs — attention tensor-parallel, MoE expert-parallel,
one allreduce after attention and one after the FFN/MoE; rank 0 compares
against the single-GPU transformers reference:
    torchrun --nproc_per_node=2 reference_check.py

Emits ``[smoke] refcheck_<arch>=pass|fail`` lines; exit code 0/1.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import torch

from generate import GenerationEngine, maybe_init_distributed
from modeling import FlashInferLLM

# Tiny but kernel-legal shapes: head_dim 64/128 for attention; hidden and
# (moe_)intermediate must be multiples of 8 for the BF16 cutlass MoE path.
# Head counts / expert count divide by 2 so the same configs run under TP=2.
COMMON = dict(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    vocab_size=2048,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    tie_word_embeddings=False,
    attention_bias=False,
    use_cache=False,
)


def build_tiny_checkpoint(arch: str, tmpdir: str, seed: int) -> str:
    torch.manual_seed(seed)
    if arch == "dense":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(**COMMON)
        model = Qwen3ForCausalLM(cfg)
    elif arch == "moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        cfg = Qwen3MoeConfig(
            **COMMON,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            router_aux_loss_coef=0.0,
        )
        model = Qwen3MoeForCausalLM(cfg)
    else:
        raise ValueError(arch)
    model = model.to(torch.bfloat16).eval()
    path = f"{tmpdir}/tiny_{arch}"
    model.save_pretrained(path, safe_serialization=True)
    return path


def check_arch(
    arch: str, rel_tol: float, seed: int, world_size: int, rank: int
) -> bool:
    device = torch.device("cuda", torch.cuda.current_device())
    tmpdir = ckpt = None
    if rank == 0:
        tmpdir = tempfile.mkdtemp()
        ckpt = build_tiny_checkpoint(arch, tmpdir, seed)
    if world_size > 1:
        obj = [ckpt]
        torch.distributed.broadcast_object_list(obj, src=0)
        ckpt = obj[0]

    try:
        our_model = FlashInferLLM.from_pretrained(
            ckpt, device, tp_size=world_size, tp_rank=rank
        )
        engine = GenerationEngine(our_model)

        g = torch.Generator().manual_seed(seed + 1)
        vocab = our_model.config.vocab_size
        prompts = [
            torch.randint(0, vocab, (n,), generator=g).tolist()
            for n in (7, 33, 61, 128)
        ]

        # Collective under TP: every rank participates; logits identical on
        # all ranks after the final allreduce.
        ours = engine.prefill_logits(prompts).float()

        ok = True
        if rank == 0:
            from transformers import AutoModelForCausalLM

            ref_model = (
                AutoModelForCausalLM.from_pretrained(
                    ckpt, dtype=torch.bfloat16, attn_implementation="eager"
                )
                .to(device)
                .eval()
            )
            for i, ids in enumerate(prompts):
                with torch.inference_mode():
                    ref = (
                        ref_model(torch.tensor([ids], device=device))
                        .logits[0, -1]
                        .float()
                    )
                rel = (ours[i] - ref).norm() / ref.norm()
                top1_match = ours[i].argmax().item() == ref.argmax().item()
                status = "ok" if rel <= rel_tol else "MISMATCH"
                print(
                    f"  {arch} len={len(ids):>3}: rel_l2={rel:.4f} "
                    f"top1_match={top1_match} [{status}]"
                )
                ok &= rel <= rel_tol
            del ref_model
        del our_model, engine
        torch.cuda.empty_cache()
        if world_size > 1:  # share the verdict so every rank exits alike
            flag = torch.tensor([int(ok)], device=device)
            torch.distributed.broadcast(flag, src=0)
            ok = bool(flag.item())
        return ok
    finally:
        if rank == 0 and tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["dense", "moe", "all"], default="all")
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.05,
        help="max relative L2 error of last-token logits vs transformers eager",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    world_size, rank = maybe_init_distributed()
    if rank != 0:  # keep stdout single-writer; stderr stays visible
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 — process-lifetime sink

    mode = f"TEP tp=ep={world_size}" if world_size > 1 else "single GPU"
    archs = ["dense", "moe"] if args.arch == "all" else [args.arch]
    failed = []
    for arch in archs:
        print(f"reference check ({mode}): tiny {arch} vs transformers eager (bf16)")
        ok = check_arch(arch, args.rel_tol, args.seed, world_size, rank)
        print(f"[smoke] refcheck_{arch}={'pass' if ok else 'fail'}")
        if not ok:
            failed.append(arch)
    if world_size > 1:
        torch.distributed.destroy_process_group()
    if failed:
        print(f"FAIL: logits mismatch for: {', '.join(failed)}")
        sys.exit(1)
    print("PASS: FlashInfer path matches transformers reference within tolerance")


if __name__ == "__main__":
    main()
