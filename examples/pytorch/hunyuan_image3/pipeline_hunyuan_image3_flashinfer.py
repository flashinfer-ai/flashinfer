#!/usr/bin/env python3
"""End-to-end HunyuanImage-3.0 driver using FlashInfer-swapped backbone.

This script mirrors vllm-omni's ``end2end.py`` (the user-facing entry-point
for ``tencent/HunyuanImage-3.0-Instruct``) but uses FlashInfer kernels
instead of vLLM.

What it does:
  1. Load ``tencent/HunyuanImage-3.0-Instruct`` from HuggingFace using
     ``trust_remote_code=True``.
  2. Call ``replace_backbone_with_flashinfer(...)`` to swap RMSNorm,
     QK-norm, attention, MLP, and MoE in every decoder layer.
  3. Drive the model's built-in pipelines:
     - ``text2img`` / ``img2img``: ``HunyuanImage3Text2ImagePipeline`` via
       the model's ``.pipeline`` property; saves PNGs to ``--output``.
     - ``img2text`` / ``text2text``: ``.generate(mode='gen_text', ...)`` and
       decode to text.

Run examples:

  # Text to image
  python pipeline_hunyuan_image3_flashinfer.py \\
      --modality text2img \\
      --prompts "A cute cat sitting on a windowsill" \\
      --steps 50 --output ./out

  # Image to text (captioning)
  python pipeline_hunyuan_image3_flashinfer.py \\
      --modality img2text \\
      --image-path /path/to/image.png \\
      --prompts "Describe this image."

Environment variables (same precedence as the wan example):

  FLASHINFER_GEMM_BACKEND       torch | bf16 | fp8 | ...
  FLASHINFER_ATTENTION_BACKEND  auto | single | cudnn | trtllm | sdpa
  FLASHINFER_MOE_IMPL           flashinfer | eager
  FLASHINFER_ONLINE_ACT_QUANT   1/0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modeling_hunyuan_image3_flashinfer import (  # noqa: E402
    replace_backbone_with_flashinfer,
)


# Modality -> task pair (matches vllm-omni's end2end.py).
_MODALITY_TASK_MAP = {
    "text2img": ("t2i", "think"),
    "img2img": ("it2i", "think"),
    "img2text": ("i2t", None),
    "text2text": ("t2t", None),
}

# Modality -> internal ``mode`` string the upstream model uses.
_MODALITY_MODE = {
    "text2img": "gen_image",
    "img2img": "gen_image",
    "img2text": "gen_text",
    "text2text": "gen_text",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HunyuanImage-3.0 FlashInfer end-to-end driver."
    )
    p.add_argument(
        "--model", default="tencent/HunyuanImage-3.0-Instruct",
        help="HuggingFace repo id or local path.",
    )
    p.add_argument(
        "--modality", default="text2img",
        choices=list(_MODALITY_TASK_MAP),
    )
    p.add_argument("--prompts", nargs="+", default=None)
    p.add_argument(
        "--image-path", type=str, default=None,
        help="Input image path(s) for img2img/img2text, comma-separated for multi-image (max 3).",
    )
    p.add_argument("--output", type=str, default=".")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--bot-task", type=str, default=None,
        choices=["none", "think", "recaption", "think_recaption", "vanilla"],
    )
    p.add_argument(
        "--sys-type", type=str, default=None,
        help="Override system prompt type (e.g. en_unified, en_vanilla).",
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--max-new-tokens", type=int, default=512)

    # FlashInfer config flags.
    p.add_argument(
        "--gemm-backend",
        default=os.getenv("FLASHINFER_GEMM_BACKEND", "torch"),
        help="GEMM backend for swapped linears.",
    )
    p.add_argument(
        "--attention-backend",
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        choices=["auto", "single", "cudnn", "trtllm", "sdpa"],
    )
    p.add_argument(
        "--moe-impl",
        default=os.getenv("FLASHINFER_MOE_IMPL", "flashinfer"),
        choices=["flashinfer", "eager"],
    )
    p.add_argument(
        "--offline-act-quant", action="store_true",
        help="Use fixed default activation quantization scale (FP8/FP4 backends).",
    )
    p.add_argument(
        "--skip-flashinfer", action="store_true",
        help="Load the model but don't swap to FlashInfer (for A/B comparison).",
    )

    # vae tiling / debugging.
    p.add_argument("--vae-use-tiling", action="store_true")
    p.add_argument("--verbose", type=int, default=1)

    return p.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_inputs(args: argparse.Namespace) -> tuple[list[str], list]:
    prompts = args.prompts or ["A cute cat sitting on a windowsill watching the sunset"]
    input_images: list = []
    if args.modality in ("img2img", "img2text"):
        if not args.image_path:
            raise ValueError(f"--image-path is required for {args.modality}.")
        from PIL import Image
        image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
        if len(image_paths) > 3:
            raise ValueError(
                f"--image-path accepts at most 3 images for HunyuanImage-3 IT2I, "
                f"got {len(image_paths)}."
            )
        for p in image_paths:
            if not os.path.exists(p):
                raise ValueError(f"Image path does not exist: {p}")
            input_images.append(Image.open(p).convert("RGB"))
    return prompts, input_images


def _print_config(args: argparse.Namespace, opts, prompts: list[str],
                  task: str, bot_task: Optional[str]) -> None:
    if args.verbose < 1:
        return
    print("=" * 60)
    print("HunyuanImage-3 FlashInfer driver configuration:")
    print(f"  Model: {args.model}")
    print(f"  Modality: {args.modality} (task={task}, bot_task={bot_task})")
    print(f"  dtype: {args.dtype}")
    print(f"  FlashInfer options: {opts}")
    if args.modality in ("text2img", "img2img"):
        print(f"  Steps: {args.steps}, guidance: {args.guidance_scale}, seed: {args.seed}")
        print(f"  Output size: {args.width}x{args.height}")
    if args.image_path:
        print(f"  Input image: {args.image_path}")
    print(f"  Prompts: {prompts}")
    print("=" * 60)


def _resolve_bot_task(args: argparse.Namespace, default_bot_task: Optional[str]) -> Optional[str]:
    if args.bot_task is None:
        return default_bot_task
    if args.bot_task == "none":
        return None
    return args.bot_task


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run HunyuanImage-3.")

    dtype = _torch_dtype(args.dtype)
    task, default_bot_task = _MODALITY_TASK_MAP[args.modality]
    bot_task = _resolve_bot_task(args, default_bot_task)
    mode = _MODALITY_MODE[args.modality]

    prompts, input_images = _load_inputs(args)

    # 1. Load the HF model with trust_remote_code. This pulls
    # modeling_hunyuan_image_3.py, hunyuan_image_3_pipeline.py, the VAE, the
    # SigLIP-2 ViT, and the tokenization into the local HF cache and wires
    # them into a HunyuanImage3ForCausalMM instance.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} (trust_remote_code=True) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
    )
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # The upstream model owns a ``_tokenizer`` slot it uses inside
    # ``prepare_model_inputs`` / ``generate``. ``load_tokenizer`` expects the
    # tokenizer dir, not the instance, so set the slot directly.
    model._tokenizer = tokenizer

    # Optionally enable VAE tiling to keep peak memory down for large images.
    if args.vae_use_tiling and hasattr(model, "vae") and hasattr(model.vae, "enable_tiling"):
        model.vae.enable_tiling()

    # 2. Swap the backbone hot paths to FlashInfer kernels (unless asked not to).
    if args.skip_flashinfer:
        print("[--skip-flashinfer] Running with unmodified HuggingFace model.")
        opts = None
    else:
        opts = replace_backbone_with_flashinfer(
            model,
            gemm_backend=args.gemm_backend,
            attention_backend=args.attention_backend,
            moe_impl=args.moe_impl,
            online_act_quant=not args.offline_act_quant,
            prepare_weights=True,
        )

    _print_config(args, opts, prompts, task, bot_task)

    # 3. Run the correct entry-point per modality. The upstream
    # ``HunyuanImage3ForCausalMM`` API takes one prompt at a time; loop here
    # to match the vllm-omni offline-example style.
    if args.modality == "text2img":
        _run_text2img(model, prompts, args)
    elif args.modality == "img2img":
        _run_img2img(model, prompts, input_images, args)
    elif args.modality == "img2text":
        _run_img2text(model, prompts, input_images, args, bot_task)
    elif args.modality == "text2text":
        _run_text2text(model, prompts, args, bot_task)
    else:
        raise ValueError(f"Unsupported modality {args.modality!r}.")


# ----------------------------------------------------------------------------
# Per-modality drivers
# ----------------------------------------------------------------------------


def _run_text2img(model, prompts: list[str], args: argparse.Namespace) -> None:
    """Use the upstream ``HunyuanImage3Text2ImagePipeline`` for text->image."""
    pipeline = model.pipeline
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Generating image for: {prompt!r}")
        images = pipeline(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            verbose=args.verbose,
        )
        # ``HunyuanImage3Text2ImagePipeline.__call__`` returns a list of PIL
        # images (one per batch element).
        for j, img in enumerate(images):
            save_path = Path(args.output) / f"output_{idx}_{j}.png"
            img.save(save_path)
            print(f"  saved: {save_path}")


def _run_img2img(model, prompts: list[str], input_images: list,
                 args: argparse.Namespace) -> None:
    """Image editing path: same pipeline, but with ``image`` arg."""
    pipeline = model.pipeline
    img_arg = input_images[0] if len(input_images) == 1 else input_images
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Editing image for: {prompt!r}")
        images = pipeline(
            prompt=prompt,
            image=img_arg,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            verbose=args.verbose,
        )
        for j, img in enumerate(images):
            save_path = Path(args.output) / f"output_{idx}_{j}.png"
            img.save(save_path)
            print(f"  saved: {save_path}")


def _run_img2text(model, prompts: list[str], input_images: list,
                  args: argparse.Namespace, bot_task: Optional[str]) -> None:
    """Image-conditioned text generation via the model's ``.generate`` path."""
    img_arg = input_images[0] if len(input_images) == 1 else input_images
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Captioning image with: {prompt!r}")
        inputs = model.prepare_model_inputs(
            prompt=prompt,
            image=img_arg,
            mode="gen_text",
            system_prompt=args.sys_type,
            bot_task=bot_task or "auto",
            max_new_tokens=args.max_new_tokens,
        )
        out = model.generate(
            **inputs,
            verbose=args.verbose,
            decode_text=True,
        )
        text = out if isinstance(out, str) else (out[0] if isinstance(out, list) else str(out))
        print(f"  output: {text}")


def _run_text2text(model, prompts: list[str], args: argparse.Namespace,
                   bot_task: Optional[str]) -> None:
    """Pure text-to-text generation via the model's ``.generate`` path."""
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Generating text for: {prompt!r}")
        inputs = model.prepare_model_inputs(
            prompt=prompt,
            mode="gen_text",
            system_prompt=args.sys_type,
            bot_task=bot_task or "auto",
            max_new_tokens=args.max_new_tokens,
        )
        out = model.generate(
            **inputs,
            verbose=args.verbose,
            decode_text=True,
        )
        text = out if isinstance(out, str) else (out[0] if isinstance(out, list) else str(out))
        print(f"  output: {text}")


if __name__ == "__main__":
    main()
