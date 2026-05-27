#!/usr/bin/env python3
"""Build a diffusers WanPipeline with FlashInfer Wan transformers."""

import argparse
import gc
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from transformer_wan_flashinfer import FlashInferWanTransformer3DModel, GEMMBackend


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _flashinfer_kwargs(args: argparse.Namespace) -> dict:
    return {
        "gemm_backend": args.gemm_backend,
        "online_act_quant": not args.offline_act_quant,
        "attention_backend": args.attention_backend,
        "use_skip_softmax_sparse": args.skip_softmax_sparse,
        "skip_softmax_threshold_scale_factor": args.skip_softmax_threshold,
    }


def _load_flashinfer_transformer(
    model_id: str,
    subfolder: str,
    dtype: torch.dtype,
    flashinfer_kwargs: dict,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    prepare_weights: bool = False,
) -> FlashInferWanTransformer3DModel:
    load_kwargs = {
        "subfolder": subfolder,
        "torch_dtype": dtype,
        **flashinfer_kwargs,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    if variant is not None:
        load_kwargs["variant"] = variant

    transformer = FlashInferWanTransformer3DModel.from_pretrained(
        model_id, **load_kwargs
    )
    transformer = transformer.to(dtype=dtype).eval()
    if prepare_weights:
        transformer.prepare_weights()
    return transformer


def load_wan_pipeline_with_flashinfer_transformers(
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    replace_transformer_2: bool = True,
    prepare_weights: bool = False,
    **flashinfer_kwargs,
):
    """Load diffusers WanPipeline and replace its denoiser(s) with FlashInfer.

    Wan2.2 T2V uses two transformer denoisers. This helper always replaces
    ``pipe.transformer`` and, when present, also replaces ``pipe.transformer_2``
    unless ``replace_transformer_2`` is false.
    """
    try:
        from diffusers import WanPipeline
    except ImportError as e:
        raise ImportError("Please install diffusers: pip install diffusers") from e

    pipe_kwargs = {"torch_dtype": dtype}
    if revision is not None:
        pipe_kwargs["revision"] = revision
    if variant is not None:
        pipe_kwargs["variant"] = variant

    pipe = WanPipeline.from_pretrained(model_id, **pipe_kwargs)

    flash_transformer = _load_flashinfer_transformer(
        model_id,
        "transformer",
        dtype,
        flashinfer_kwargs,
        revision=revision,
        variant=variant,
        prepare_weights=prepare_weights,
    )
    pipe.register_modules(transformer=flash_transformer)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if replace_transformer_2 and getattr(pipe, "transformer_2", None) is not None:
        flash_transformer_2 = _load_flashinfer_transformer(
            model_id,
            "transformer_2",
            dtype,
            flashinfer_kwargs,
            revision=revision,
            variant=variant,
            prepare_weights=prepare_weights,
        )
        pipe.register_modules(transformer_2=flash_transformer_2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if device:
        pipe.to(device)
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diffusers WanPipeline with FlashInfer Wan transformer(s)."
    )
    parser.add_argument(
        "--model-id",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Hugging Face repo id or local path for a diffusers Wan T2V pipeline.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gemm-backend",
        default=os.getenv("FLASHINFER_GEMM_BACKEND", "torch"),
        choices=[backend.value for backend in GEMMBackend],
    )
    parser.add_argument("--offline-act-quant", action="store_true")
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        choices=["auto", "single", "cudnn", "trtllm"],
    )
    parser.add_argument("--skip-softmax-sparse", action="store_true")
    parser.add_argument("--skip-softmax-threshold", type=float, default=1.0)
    parser.add_argument("--prepare-weights", action="store_true")
    parser.add_argument(
        "--keep-transformer-2",
        action="store_true",
        help="Only replace pipe.transformer; keep pipe.transformer_2 unchanged.",
    )
    parser.add_argument(
        "--prompt",
        default="A cat and a dog baking a cake together in a cozy kitchen.",
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--guidance-scale-2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-type",
        default="np",
        choices=["latent", "np", "pil"],
        help="Use latent for numeric checks, np/pil for video export.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="Run original diffusers pipeline and compare latent output against FlashInfer.",
    )
    return parser.parse_args()


def _pipeline_call_kwargs(args: argparse.Namespace, output_type: Optional[str] = None):
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "generator": generator,
        "output_type": output_type or args.output_type,
    }


def _compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    diff = (reference.float() - candidate.float()).abs()
    ref_flat = reference.flatten().float()
    cand_flat = candidate.flatten().float()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "cosine_similarity": F.cosine_similarity(
            ref_flat.unsqueeze(0), cand_flat.unsqueeze(0)
        ).item(),
    }


def _save_or_print_output(args: argparse.Namespace, frames) -> None:
    output_path = Path(args.output) if args.output is not None else None
    if args.output_type == "latent":
        if output_path is not None:
            torch.save(frames.detach().cpu(), output_path)
    else:
        from diffusers.utils import export_to_video

        if output_path is None:
            output_path = Path("wan_flashinfer.mp4")
        export_to_video(frames[0], str(output_path), fps=16)
        print(f"Saved video to {output_path}")

    if args.output_type == "latent":
        tensor = frames.detach()
        print(
            "Latent output: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"mean={tensor.float().mean().item():.6f}, "
            f"std={tensor.float().std().item():.6f}"
        )
    else:
        print(f"Generated {len(frames[0])} frames.")


def main() -> None:
    args = parse_args()
    dtype = _torch_dtype(args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if args.compare_original:
        from diffusers import WanPipeline

        pipe_kwargs = {"torch_dtype": dtype}
        if args.revision is not None:
            pipe_kwargs["revision"] = args.revision
        if args.variant is not None:
            pipe_kwargs["variant"] = args.variant

        original_pipe = WanPipeline.from_pretrained(args.model_id, **pipe_kwargs).to(
            args.device
        )
        original = original_pipe(
            **_pipeline_call_kwargs(args, output_type="latent")
        ).frames
        del original_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        original = None

    pipe = load_wan_pipeline_with_flashinfer_transformers(
        model_id=args.model_id,
        dtype=dtype,
        device=args.device,
        revision=args.revision,
        variant=args.variant,
        replace_transformer_2=not args.keep_transformer_2,
        prepare_weights=args.prepare_weights,
        **_flashinfer_kwargs(args),
    )

    output_type = "latent" if args.compare_original else args.output_type
    output = pipe(**_pipeline_call_kwargs(args, output_type=output_type))
    frames = output.frames

    if original is not None:
        metrics = _compare_tensors(original, frames)
        print(
            "Original vs FlashInfer latent metrics: "
            f"max_abs_error={metrics['max_abs_error']:.6f}, "
            f"mean_abs_error={metrics['mean_abs_error']:.6f}, "
            f"cosine_similarity={metrics['cosine_similarity']:.6f}"
        )
        if args.output is not None:
            torch.save(
                {
                    "original": original.detach().cpu(),
                    "flashinfer": frames.detach().cpu(),
                    "metrics": metrics,
                },
                Path(args.output),
            )
        return

    _save_or_print_output(args, frames)


if __name__ == "__main__":
    main()
