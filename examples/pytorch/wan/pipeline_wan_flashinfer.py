#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a diffusers WanPipeline with FlashInfer Wan transformers."""

import argparse
import gc
import os
from pathlib import Path
from typing import Optional

import torch

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
    prepare_weights: bool = False,
    **flashinfer_kwargs,
):
    """Load diffusers WanPipeline and replace its denoiser(s) with FlashInfer."""
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

    if getattr(pipe, "transformer_2", None) is not None:
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
        help=(
            "GEMM backend (base name from GEMMBackend, e.g. "
            + ", ".join(backend.value for backend in GEMMBackend)
            + "; optional '-<kernel>' suffix like 'fp4-cudnn' is forwarded "
            "to the kernel's own backend kwarg)."
        ),
    )
    parser.add_argument("--offline-act-quant", action="store_true")
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        help=(
            "Attention backend (auto|single|cudnn|trtllm|torch); '-<kernel>' "
            "suffix on 'single' (e.g. 'single-fa3') is forwarded to "
            "single_prefill_with_kv_cache."
        ),
    )
    parser.add_argument("--skip-softmax-sparse", action="store_true")
    parser.add_argument("--skip-softmax-threshold", type=float, default=1.0)
    parser.add_argument("--prepare-weights", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="gaudy colors, overexposed, static, blurry details, subtitles, style, artwork, painting, image, still, washed out, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, three legs, crowded background, walking backwards",
    )

    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-type",
        default="np",
        choices=["latent", "np", "pil"],
        help="Use latent for numeric checks, np/pil for video export.",
    )
    parser.add_argument("--output", default=None)
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

    pipe = load_wan_pipeline_with_flashinfer_transformers(
        model_id=args.model_id,
        dtype=dtype,
        device=args.device,
        revision=args.revision,
        variant=args.variant,
        prepare_weights=args.prepare_weights,
        **_flashinfer_kwargs(args),
    )

    output = pipe(**_pipeline_call_kwargs(args, output_type=args.output_type))
    frames = output.frames

    _save_or_print_output(args, frames)


if __name__ == "__main__":
    main()
