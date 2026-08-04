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
import contextlib
import gc
import os
import time
from pathlib import Path
from typing import Optional

import torch

from transformer_wan_flashinfer import (
    FlashInferWanTransformer3DModel,
    GEMMBackend,
    set_ulysses_communicator,
)


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
        "use_vsa": args.use_vsa,
        "vsa_sparsity": args.vsa_sparsity,
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

    # Skip diffusers' own denoiser load: we replace it immediately anyway, and
    # instantiating a second 14B transformer doubles peak host memory. It also
    # sidesteps checkpoints whose sharded weights ship without an
    # ``*.index.json`` (the FastVideo VSA releases), which diffusers can't load
    # but ``_load_checkpoint_state_dict`` can.
    model_index = WanPipeline.load_config(model_id)
    denoisers = [
        name for name in ("transformer", "transformer_2") if name in model_index
    ]
    for name in denoisers:
        pipe_kwargs[name] = None

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

    if "transformer_2" in denoisers:
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


def _visual_token_count(pipe, height: int, width: int, num_frames: int) -> int:
    """Post-patch token count of one latent video — the sequence Ulysses shards."""
    transformer = pipe.transformer
    p_t, p_h, p_w = transformer.patch_size
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_height = height // pipe.vae_scale_factor_spatial
    latent_width = width // pipe.vae_scale_factor_spatial
    return (latent_frames // p_t) * (latent_height // p_h) * (latent_width // p_w)


@contextlib.contextmanager
def ulysses_context(pipe, args: argparse.Namespace, dtype: torch.dtype):
    """Install a :class:`UlyssesCommunicator` for the duration of the block.

    A no-op outside a distributed launch, so the same script runs on one GPU
    with ``python`` and on N with ``torchrun --nproc_per_node=N``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        yield None
        return

    import torch.distributed as dist
    from flashinfer.comm import UlyssesCommunicator

    tokens = _visual_token_count(pipe, args.height, args.width, args.num_frames)
    if tokens % world_size != 0:
        raise ValueError(
            f"visual sequence length {tokens} is not divisible by world size "
            f"{world_size}; pick a resolution/frame count that divides evenly."
        )
    heads = pipe.transformer.config.num_attention_heads
    if heads % world_size != 0:
        raise ValueError(
            f"num_attention_heads {heads} is not divisible by world size {world_size}"
        )
    head_dim = pipe.transformer.config.attention_head_dim

    comm = UlyssesCommunicator(
        dist.group.WORLD,
        # Both all-to-all directions move the same element count:
        # [B, S/world, H, D] <-> [B, S, H/world, D].
        max_elems=(tokens // world_size) * heads * head_dim,
        dtype=dtype,
        backend=args.ulysses_backend,
    )
    if dist.get_rank() == 0:
        print(
            f"Ulysses: world_size={world_size} requested={args.ulysses_backend!r} "
            f"effective={comm.backend!r}"
            + (f" (fallback: {comm.fallback_reason})" if comm.fallback_reason else "")
        )
    set_ulysses_communicator(comm)
    try:
        yield comm
    finally:
        # Clear the global first: closing is collective on the NVLink backend,
        # and no forward may reference the communicator past this point.
        set_ulysses_communicator(None)
        comm.close()


def _assert_ranks_in_sync(latents: torch.Tensor, step: int) -> None:
    """Every rank must hold identical latents; the sharded forward assumes it.

    The block stack all-gathers before the output projection, so ``noise_pred``
    is the same tensor on every rank and the scheduler keeps them in lockstep.
    If that ever stopped holding, each rank would shard a different sequence
    and the result would be quietly wrong rather than crash — hence the check.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    checksum = latents.detach().float().sum()
    gathered = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    if spread != 0.0:
        raise RuntimeError(
            f"ranks diverged at denoising step {step}: latent checksums differ "
            f"by up to {spread:g}"
        )


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
    parser.add_argument(
        "--use-vsa",
        action="store_true",
        help=(
            "Use Video Sparse Attention for self-attention (SM100, bf16, "
            "head_dim=128). Needs a VSA-finetuned checkpoint."
        ),
    )
    parser.add_argument(
        "--vsa-sparsity",
        type=float,
        default=0.9,
        help="Fraction of KV blocks VSA drops (0.9 keeps the top 10%%)",
    )
    parser.add_argument(
        "--ulysses-backend",
        default="auto",
        choices=["auto", "nvlink", "nccl"],
        help=(
            "All-to-all backend for Ulysses context parallelism. 'nvlink' is "
            "FlashInfer's NVLink-P2P kernel, 'nccl' is dist.all_to_all_single. "
            "Only used under torchrun with world size > 1."
        ),
    )
    parser.add_argument(
        "--check-rank-sync",
        action="store_true",
        help="Verify all ranks ended with identical latents (multi-GPU only).",
    )
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
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help=(
            "Override the scheduler's flow_shift. Model cards often specify a "
            "value different from the one baked into scheduler_config.json "
            "(the FastVideo VSA 14B card asks for 5.0, the config ships 3.0)."
        ),
    )
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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Run a throwaway generation with this many denoising steps before "
            "the timed one, so JIT compilation and autotuning don't land in "
            "the measurement. 0 disables."
        ),
    )
    parser.add_argument(
        "--timing-json",
        default=None,
        help="Write the timing breakdown to this JSON file (rank 0 only).",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Bracket the timed generation with cudaProfilerStart/Stop so a "
            "profiler can capture only that region. Pair with "
            "`nsys profile --capture-range=cudaProfilerApi "
            "--capture-range-end=stop`, otherwise the multi-minute model load "
            "dominates the trace."
        ),
    )
    return parser.parse_args()


def _pipeline_call_kwargs(
    args: argparse.Namespace,
    output_type: Optional[str] = None,
    pipe=None,
):
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_type": output_type or args.output_type,
    }
    # ``guidance_scale_2`` is the second-stage CFG scale of the dual-transformer
    # (Wan 2.2 MoE) pipelines, which is what ``boundary_ratio`` gates on. Wan 2.1
    # pipelines accept the kwarg in their signature but reject a non-None value.
    if pipe is None or getattr(pipe.config, "boundary_ratio", None) is not None:
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    return kwargs


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


class _StepTimer:
    """Records per-denoising-step wall time via ``callback_on_step_end``.

    Optionally doubles as the rank-sync check: comparing the latents every rank
    holds at the end of a step is the direct test that the sequence-parallel
    forward kept them identical, and it costs one scalar all-gather.
    """

    def __init__(self, check_rank_sync: bool = False) -> None:
        self.step_times: list[float] = []
        self._last = None
        self.check_rank_sync = check_rank_sync

    def __call__(self, pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        if self.check_rank_sync and "latents" in callback_kwargs:
            _assert_ranks_in_sync(callback_kwargs["latents"], step)
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._last is not None:
            self.step_times.append(now - self._last)
        self._last = now
        return callback_kwargs

    def start(self) -> None:
        torch.cuda.synchronize()
        self._last = time.perf_counter()

    @property
    def mean_ms(self) -> float:
        # Drop the first recorded interval: it still carries lazy allocations
        # and any first-call kernel selection.
        useful = self.step_times[1:] or self.step_times
        return 1000.0 * sum(useful) / len(useful) if useful else float("nan")


def _init_distributed(device: str) -> int:
    """Join the torchrun process group and bind this rank's GPU. Returns rank."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.get_rank()


def main() -> None:
    args = parse_args()
    dtype = _torch_dtype(args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    rank = _init_distributed(args.device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = args.device
    if world_size > 1 and device.startswith("cuda") and ":" not in device:
        device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"

    load_start = time.perf_counter()
    pipe = load_wan_pipeline_with_flashinfer_transformers(
        model_id=args.model_id,
        dtype=dtype,
        device=device,
        revision=args.revision,
        variant=args.variant,
        prepare_weights=args.prepare_weights,
        **_flashinfer_kwargs(args),
    )
    load_seconds = time.perf_counter() - load_start

    if args.flow_shift is not None:
        pipe.scheduler = pipe.scheduler.from_config(
            pipe.scheduler.config, flow_shift=args.flow_shift
        )
        if rank == 0:
            print(f"Scheduler flow_shift set to {pipe.scheduler.config.flow_shift}")

    with ulysses_context(pipe, args, dtype):
        if args.warmup_steps > 0:
            warmup_args = argparse.Namespace(**vars(args))
            warmup_args.num_inference_steps = args.warmup_steps
            if rank == 0:
                print(f"Warmup: {args.warmup_steps} denoising step(s)...")
            pipe(**_pipeline_call_kwargs(warmup_args, output_type="latent", pipe=pipe))

        timer = _StepTimer(check_rank_sync=args.check_rank_sync and world_size > 1)
        timer.start()
        total_start = time.perf_counter()
        if args.cuda_profiler_range:
            torch.cuda.profiler.start()
        output = pipe(
            **_pipeline_call_kwargs(args, output_type=args.output_type, pipe=pipe),
            callback_on_step_end=timer,
        )
        torch.cuda.synchronize()
        if args.cuda_profiler_range:
            torch.cuda.profiler.stop()
        total_seconds = time.perf_counter() - total_start
        frames = output.frames
        if timer.check_rank_sync and rank == 0:
            print("Rank sync check: all ranks held identical latents every step.")

    denoise_seconds = sum(timer.step_times)
    if rank == 0:
        print(
            f"Timing: total={total_seconds:.2f}s  "
            f"denoise={denoise_seconds:.2f}s over {len(timer.step_times)} steps  "
            f"per_step={timer.mean_ms:.1f}ms  "
            f"(model load {load_seconds:.1f}s, not counted)"
        )
        if args.timing_json is not None:
            import json

            Path(args.timing_json).write_text(
                json.dumps(
                    {
                        "model_id": args.model_id,
                        "world_size": world_size,
                        "height": args.height,
                        "width": args.width,
                        "num_frames": args.num_frames,
                        "num_inference_steps": args.num_inference_steps,
                        "gemm_backend": args.gemm_backend,
                        "attention_backend": args.attention_backend,
                        "online_act_quant": not args.offline_act_quant,
                        "use_vsa": args.use_vsa,
                        "vsa_sparsity": args.vsa_sparsity if args.use_vsa else None,
                        "ulysses_backend": args.ulysses_backend
                        if world_size > 1
                        else None,
                        "total_seconds": total_seconds,
                        "denoise_seconds": denoise_seconds,
                        "per_step_ms": timer.mean_ms,
                        "step_times_s": timer.step_times,
                    },
                    indent=2,
                )
            )
        _save_or_print_output(args, frames)

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
