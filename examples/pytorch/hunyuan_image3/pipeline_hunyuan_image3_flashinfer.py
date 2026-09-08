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

Environment variables (same precedence as the wan example; command-line
flags override them):

  FLASHINFER_GEMM_BACKEND       torch | bf16 | fp8 | ...
  FLASHINFER_ATTENTION_BACKEND  auto | single | cudnn | trtllm | sdpa
  FLASHINFER_MOE_BACKEND        cutlass | cutlass_fp8 |
                                cutlass_fp8_blockscale | cutlass_w4a16 |
                                trtllm | torch | eager
  FLASHINFER_MOE_IMPL           flashinfer | eager (deprecated alias of
                                FLASHINFER_MOE_BACKEND)
  FLASHINFER_ONLINE_ACT_QUANT   1/0

Note: the checkpoint's remote code requires transformers 4.x (upstream
tests 4.56) and a local model directory whose name contains no dots, e.g.
``hf download tencent/HunyuanImage-3.0-Instruct --local-dir
./HunyuanImage-3-Instruct``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modeling_hunyuan_image3_flashinfer import (  # noqa: E402
    _MOE_BACKENDS,
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
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="HuggingFace repo id or local path.",
    )
    p.add_argument(
        "--modality",
        default="text2img",
        choices=list(_MODALITY_TASK_MAP),
    )
    p.add_argument("--prompts", nargs="+", default=None)
    p.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Input image path(s) for img2img/img2text, comma-separated for multi-image (max 3).",
    )
    p.add_argument("--output", type=str, default=".")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--bot-task",
        type=str,
        default=None,
        choices=["none", "think", "recaption", "think_recaption", "vanilla"],
    )
    p.add_argument(
        "--sys-type",
        type=str,
        default=None,
        help="Override system prompt type (e.g. en_unified, en_vanilla).",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--device-map",
        default=None,
        help="HF device_map for from_pretrained (e.g. 'auto' to shard the "
        "168GB model across multiple GPUs). Default: load on CPU then "
        ".cuda() onto a single GPU.",
    )
    p.add_argument(
        "--max-memory-per-gpu",
        default="140GiB",
        help="Per-GPU cap passed as from_pretrained(max_memory=...) when "
        "--device-map is set. 'auto' greedily fills GPU 0 to capacity, "
        "leaving no room for FlashInfer weight-prep caches / activations; "
        "a cap (default 140GiB on a 178GiB B200) forces headroom on every "
        "GPU. Set to 'none' to disable.",
    )

    # FlashInfer config flags.
    p.add_argument(
        "--gemm-backend",
        default=os.getenv("FLASHINFER_GEMM_BACKEND", "torch"),
        help="GEMM backend for swapped linears.",
    )
    p.add_argument(
        "--attention-backend",
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        choices=["auto", "single", "cudnn", "trtllm", "torch", "sdpa"],
    )
    p.add_argument(
        "--moe-backend",
        default=os.getenv("FLASHINFER_MOE_BACKEND", "cutlass"),
        choices=list(_MOE_BACKENDS),
        help="Fused-MoE backend for the routed experts (see "
        "flashinfer_modules.MoEBackend). 'eager' keeps the upstream "
        "per-expert loop.",
    )
    p.add_argument(
        "--moe-impl",
        default=None,
        choices=["flashinfer", "eager"],
        help="Deprecated alias for --moe-backend "
        "(flashinfer -> cutlass, eager -> eager).",
    )
    p.add_argument(
        "--offline-act-quant",
        action="store_true",
        help="Use fixed default activation quantization scale (FP8/FP4 backends).",
    )
    p.add_argument(
        "--skip-flashinfer",
        action="store_true",
        help="Load the model but don't swap to FlashInfer (for A/B comparison).",
    )

    # torch.compile / CUDA-graph acceleration of the per-step backbone forward.
    p.add_argument(
        "--compile-mode",
        default="none",
        choices=[
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help=(
            "torch.compile mode for the HunyuanImage3Model backbone "
            "(model.model). 'none' disables compilation. 'reduce-overhead' and "
            "'max-autotune' enable CUDA graphs; 'default' / "
            "'max-autotune-no-cudagraphs' are compile-only (no CUDA graph)."
        ),
    )
    p.add_argument(
        "--taylor-cache",
        action="store_true",
        help=(
            "Enable the upstream Taylor activation cache across timesteps. Off "
            "by default; leave off for torch.compile / CUDA-graph runs (it makes "
            "the per-step forward stateful and breaks static-shape capture)."
        ),
    )
    # Benchmarking: warmup + timed repeats so compile/graph cost is amortized.
    p.add_argument(
        "--bench",
        action="store_true",
        help="Report end-to-end and per-step latency (GPU-synced) for each run.",
    )
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Throwaway generations before timing (warms torch.compile / JIT / "
        "CUDA-graph capture). Recommended >=1 when --compile-mode != none.",
    )
    p.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Number of timed generations to average over when --bench is set.",
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


def _print_config(
    args: argparse.Namespace,
    opts,
    prompts: list[str],
    task: str,
    bot_task: Optional[str],
) -> None:
    if args.verbose < 1:
        return
    print("=" * 60)
    print("HunyuanImage-3 FlashInfer driver configuration:")
    print(f"  Model: {args.model}")
    print(f"  Modality: {args.modality} (task={task}, bot_task={bot_task})")
    print(f"  dtype: {args.dtype}")
    print(f"  FlashInfer options: {opts}")
    if args.modality in ("text2img", "img2img"):
        print(
            f"  Steps: {args.steps}, guidance: {args.guidance_scale}, seed: {args.seed}"
        )
        print(f"  Output size: {args.width}x{args.height}")
    if args.image_path:
        print(f"  Input image: {args.image_path}")
    print(f"  Prompts: {prompts}")
    print("=" * 60)


def _resolve_bot_task(
    args: argparse.Namespace, default_bot_task: Optional[str]
) -> Optional[str]:
    if args.bot_task is None:
        return default_bot_task
    if args.bot_task == "none":
        return None
    return args.bot_task


def _check_transformers_version() -> None:
    """Fail fast on transformers >= 5.

    The checkpoint's remote code targets the transformers 4.x generation and
    cache APIs (e.g. it calls ``StaticLayer.lazy_initialization(key_states)``
    with one argument and relies on 4.x ``generate`` internals). On
    transformers 5.x it crashes deep inside ``generate``; the upstream model
    card tests with transformers 4.56.
    """
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise RuntimeError(
            f"HunyuanImage-3's remote code requires transformers 4.x "
            f"(upstream tests 4.56), but {transformers.__version__} is "
            f"installed. Run: pip install 'transformers==4.56.2'"
        )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    _check_transformers_version()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run HunyuanImage-3.")

    dtype = _torch_dtype(args.dtype)
    task, default_bot_task = _MODALITY_TASK_MAP[args.modality]
    bot_task = _resolve_bot_task(args, default_bot_task)

    prompts, input_images = _load_inputs(args)

    # 1. Load the HF model with trust_remote_code. This pulls
    # modeling_hunyuan_image_3.py, hunyuan_image_3_pipeline.py, the VAE, the
    # SigLIP-2 ViT, and the tokenization into the local HF cache and wires
    # them into a HunyuanImage3ForCausalMM instance.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} (trust_remote_code=True) ...")
    from_kwargs = dict(trust_remote_code=True, torch_dtype=dtype)
    if args.skip_flashinfer:
        # Pure-HF baseline: the model card loads with an explicit moe_impl /
        # attn_implementation. Use the upstream eager MoE + SDPA so the
        # baseline depends on no FlashInfer kernels at all.
        from_kwargs["moe_impl"] = "eager"
        from_kwargs["attn_implementation"] = "sdpa"
    if args.device_map is not None:
        from_kwargs["device_map"] = args.device_map
        mm = args.max_memory_per_gpu
        if mm and mm.lower() != "none":
            n_gpus = torch.cuda.device_count()
            from_kwargs["max_memory"] = {i: mm for i in range(n_gpus)}
            print(
                f"[load] device_map={args.device_map} "
                f"max_memory={{0..{n_gpus - 1}: {mm}}}"
            )
    model = AutoModelForCausalLM.from_pretrained(args.model, **from_kwargs)
    model = model.eval()
    if args.device_map is None:
        model = model.cuda()

    # The upstream model owns a ``_tokenizer`` slot it uses inside
    # ``prepare_model_inputs`` / ``generate`` / ``generate_image``. The
    # canonical setup (per the model card) is ``model.load_tokenizer(path)``,
    # which wires the HunyuanImage-3 tokenizer wrapper (special image/ratio
    # tokens etc.). Fall back to attaching an AutoTokenizer instance if that
    # method is unavailable.
    if hasattr(model, "load_tokenizer"):
        # ``load_tokenizer`` reads ``self.config.model_version`` and forwards it
        # to the tokenizer (which ignores it). This checkpoint's config.json
        # omits the field, so provide a default to avoid an AttributeError.
        if getattr(model.config, "model_version", None) is None:
            model.config.model_version = "3.0"
        model.load_tokenizer(args.model)
    else:
        model._tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )

    # Optionally enable VAE tiling to keep peak memory down for large images.
    if (
        args.vae_use_tiling
        and hasattr(model, "vae")
        and hasattr(model.vae, "enable_tiling")
    ):
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
            moe_backend=args.moe_backend,
            moe_impl=args.moe_impl,
            online_act_quant=not args.offline_act_quant,
            prepare_weights=True,
        )

    # 2b. Optionally wrap the backbone with torch.compile (+ CUDA graphs).
    _maybe_compile_backbone(model, args)

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


def _maybe_compile_backbone(model, args: argparse.Namespace) -> None:
    """Wrap ``model.model`` (HunyuanImage3Model backbone) with torch.compile.

    The per-timestep image-generation forward in
    ``HunyuanImage3Text2ImagePipeline`` calls ``self.model(**inputs)``, which
    runs ``HunyuanImage3ForCausalMM.forward`` -> ``self.model(...)`` (the
    transformer backbone). Reassigning ``model.model`` to a compiled module
    transparently routes that hot path through torch.compile.

    ``reduce-overhead`` / ``max-autotune`` use CUDA graphs under the hood;
    ``default`` / ``max-autotune-no-cudagraphs`` are compile-only. The image
    branch uses a static sequence length and a fixed attention mask per
    generation, which is what makes CUDA-graph capture viable here. The Taylor
    activation cache is forced off unless ``--taylor-cache`` is given, because
    it makes the per-step forward stateful and defeats graph capture.
    """
    # Honor the Taylor-cache toggle regardless of compilation.
    if hasattr(model, "use_taylor_cache"):
        model.use_taylor_cache = bool(args.taylor_cache)

    if args.compile_mode == "none":
        return
    if args.taylor_cache:
        print(
            "[warn] --taylor-cache with --compile-mode may trigger frequent "
            "recompiles / graph breaks."
        )

    backbone = getattr(model, "model", None)
    if backbone is None:
        print("[warn] model has no .model backbone; skipping torch.compile.")
        return

    # A roomy dynamo cache: the first vs. later denoising step and the
    # text-decode path specialize to different guards.
    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)

    mode = None if args.compile_mode == "default" else args.compile_mode
    uses_cudagraph = args.compile_mode in ("reduce-overhead", "max-autotune")
    print(
        f"[compile] torch.compile(model.model, mode={args.compile_mode!r}) "
        f"(CUDA graphs: {uses_cudagraph})"
    )
    compiled = torch.compile(backbone, mode=mode)

    if uses_cudagraph:
        # CUDA-graph modes reuse a static output buffer, but the denoising loop
        # reads the previous step's backbone output while preparing the next
        # call -> "accessing tensor output of CUDAGraphs that has been
        # overwritten". Marking a new cudagraph step before each invocation
        # tells the runtime the prior output is consumed.
        model.model = _CudagraphStepModule(compiled)
    else:
        model.model = compiled


class _CudagraphStepModule(torch.nn.Module):
    """Calls ``torch.compiler.cudagraph_mark_step_begin()`` before each forward,
    delegating attribute access to the wrapped (compiled) backbone so the rest
    of the model (which reads ``model.model.<attr>``) is unaffected."""

    def __init__(self, compiled):
        super().__init__()
        self._compiled = compiled

    def forward(self, *args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        out = self._compiled(*args, **kwargs)
        # The denoising loop reads the backbone output while the next step's
        # graph replay overwrites the static buffer. Clone the read-downstream
        # hidden state (NOT the static KV cache) so callers hold a private copy.
        try:
            if getattr(out, "last_hidden_state", None) is not None:
                out.last_hidden_state = out.last_hidden_state.clone()
            elif torch.is_tensor(out):
                out = out.clone()
            elif isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                cloned = list(out)
                cloned[0] = cloned[0].clone()
                out = type(out)(cloned)
        except (AttributeError, TypeError, NotImplementedError):
            # Output type we don't know how to clone — hand it back as-is.
            # Real runtime failures (RuntimeError, CUDA OOM) must propagate.
            pass
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._compiled, name)


def _bench_generate(
    gen_fn, args: argparse.Namespace, label: str, steps: Optional[int] = None
):
    """Run ``gen_fn`` with warmup + timed repeats; print GPU-synced latency.

    ``gen_fn`` must be a zero-arg callable that performs one full generation
    and returns its result (e.g. a list of PIL images). Returns the result of
    the final timed run. ``steps`` enables the per-step latency breakdown for
    diffusion modalities; autoregressive text generation passes ``None`` and
    reports end-to-end time only (the generated token count varies per run).
    """
    import time

    for w in range(args.warmup_runs):
        print(f"[bench:{label}] warmup {w + 1}/{args.warmup_runs} ...")
        gen_fn()
        torch.cuda.synchronize()

    if not args.bench:
        return gen_fn()

    times = []
    result = None
    for r in range(max(1, args.timed_runs)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = gen_fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        per_step = (
            ""
            if steps is None
            else (f", {dt / max(1, steps) * 1e3:.1f} ms/step ({steps} steps)")
        )
        print(
            f"[bench:{label}] run {r + 1}/{args.timed_runs}: {dt:.3f} s total{per_step}"
        )

    best = min(times)
    mean = sum(times) / len(times)
    peak = torch.cuda.max_memory_allocated() / 1e9
    per_step_best = (
        "" if steps is None else (f"per_step_best={best / max(1, steps) * 1e3:.1f}ms ")
    )
    print(
        f"[bench:{label}] SUMMARY best={best:.3f}s mean={mean:.3f}s "
        f"{per_step_best}"
        f"peak_mem={peak:.1f}GB"
    )
    return result


# ----------------------------------------------------------------------------
# Per-modality drivers
# ----------------------------------------------------------------------------


def _run_text2img(model, prompts: list[str], args: argparse.Namespace) -> None:
    """Text->image via the model's canonical ``generate_image`` entrypoint.

    ``HunyuanImage3ForCausalMM.generate_image`` is the documented API (model
    card): it optionally runs a chain-of-thought / recaption *text* stage
    (``bot_task`` in {think, recaption, think_recaption}) and then the diffusion
    *image* stage (``mode='gen_image'`` -> ``HunyuanImage3Text2ImagePipeline``).
    It returns ``(cot_text, samples)`` with ``samples`` a list of PIL images.

    For perf isolation of the FlashInfer-accelerated denoising backbone, pass a
    ``--bot-task`` outside the think/recaption set (e.g. ``vanilla``/``none``)
    and an explicit ``--height/--width`` so the text stage and the auto
    aspect-ratio decode are skipped and only the static-shape image denoising
    runs (which is also what torch.compile / CUDA graphs target).
    """
    # bot_task: 'none' (driver) -> skip the chain-of-thought / recaption *text*
    # stage and go straight to image denoising. We pass 'vanilla' rather than
    # None because generate_image() replaces None with the generation_config
    # default (think_recaption), which adds a long (~300s) autoregressive text
    # stage that swamps the denoising latency we want to measure. 'vanilla' is
    # not in {think,recaption,think_recaption}, so the CoT block is skipped; an
    # explicit image_size also skips the auto aspect-ratio decode.
    bt = args.bot_task
    bot_task = "vanilla" if bt in (None, "none") else bt
    image_size = [args.height, args.width]

    # The denoising step count / guidance are read from the generation config
    # (gen_config.diff_infer_steps / diff_guidance_scale) inside generate(),
    # NOT from generate_image kwargs, so set them on the config directly.
    if hasattr(model, "generation_config"):
        model.generation_config.diff_infer_steps = args.steps
        model.generation_config.diff_guidance_scale = args.guidance_scale

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Generating image for: {prompt!r}")

        def _gen():
            cot_text, samples = model.generate_image(
                prompt=prompt,
                seed=args.seed,
                image_size=image_size,
                bot_task=bot_task,
                system_prompt=args.sys_type,
                verbose=args.verbose,
            )
            return samples

        images = _bench_generate(_gen, args, label=f"text2img[{idx}]", steps=args.steps)
        for j, img in enumerate(images):
            save_path = Path(args.output) / f"output_{idx}_{j}.png"
            img.save(save_path)
            print(f"  saved: {save_path}")


def _run_img2img(
    model, prompts: list[str], input_images: list, args: argparse.Namespace
) -> None:
    """Image editing path: same pipeline, but with ``image`` arg."""
    pipeline = model.pipeline
    img_arg = input_images[0] if len(input_images) == 1 else input_images
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Editing image for: {prompt!r}")

        def _gen():
            return pipeline(
                prompt=prompt,
                image=img_arg,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                verbose=args.verbose,
            )

        images = _bench_generate(_gen, args, label=f"img2img[{idx}]", steps=args.steps)
        for j, img in enumerate(images):
            save_path = Path(args.output) / f"output_{idx}_{j}.png"
            img.save(save_path)
            print(f"  saved: {save_path}")


def _run_img2text(
    model,
    prompts: list[str],
    input_images: list,
    args: argparse.Namespace,
    bot_task: Optional[str],
) -> None:
    """Image-conditioned text generation via the model's ``.generate`` path."""
    img_arg = input_images[0] if len(input_images) == 1 else input_images
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Captioning image with: {prompt!r}")

        def _gen():
            inputs = model.prepare_model_inputs(
                prompt=prompt,
                image=img_arg,
                mode="gen_text",
                system_prompt=args.sys_type,
                bot_task=bot_task or "auto",
                max_new_tokens=args.max_new_tokens,
            )
            return model.generate(
                **inputs,
                verbose=args.verbose,
                decode_text=True,
            )

        out = _bench_generate(_gen, args, label=f"img2text[{idx}]")
        text = (
            out
            if isinstance(out, str)
            else (out[0] if isinstance(out, list) else str(out))
        )
        print(f"  output: {text}")


def _run_text2text(
    model, prompts: list[str], args: argparse.Namespace, bot_task: Optional[str]
) -> None:
    """Pure text-to-text generation via the model's ``.generate`` path."""
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] Generating text for: {prompt!r}")

        def _gen():
            inputs = model.prepare_model_inputs(
                prompt=prompt,
                mode="gen_text",
                system_prompt=args.sys_type,
                bot_task=bot_task or "auto",
                max_new_tokens=args.max_new_tokens,
            )
            return model.generate(
                **inputs,
                verbose=args.verbose,
                decode_text=True,
            )

        out = _bench_generate(_gen, args, label=f"text2text[{idx}]")
        text = (
            out
            if isinstance(out, str)
            else (out[0] if isinstance(out, list) else str(out))
        )
        print(f"  output: {text}")


if __name__ == "__main__":
    main()
