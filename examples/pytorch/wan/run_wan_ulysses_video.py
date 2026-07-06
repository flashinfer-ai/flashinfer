"""Generate a full Wan T2V video with Ulysses sequence parallelism on N GPUs.

Every rank loads the full pipeline (text encoder / VAE / scheduler replicated),
shards the transformer token sequence 1/N per rank, and runs self-attention
through :class:`flashinfer.comm.UlyssesCommunicator` with ``backend="auto"``:
the fused-transpose NVLink-P2P kernel on a verified full-NVLink topology, an
automatic NCCL fallback elsewhere (rank 0 prints the effective backend and
the fallback reason). All ranks hold the identical full output after each
transformer call, so the scheduler/VAE stay in lockstep; rank 0 exports the
video.

Example (8 GPUs, 480x832, 81 frames):
    python run_wan_ulysses_video.py --world-size 8 --output wan_ulysses.mp4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

_WAN_DIR = Path(__file__).resolve().parent
for _p in (str(_WAN_DIR), str(_WAN_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


WAN_HEADS, WAN_DIM_HEAD = 40, 128  # Wan A14B attention geometry


def validate_ulysses_video_config(
    world_size: int, seq_global: int, device_count: int, heads: int = WAN_HEADS
) -> None:
    """Pure preflight: reject impossible run configs before any communicator
    construction or (slow) model load. Notably, Wan's 40-head attention needs
    ``heads % world_size == 0`` — the fused kernel also supports W=6, and the
    default token count divides by 6, so without this check ``--world-size 6``
    would construct the communicator, load the model, and only fail at the
    first scatter."""
    if world_size <= 0:
        raise ValueError(f"world size must be positive, got {world_size}")
    if device_count < world_size:
        raise ValueError(
            f"world size {world_size} exceeds visible GPU count {device_count}"
        )
    if seq_global <= 0:
        raise ValueError(
            f"token count must be positive, got {seq_global}; "
            "check --height/--width/--num-frames"
        )
    if seq_global % world_size != 0:
        raise ValueError(
            f"token count {seq_global} not divisible by world size {world_size}; "
            "adjust --height/--width/--num-frames"
        )
    if heads % world_size != 0:
        usable = [w for w in (2, 4, 6, 8) if heads % w == 0]
        raise ValueError(
            f"Wan attention has {heads} heads; the Ulysses world size must "
            f"divide it (of the fused-kernel sizes 2/4/6/8, this model can use "
            f"{usable} — world size {world_size} cannot)"
        )


def _worker(world_size: int, rank: int, port: int, args) -> None:
    import torch
    import torch.distributed as dist

    # Defense-in-depth preflight (main() already validated before spawning):
    # must run BEFORE set_device / init_process_group so an impossible config
    # cannot strand peers in a process-group init that times out in hours.
    lat_t = (args.num_frames - 1) // 4 + 1
    seq_global = lat_t * (args.height // 16) * (args.width // 16)
    validate_ulysses_video_config(world_size, seq_global, torch.cuda.device_count())

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # heavy imports (transformer pulls the full FlashInfer stack) only AFTER
    # the preflight and the device binding, so nothing initializes on GPU 0;
    # the pipeline loader below reuses the same cached module identity
    from transformer_wan_flashinfer import set_ulysses_communicator

    from flashinfer.comm import UlyssesCommunicator

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=120),
    )
    group = dist.group.WORLD
    # Largest single a2a operand: B * S_local * heads * dim_head elements
    # (input and output numel are equal), with B up to 2 for CFG batching.
    seq_local = seq_global // world_size
    max_elems = 2 * seq_local * WAN_HEADS * WAN_DIM_HEAD

    # IPC buffers involve collectives: create while all ranks are in lockstep,
    # before the (slow, unsynchronized) model load.
    comm = UlyssesCommunicator(
        group,
        max_elems=max_elems,
        dtype=torch.bfloat16,
        backend="auto",
        device=device,
    )
    if rank == 0:
        print(f"[rank0] ulysses effective backend: {comm.backend}", flush=True)
        if comm.fallback_reason is not None:
            print(f"[rank0] !! fallback: {comm.fallback_reason}", flush=True)

    from pipeline_wan_flashinfer import load_wan_pipeline_with_flashinfer_transformers

    t0 = time.time()
    pipe = load_wan_pipeline_with_flashinfer_transformers(
        model_id=args.model_id,
        dtype=torch.bfloat16,
        device=str(device),
        attention_backend=args.attention_backend,
        gemm_backend="torch",
    )
    if rank == 0:
        print(f"[rank0] pipeline loaded in {time.time() - t0:.0f}s", flush=True)
    dist.barrier(group=group)

    try:
        set_ulysses_communicator(comm)
        generator = torch.Generator(device=str(device)).manual_seed(args.seed)
        t0 = time.time()
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            output_type=args.output_type,
        )
        gen_s = time.time() - t0
    finally:
        # cleared BEFORE the collective close: an exception above must not
        # leave the transformer pointing at a dead communicator
        set_ulysses_communicator(None)

    # Collective teardown BEFORE any rank-0-only export: an I/O failure below
    # must fail rank 0 alone, not strand the other ranks in the barrier or
    # leave IPC mappings open.
    dist.barrier(group=group)
    comm.close()
    dist.destroy_process_group()

    if rank == 0:
        if args.output_type == "latent":
            lat = output.frames.detach().cpu()
            torch.save(lat, args.output)
            print(
                f"[rank0] latent {tuple(lat.shape)} mean={lat.float().mean():.6f} "
                f"std={lat.float().std():.6f} in {gen_s:.0f}s (ws={world_size}) "
                f"-> {args.output}",
                flush=True,
            )
        else:
            import numpy as np

            frames = output.frames[0]
            # Safety net first: raw frames survive even if video export fails.
            np.save(str(Path(args.output).with_suffix(".npy")), np.asarray(frames))
            from diffusers.utils import export_to_video

            export_to_video(frames, args.output, fps=16)
            print(
                f"[rank0] generated {len(frames)} frames in {gen_s:.0f}s "
                f"({args.num_inference_steps} steps, ws={world_size}) -> {args.output}",
                flush=True,
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--model-id", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    p.add_argument(
        "--prompt",
        default=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage."
        ),
    )
    p.add_argument(
        "--negative-prompt",
        default=(
            "gaudy colors, overexposed, static, blurry details, subtitles, style, "
            "artwork, painting, image, still, washed out, worst quality, low "
            "quality, JPEG artifacts, ugly, mutilated, extra fingers, poorly "
            "drawn hands, poorly drawn face, deformed, disfigured, malformed "
            "limbs, fused fingers, still image, cluttered background, three legs, "
            "crowded background, walking backwards"
        ),
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attention-backend", default="torch")
    p.add_argument("--output-type", default="np", choices=["np", "latent"])
    p.add_argument("--output", default="wan_ulysses.mp4")
    args = p.parse_args()

    # CLI preflight: reject impossible configs before creating ANY worker —
    # range(0) would otherwise spawn nothing and exit 0 silently, and an
    # oversized world size would strand ranks in process-group init.
    import torch

    lat_t = (args.num_frames - 1) // 4 + 1
    seq_global = lat_t * (args.height // 16) * (args.width // 16)
    validate_ulysses_video_config(
        args.world_size, seq_global, torch.cuda.device_count()
    )

    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = [
        mp.Process(target=_worker, args=(args.world_size, r, port, args))
        for r in range(args.world_size)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
        assert pr.exitcode == 0, f"worker failed with exit code {pr.exitcode}"


if __name__ == "__main__":
    main()
