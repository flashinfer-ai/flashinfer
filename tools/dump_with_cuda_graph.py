#!/usr/bin/env python3
"""
Run a Python program with FLASHINFER_LOGLEVEL=10 dumps that work under
``torch.cuda.graph(...)`` capture, without modifying the program.

Why this exists
---------------
Level-10 dumps stage tensors into pinned host buffers via captured D2H
``copy_()`` ops; the actual ``inputs.pt`` / ``outputs.pt`` writes are
deferred until ``flashinfer.api_logging.flush_graph_dumps()`` is called
after each ``g.replay()``.

When the host program is something you can't modify (e.g. ``sglang``),
this wrapper monkey-patches ``torch.cuda.CUDAGraph.replay`` so the flush
fires automatically after every replay. The patched program runs unaware
that anything is different.

Usage
-----
::

    python tools/dump_with_cuda_graph.py \\
        --dump-dir /tmp/fi_dumps \\
        --include '*decode*' \\
        --max-count 10 \\
        -- \\
        python -m sglang.launch_server --model meta-llama/Llama-3-8B ...

Or as a runnable module (after ``pip install -e .``)::

    python -m tools.dump_with_cuda_graph -- python -m sglang.launch_server ...

Anything before ``--`` configures the wrapper (env vars). Anything after
``--`` is the command to run.

Caveats
-------
* sglang's CUDA-graph capture warms up eagerly first, so the pinned
  buffer cache is primed by the time capture runs. If you see
  ``RuntimeError: Pinned host memory cannot be allocated during capture``
  the capture path saw a tensor it did not see during warmup; force a
  warmup or narrow the dump scope with ``--include``.
* Every replay overwrites the same dump files. Use ``--include`` and
  ``--max-count`` aggressively, capture a few replays, then kill the
  server and inspect.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import List, Optional


def _split_argv(argv: List[str]) -> tuple[List[str], List[str]]:
    """Split argv at the first ``--`` separator.

    Everything before ``--`` configures the wrapper; everything after is
    the command to exec. If ``--`` is absent, treat all args as wrapper
    config and read the command from environment / fail.
    """
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _build_wrapper_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dump_with_cuda_graph",
        description=(
            "Run a Python program with FLASHINFER_LOGLEVEL=10 + auto-flush "
            "after every torch.cuda.CUDAGraph.replay()."
        ),
        epilog="Anything after `--` is the command to exec.",
    )
    p.add_argument(
        "--dump-dir",
        default=os.environ.get("FLASHINFER_DUMP_DIR", "flashinfer_dumps"),
        help=(
            "Directory for tensor dumps. Sets FLASHINFER_DUMP_DIR. "
            "Default: env or 'flashinfer_dumps'."
        ),
    )
    p.add_argument(
        "--include",
        default=os.environ.get("FLASHINFER_DUMP_INCLUDE", ""),
        help=(
            "Comma-separated fnmatch patterns of API names to dump "
            "(e.g. '*decode*,BatchDecodeWrapper.run'). Strongly recommended "
            "to keep the dump volume manageable."
        ),
    )
    p.add_argument(
        "--exclude",
        default=os.environ.get("FLASHINFER_DUMP_EXCLUDE", ""),
        help="Comma-separated fnmatch patterns of API names to skip.",
    )
    p.add_argument(
        "--max-count",
        type=int,
        default=int(os.environ.get("FLASHINFER_DUMP_MAX_COUNT", "10")),
        help="Cap on number of distinct (call) dumps. Default: 10.",
    )
    p.add_argument(
        "--max-size-gb",
        type=float,
        default=float(os.environ.get("FLASHINFER_DUMP_MAX_SIZE_GB", "20")),
        help="Cap on total dump size in GB. Default: 20.",
    )
    p.add_argument(
        "--safetensors",
        action="store_true",
        default=os.environ.get("FLASHINFER_DUMP_SAFETENSORS", "0") == "1",
        help="Use safetensors format instead of torch.save (loses strides).",
    )
    p.add_argument(
        "--logdest",
        default=os.environ.get("FLASHINFER_LOGDEST", "stderr"),
        help="FLASHINFER_LOGDEST: stdout | stderr | <file path>.",
    )
    return p


def _apply_env(args: argparse.Namespace) -> None:
    os.environ["FLASHINFER_LOGLEVEL"] = "10"
    os.environ["FLASHINFER_DUMP_DIR"] = args.dump_dir
    os.environ["FLASHINFER_DUMP_INCLUDE"] = args.include
    os.environ["FLASHINFER_DUMP_EXCLUDE"] = args.exclude
    os.environ["FLASHINFER_DUMP_MAX_COUNT"] = str(args.max_count)
    os.environ["FLASHINFER_DUMP_MAX_SIZE_GB"] = str(args.max_size_gb)
    os.environ["FLASHINFER_DUMP_SAFETENSORS"] = "1" if args.safetensors else "0"
    os.environ["FLASHINFER_LOGDEST"] = args.logdest


def install_replay_autoflush() -> Optional[object]:
    """Monkey-patch ``torch.cuda.CUDAGraph.replay`` to call
    :func:`flashinfer.api_logging.flush_graph_dumps` after every replay.

    Returns the original ``replay`` callable (so callers can reverse the
    patch in tests), or ``None`` if torch/flashinfer aren't importable.
    """
    try:
        import torch
        from flashinfer.api_logging import flush_graph_dumps
    except Exception as exc:  # pragma: no cover - sanity guard
        print(
            f"[dump_with_cuda_graph] failed to import torch / flashinfer: {exc}",
            file=sys.stderr,
        )
        return None

    if getattr(torch.cuda.CUDAGraph.replay, "_flashinfer_autoflush", False):
        # Already patched (idempotent).
        return torch.cuda.CUDAGraph.replay

    original = torch.cuda.CUDAGraph.replay

    def replay_with_flush(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # flush_graph_dumps() syncs first, so the buffers are coherent
        # before we serialize them. No-op if the registry is empty.
        try:
            flush_graph_dumps()
        except Exception as exc:
            # Never let a dump error mask the inference call's result.
            print(
                f"[dump_with_cuda_graph] flush_graph_dumps failed: {exc}",
                file=sys.stderr,
            )
        return result

    replay_with_flush._flashinfer_autoflush = True  # type: ignore[attr-defined]
    torch.cuda.CUDAGraph.replay = replay_with_flush  # type: ignore[assignment]
    return original


def _exec_target(target_argv: List[str]) -> int:
    """Replace this process with the user's command, preserving env vars."""
    if not target_argv:
        print(
            "[dump_with_cuda_graph] no command after `--`; nothing to run.",
            file=sys.stderr,
        )
        return 2

    # Special-case `python ...` so we run inside this interpreter (which
    # already has the monkey-patch active). For any other executable, fall
    # back to os.execvp; that re-execs into a fresh interpreter, so the
    # monkey-patch must be re-installed there. Document this.
    head = target_argv[0]
    if head in ("python", "python3", sys.executable):
        # Run as a Python script / module in-process so our patch survives.
        rest = target_argv[1:]
        if rest and rest[0] == "-m":
            module = rest[1]
            sys.argv = [module] + rest[2:]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
            return 0
        if rest:
            sys.argv = rest
            runpy.run_path(rest[0], run_name="__main__")
            return 0
        # `python` with no args -> drop to interactive: just exec.
    # External binary: re-exec. The child process won't have our patch
    # unless it imports this module again — print a warning.
    print(
        "[dump_with_cuda_graph] note: command is not an in-process Python "
        "invocation, so the CUDAGraph.replay patch will not propagate. "
        "Prefer `python -m sglang...` over a launcher binary.",
        file=sys.stderr,
    )
    os.execvp(head, target_argv)
    return 0  # unreachable


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    wrapper_args, target_argv = _split_argv(argv)
    args = _build_wrapper_argparser().parse_args(wrapper_args)

    _apply_env(args)
    install_replay_autoflush()

    Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"[dump_with_cuda_graph] FLASHINFER_LOGLEVEL=10, dump_dir={args.dump_dir}, "
        f"include='{args.include}', exclude='{args.exclude}', "
        f"max_count={args.max_count}",
        file=sys.stderr,
    )

    return _exec_target(target_argv)


if __name__ == "__main__":
    sys.exit(main())
