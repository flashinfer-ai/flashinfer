# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120 sparse-MLA ``batch_invariant``: issue #5553 reproduction and latency.

Both subcommands call the public ``trtllm_batch_decode_with_kv_cache_mla`` with
``backend="sparse"`` over packed FP8 KV in the DSv3.2 (``pow2_fp32``) or
GLM-NSA (``arbitrary_fp32``) 656-byte layout, 576-wide queries, top-k 2048 and
64-token pages. The KV pool defaults to at least 8x the device L2 so random
top-k indices read from DRAM; the header reports the pool size, the
calibration store and the timing backend.

``repro``
    Runs one request of ``q_len`` tokens alone and again together with
    ``others`` requests of the same length (the request first), and reports,
    for that request, how many (token, head) rows differ in the output and in
    the LSE, the max |diff| of the output, whether both are bit-identical, and
    the kernel variant and ``chunks_per_block`` each of the two calls used.
    Defaults cover the issue's steps (q_len 1 and 4 with one other request,
    i.e. T=1->2 and 4->8) plus the 64-token decode/prefill boundary and
    batch-invariant multi-launch calls (q_len 64, and 64 other requests).
    ``batch_invariant=False`` is expected to differ for most shapes (the #5553
    bug), but it need not on every device: the table marks each case
    REPRODUCED or NOT REPRODUCED. ``batch_invariant=True`` must be
    bit-identical; the command exits non-zero otherwise, and also when
    ``--require-default-mismatch`` is set and no case reproduced the bug. The
    issue was observed without a calibration store; ``--no-calibration``
    hides any local store so the default route matches that setup.

``bench``
    Median latency of ``batch_invariant=False`` vs ``True`` over total token
    counts ``T`` laid out as ``T // q_len`` requests of ``q_len`` tokens
    (``q_len=1`` is the one-token-per-request layout vLLM uses). Defaults
    include partial final launches around the 64-token cutoff. Each row also
    reports the variant and ``chunks_per_block`` each route resolved and the
    number of batch-invariant 64-row chunks (each launches a decode and a
    merge kernel). With cupti-python (>= 13), calls are
    replayed from a CUDA graph, the L2 is flushed before every call and CUPTI
    reports kernel time, not Python dispatch overhead. Without it, calls run
    eagerly with CUDA events and a per-call L2 flush, so short calls also
    include host launch gaps. The CUDA-graph event fallback is not used: it
    keeps the L2 cold only by rotating whole input tensors, while a sparse call
    reads a small, fixed subset of the KV pool (2048 rows per token) that can
    stay cached across replays.

Examples::

    python benchmarks/bench_sparse_mla_sm120_batch_invariant.py repro
    python benchmarks/bench_sparse_mla_sm120_batch_invariant.py bench
    python benchmarks/bench_sparse_mla_sm120_batch_invariant.py bench \\
        --formats glm_nsa --num-heads 32 128 --q-lens 1 4 65 128
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size
from flashinfer.utils import is_sm12x_supported

_D_NOPE = 512
_D_ROPE = 64
_D_QK = _D_NOPE + _D_ROPE
_D_V = 512
_TILE = 128
_PAGE = 64
_ROW_BYTES = _D_NOPE + 4 * (_D_NOPE // _TILE) + 2 * _D_ROPE  # 656
_TOPK = 2048
_DECODE_CUTOFF = 64
_SM_SCALE = _D_QK**-0.5
_KV_SCALE_FORMAT = {"dsv3_2": "pow2_fp32", "glm_nsa": "arbitrary_fp32"}
_MODEL_TYPE = {"dsv3_2": 0, "glm_nsa": 2}  # sparse-MLA model ids of the formats
_VARIANTS = {
    0: "decode",
    1: "prefill-sg",
    2: "prefill-mg",
    3: "prefill-mg-dual",
    4: "prefill-swapab",
}


def _pack_kv(kv: torch.Tensor, fmt: str) -> torch.Tensor:
    """Pack bf16 ``[blocks, page, 576]`` into the 656-byte inline FP8 layout.

    Per token: 512 FP8 nope values, four FP32 per-128 tile scales, 64 BF16 rope
    values. DSv3.2 rounds each scale up to a power of two; GLM-NSA keeps it.
    """
    blocks, page, _ = kv.shape
    rows = kv.reshape(-1, _D_QK)
    nope = rows[:, :_D_NOPE].float().view(-1, _D_NOPE // _TILE, _TILE)
    scale = nope.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
    if fmt == "dsv3_2":
        scale = torch.pow(2.0, scale.log2().ceil())
    fp8 = (nope / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    packed = torch.cat(
        [
            fp8.view(torch.uint8).reshape(-1, _D_NOPE),
            scale.contiguous().view(torch.uint8),
            rows[:, _D_NOPE:].to(torch.bfloat16).contiguous().view(torch.uint8),
        ],
        dim=1,
    )
    return packed.view(blocks, 1, page, _ROW_BYTES)


def _make_cache(fmt: str, num_blocks: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    chunks = []
    for start in range(0, num_blocks, 1024):  # bound the bf16 staging copy
        count = min(1024, num_blocks - start)
        kv = torch.randn(
            count, _PAGE, _D_QK, device="cuda", dtype=torch.bfloat16, generator=g
        )
        chunks.append(_pack_kv((kv / 10).clamp(-1, 1), fmt))
    return torch.cat(chunks)


def _make_rows(
    num_tokens: int, num_heads: int, pool_tokens: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        num_tokens,
        num_heads,
        _D_QK,
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    indices = torch.randint(
        0,
        pool_tokens,
        (num_tokens, _TOPK),
        device="cuda",
        dtype=torch.int32,
        generator=g,
    )
    return (q / 10).clamp(-1, 1), indices


def _decode(
    fmt: str,
    kv_cache: torch.Tensor,
    workspace: torch.Tensor,
    q: torch.Tensor,
    indices: torch.Tensor,
    *,
    q_len: int,
    batch_invariant: bool,
    out: torch.Tensor | None = None,
    return_lse: bool = True,
):
    num_tokens, num_heads, _ = q.shape
    batch = num_tokens // q_len
    return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=q.view(batch, q_len, num_heads, _D_QK),
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=_D_NOPE,
        kv_lora_rank=_D_NOPE,
        qk_rope_head_dim=_D_ROPE,
        block_tables=indices.view(batch, q_len, _TOPK),
        seq_lens=None,
        max_seq_len=_TOPK,
        sparse_mla_top_k=_TOPK,
        out=out,
        bmm1_scale=_SM_SCALE,
        bmm2_scale=1.0,
        backend="sparse",
        return_lse=return_lse,
        kv_scale_format=_KV_SCALE_FORMAT[fmt],
        batch_invariant=batch_invariant,
    )


def _cupti_available() -> bool:
    try:
        from importlib.metadata import version

        # cupti-python's enum module warns about private names on import.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from cupti import cupti  # noqa: F401

        return int(version("cupti-python").split(".")[0]) >= 13
    except Exception:
        return False


def _timing(args: argparse.Namespace) -> tuple[bool, bool, str]:
    """``(enable_cupti, use_cuda_graph, label)`` with a per-call L2 flush."""
    if not args.no_cupti and _cupti_available():
        graph = not args.no_cuda_graph
        return (
            True,
            graph,
            f"CUPTI{' + CUDA graph' if graph else ''}, L2 flushed per call",
        )
    return (
        False,
        False,
        "CUDA events, eager, L2 flushed per call (includes host launch gaps; "
        "install cupti-python for kernel-only time)",
    )


def _num_blocks(args: argparse.Namespace) -> int:
    if args.num_blocks:
        return args.num_blocks
    pool_bytes = 8 * get_l2_cache_size(torch.device("cuda"))
    return max(math.ceil(pool_bytes / (_PAGE * _ROW_BYTES)), _TOPK // _PAGE)


def _print_header(args: argparse.Namespace, num_blocks: int) -> None:
    from flashinfer.mla._sparse_mla_sm120._calibration import default_cache_path

    props = torch.cuda.get_device_properties(0)
    package_dir = pathlib.Path(flashinfer.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    l2_mib = get_l2_cache_size(torch.device("cuda")) / 2**20
    pool_mib = num_blocks * _PAGE * _ROW_BYTES / 2**20
    calibration = default_cache_path().expanduser()
    print(
        f"GPU: {props.name} (SM{props.major}{props.minor}, "
        f"{props.multi_processor_count} SMs, L2 {l2_mib:.0f} MiB) | "
        f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"flashinfer {flashinfer.__version__} @ {commit} ({package_dir})"
    )
    print(
        f"KV pool: {num_blocks} pages x {_PAGE} x {_ROW_BYTES} B = "
        f"{pool_mib:.0f} MiB | top-k {_TOPK} | calibration store "
        f"{'present' if calibration.exists() else 'absent'} ({calibration})"
    )
    if args.command == "bench":
        print(f"Timing: {_timing(args)[2]}, median")
    print()


def _repro(args: argparse.Namespace) -> int:
    num_blocks = _num_blocks(args)
    _print_header(args, num_blocks)
    workspace = torch.empty(args.workspace_mb << 20, dtype=torch.uint8, device="cuda")
    heads = args.num_heads
    bi_failed, reproduced = False, False
    print(f"{heads} heads; the first request alone vs. joined by `others` requests\n")
    print(
        "| format | q_len | others | total T | batch_invariant | alone route | "
        "joined route | output rows differ | LSE rows differ | max abs diff | "
        "bitwise | repro status |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fmt in args.formats:
        kv_cache = _make_cache(fmt, num_blocks, args.seed)
        for q_len in args.q_lens:
            for others in args.others:
                total = (1 + others) * q_len
                q, indices = _make_rows(
                    total, heads, num_blocks * _PAGE, args.seed + q_len
                )
                for batch_invariant in (False, True):
                    bi_q_len = q_len if batch_invariant else 0
                    alone_out, alone_lse = _decode(
                        fmt,
                        kv_cache,
                        workspace,
                        q[:q_len],
                        indices[:q_len],
                        q_len=q_len,
                        batch_invariant=batch_invariant,
                    )
                    alone_route = _route(fmt, q_len, heads, bi_q_len)
                    all_out, all_lse = _decode(
                        fmt,
                        kv_cache,
                        workspace,
                        q,
                        indices,
                        q_len=q_len,
                        batch_invariant=batch_invariant,
                    )
                    joined_route = _route(fmt, total, heads, bi_q_len)
                    alone_out = alone_out.reshape(q_len, heads, _D_V)
                    first_out = all_out.reshape(total, heads, _D_V)[:q_len]
                    alone_lse = alone_lse.reshape(q_len, heads)
                    first_lse = all_lse.reshape(total, heads)[:q_len]
                    out_rows = int((alone_out != first_out).any(dim=-1).sum())
                    lse_rows = int((alone_lse != first_lse).sum())
                    max_diff = (alone_out.float() - first_out.float()).abs().max()
                    bitwise = out_rows == 0 and lse_rows == 0
                    if batch_invariant:
                        bi_failed |= not bitwise
                        verdict = ""
                    else:
                        reproduced |= not bitwise
                        verdict = "REPRODUCED" if not bitwise else "NOT REPRODUCED"
                    print(
                        f"| {fmt} | {q_len} | {others} | {q_len} -> {total} | "
                        f"{batch_invariant} | {alone_route} | {joined_route} | "
                        f"{out_rows}/{q_len * heads} | "
                        f"{lse_rows}/{q_len * heads} | {max_diff.item():.3g} | "
                        f"{'yes' if bitwise else 'no'} | {verdict} |"
                    )
    status = 0
    if bi_failed:
        print("\nFAIL: batch_invariant=True rows are not bit-identical.")
        status = 1
    if not reproduced:
        print(
            "\nNOT REPRODUCED: the default route was already invariant for every "
            "case on this device and shape."
        )
        status |= int(args.require_default_mismatch)
    return status


def _route(fmt: str, num_tokens: int, num_heads: int, bi_q_len: int) -> str:
    """Variant and cpb of the cached plan the last such call used.

    Reads the private functional plan cache. A key is ``(model, is_nvfp4,
    extra_fp4, preference, value_dim, decode_only, batch_invariant_q_len,
    tensor signatures)``; the signatures start with the query and then the
    cache and the indices. A batch-invariant call above the cutoff is matched
    by its first 64-row launch; all its launches share one cpb.
    """
    from flashinfer.mla._sparse_mla_sm120 import _prepared

    rows = num_tokens if not bi_q_len else min(num_tokens, _DECODE_CUTOFF)
    for key, prepared in reversed(_prepared._functional_plans.items()):
        signatures = key[7]
        if (
            key[0] == _MODEL_TYPE[fmt]
            and key[6] == bi_q_len
            and signatures[0][0] == (rows, num_heads, _D_QK)
            and signatures[2][0] == (rows, _TOPK)
        ):
            facts = prepared.plan.inspect()
            variant = _VARIANTS.get(int(facts["variant"]), str(facts["variant"]))
            if variant != "decode":
                return variant
            return f"decode cpb={int(facts['cpb'])}"
    warnings.warn(
        f"no cached plan for {fmt} T={num_tokens} heads={num_heads} "
        f"batch_invariant_q_len={bi_q_len}; the plan cache layout may have changed",
        stacklevel=2,
    )
    return "unknown"


def _median_us(fn, inputs: tuple, args: argparse.Namespace) -> float:
    fn(*inputs)
    torch.cuda.synchronize()
    enable_cupti, use_cuda_graph, _ = _timing(args)
    times = bench_gpu_time(
        fn,
        dry_run_time_ms=args.warmup_ms,
        repeat_time_ms=args.measure_ms,
        enable_cupti=enable_cupti,
        use_cuda_graph=use_cuda_graph,
        input_args=inputs,
        cold_l2_cache=True,
    )
    return float(np.median(times)) * 1e3


def _bench(args: argparse.Namespace) -> int:
    num_blocks = _num_blocks(args)
    _print_header(args, num_blocks)
    workspace = torch.empty(args.workspace_mb << 20, dtype=torch.uint8, device="cuda")
    print(
        "| format | heads | q_len | total T | default (us) | default route | "
        "batch_invariant (us) | batch_invariant route | BI chunks | ratio |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for fmt in args.formats:
        kv_cache = _make_cache(fmt, num_blocks, args.seed)
        for num_heads in args.num_heads:
            for q_len in args.q_lens:
                for num_tokens in args.num_tokens:
                    if num_tokens % q_len:
                        continue
                    q, indices = _make_rows(
                        num_tokens, num_heads, num_blocks * _PAGE, args.seed
                    )
                    out = torch.empty(
                        num_tokens // q_len,
                        q_len,
                        num_heads,
                        _D_V,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    us, route = {}, {}
                    for batch_invariant in (False, True):

                        def call(kv, q_, idx, out_, bi: bool = batch_invariant):
                            _decode(
                                fmt,
                                kv,
                                workspace,
                                q_,
                                idx,
                                q_len=q_len,
                                batch_invariant=bi,
                                out=out_,
                                return_lse=False,
                            )

                        us[batch_invariant] = _median_us(
                            call, (kv_cache, q, indices, out), args
                        )
                        route[batch_invariant] = _route(
                            fmt, num_tokens, num_heads, q_len if batch_invariant else 0
                        )
                    print(
                        f"| {fmt} | {num_heads} | {q_len} | {num_tokens} | "
                        f"{us[False]:.1f} | {route[False]} | {us[True]:.1f} | "
                        f"{route[True]} | {math.ceil(num_tokens / _DECODE_CUTOFF)} | "
                        f"{us[True] / us[False]:.2f}x |"
                    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("repro", "bench"):
        p = sub.add_parser(name)
        p.add_argument(
            "--formats",
            nargs="+",
            choices=tuple(_KV_SCALE_FORMAT),
            default=tuple(_KV_SCALE_FORMAT),
        )
        p.add_argument(
            "--num-blocks",
            type=int,
            default=0,
            help="KV pool pages (default: at least 8x the device L2)",
        )
        p.add_argument("--workspace-mb", type=int, default=512)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--no-calibration",
            action="store_true",
            help="point FLASHINFER_AUTOTUNE_DIR at an empty directory so the "
            "default route sees no calibration store (the issue's setup)",
        )
    repro = sub.choices["repro"]
    repro.add_argument("--num-heads", type=int, default=32)
    repro.add_argument("--q-lens", type=int, nargs="+", default=(1, 4, 64))
    repro.add_argument(
        "--others",
        type=int,
        nargs="+",
        default=(1, 64),
        help="other requests of the same q_len joined to the first one",
    )
    repro.add_argument(
        "--require-default-mismatch",
        action="store_true",
        help="exit non-zero if the default route never differed (pinned CI only)",
    )
    bench = sub.choices["bench"]
    bench.add_argument("--num-heads", type=int, nargs="+", default=(32,))
    bench.add_argument("--q-lens", type=int, nargs="+", default=(1, 4))
    bench.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=(1, 4, 8, 60, 63, 64, 65, 68, 128, 129, 512, 8192),
    )
    bench.add_argument("--warmup-ms", type=int, default=25)
    bench.add_argument("--measure-ms", type=int, default=100)
    bench.add_argument("--no-cupti", action="store_true")
    bench.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        sys.exit("This benchmark requires an SM120/SM121 GPU.")
    if args.no_calibration:
        # Read lazily by the sparse-MLA calibration store on its first refresh.
        os.environ["FLASHINFER_AUTOTUNE_DIR"] = tempfile.mkdtemp(
            prefix="flashinfer_no_calibration_"
        )
    sys.exit(_repro(args) if args.command == "repro" else _bench(args))


if __name__ == "__main__":
    main()
