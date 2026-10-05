"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

__doc__ = """SM90 (Hopper) native BF16 MegaMoE token sweep through ``MoEEpLayer``.

Times two arms at identical routing and expert weights on every EP rank:

* ``cand`` -- ``sm90_bf16_bf16_bf16_push_cake``: the Cake-generated native BF16
  push-style MegaMoE (bf16 dispatch payload, WGMMA FC1 with fused SwiGLU and
  FC2 with fp32 accumulation and a bf16 intermediate, bf16 combine wire, bf16
  output), selected with ``MegaConfig(megakernel=
  Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig(intermediate_size, top_k,
  capacity_factor, dedup_dispatch, clamp_limit), quantize_input=True,
  preprocess_weights=True)``.
* ``b1`` -- the same-precision split baseline through the same ``MoEEpLayer``:
  ``SplitConfig(comm=NCCLEPConfig(), kernel=<CUTLASS BF16 split kernel>)`` over
  the ``nccl_ep`` transport selected by ``--b1-transport`` (default ``ht``),
  fed the very same canonical bf16 ``MoEWeightPack(w13=[E_local, 2I, H]
  (gate | up rows), w2=[E_local, H, I])`` that ``cand`` receives.

What B1 is exactly, and why.  At this tree the stock split kernel
(``FusedMoeKernelConfig``) has no SM90 BF16 compute path:
``flashinfer/moe_ep/backends/split/kernel/fused_moe/weights.py::
materialize_fused_moe_weights`` materializes BF16 weights only for a
``TrtllmBf16Config`` candidate (``supported(arch)`` is ``arch in (100, 103,
107)``), and ``CutlassBf16Config`` -- the CUTLASS BF16 fused MoE, the only SM90
BF16 runner -- has no branch there and inherits ``supports_expert_parallelism =
False``.  ``--b1-backend cutlass_bf16`` (default) therefore uses a
benchmark-local ``SplitKernelBackend`` defined in this file and registered
lazily when the ``b1`` arm is first built (never at ``flashinfer`` import).  It
reuses the split path's own bridge helpers (``build_activation_pack`` /
``build_activation_pack_rank_major`` / ``reshape_for_combine``) and
re-expresses the compute as a local, non-EP ``MoELayer(MoEConfig(
RoutingConfig(num_experts=E_local, top_k), QuantConfig(BF16, BF16),
ExpertConfig(I), BackendOptions((CutlassBf16Config(),))))`` over this rank's
experts: LL expert_major rows arrive pre-routed (synthesized ``top_k=1``,
weight 1; ``combine`` applies the route weights), LL rank_major and HT rows
carry their received local top-k and are pre-reduced here at the real ``K``.
Weights go through ``CutlassBf16Config.prepare_weights`` after swapping the two
w13 halves: the fused_moe API consumes GEMM1 rows as ``[up, gate]`` while the
canonical pack and the torch reference are ``(gate | up)`` (one copy of the
local w13 shard per built layer).  Unless ``--no-b1-autotune``, the first
``b1`` forward of every point runs inside
``flashinfer.autotuner.autotune(True)`` so the inner layer profiles the CUTLASS
tactics for that token bucket.  ``--b1-backend trtllm_bf16`` keeps the stock
``FusedMoeKernelConfig(MoEConfig(..., BackendOptions((TrtllmBf16Config(),))))``
spelling (SM100 family only; on H100 those rows are recorded ``failed`` with
the exception class in ``detail``).

Transport.  ``--b1-transport ht`` (default) is the NCCL-EP high-throughput
algorithm (receive buffer ``[world, max_tokens_per_rank, H]``).
``ll_expert_major`` pads the receive buffer to ``E_local x max_tokens_per_rank
x world`` rows and runs out of memory at large token counts (those points
become ``skip_oom`` rows).  ``ll_rank_major`` produced whole-token output
corruption at ``tokens_per_rank >= 1024`` in our runs -- a transport issue,
not the compute -- and ``--check`` at the first token count does not cover
that regime.  Rank 0 prints a ``# WARNING`` line for the two non-default
transports.  B1's comm is ``nccl_ep`` only: ``nixl_ep`` needs a NIXL
rendezvous store (out of scope) and this tree has no torch all-to-all comm
backend.

Per (arm, tokens) point two series are timed with CUDA events on every rank,
barrier-aligned per iteration: ``eager`` repeats ``layer.forward`` and
``graph`` replays a ``torch.cuda.CUDAGraph`` captured after warmup (an arm
whose capture raises is recorded ``n/a`` with the exception class).  Rows carry
the per-rank medians, their max over ranks and TFLOPS at that max
(``2*T*K*H*2I + 2*T*K*H*I`` per rank).  ``--check`` compares both arms against
each other and against the independent torch reference in
``tests/moe_ep/_sm90_bf16_reference.py`` at the first token count
(``atol = rtol = 1e-2``, never loosened).  Layers are destroyed between points;
an OOM point becomes a ``skip_oom`` row and the sweep continues.

Routing is uniform random top-k per token from a fixed per-(rank, tokens) seed,
route weights = fp32 softmax over the selected k.  Expert weights are generated
per expert from ``--seed`` so each rank materializes only its local shard and
``--check`` can regenerate any expert for the reference.

Launch (one process per GPU, world size = EP size 2/4/8):

    python -m torch.distributed.run --nproc_per_node=8 \\
        benchmarks/bench_moe_ep_sm90_bf16_mega.py --geometry A

Geometries: A = hidden 7168 / intermediate 2048 / 256 experts / top-k 8
(default), B = 7168 / 3072 / 384 / 6, C = 4096 / 2048 / 256 / 6; each value can
be overridden explicitly.  Rank 0 prints ``BENCH_CSV`` rows (header once) and
optionally writes them to ``--output-csv``.
"""

import argparse
import contextlib
import gc
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_REFERENCE_PATH = os.path.join(_REPO, "tests", "moe_ep", "_sm90_bf16_reference.py")

# name: (hidden, intermediate (post-SwiGLU), total experts, top_k)
GEOMETRIES = {
    "A": (7168, 2048, 256, 8),
    "B": (7168, 3072, 384, 6),
    "C": (4096, 2048, 256, 6),
}
DEFAULT_TOKENS = "8,16,32,64,128,512,1024,2048,4096"
ARMS = ("cand", "b1")
MODES = ("eager", "graph")
B1_BACKENDS = ("cutlass_bf16", "trtllm_bf16")
B1_TRANSPORTS = ("ht", "ll_expert_major", "ll_rank_major")
B1_TRANSPORT_WARNINGS = {
    "ll_expert_major": "--b1-transport ll_expert_major pads the receive buffer to "
    "local_experts x max_tokens_per_rank x world rows and runs out of memory at "
    "large token counts (those points become skip_oom rows)",
    "ll_rank_major": "--b1-transport ll_rank_major produced whole-token output "
    "corruption at tokens_per_rank >= 1024 in our runs (nccl_ep transport issue, "
    "not the compute); --check only covers the first token count",
}
ATOL = 1e-2
RTOL = 1e-2

CSV_FIELDS = (
    "arm,mode,geometry,tokens_per_rank,max_tokens_per_rank,top_k,world_size,"
    "total_experts,local_experts,hidden,intermediate,warmup,iters,status,"
    "max_rank_median_us,min_rank_median_us,mean_rank_median_us,rank_median_us,"
    "flops_per_rank,tflops_at_max_rank,tok_s,detail"
)
CSV_HEADER = "BENCH_CSV," + CSV_FIELDS


def _csv_list(value: str, allowed: tuple[str, ...], flag: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad or not items:
        raise SystemExit(f"{flag}: expected a comma list from {allowed}, got {value!r}")
    return items


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--geometry",
        choices=sorted(GEOMETRIES),
        default="A",
        help="preset (hidden/intermediate/experts/top_k): "
        + ", ".join(f"{k}={v[0]}/{v[1]}/{v[2]}/{v[3]}" for k, v in GEOMETRIES.items()),
    )
    p.add_argument(
        "--hidden", type=int, default=None, help="override the preset hidden size"
    )
    p.add_argument(
        "--intermediate",
        type=int,
        default=None,
        help="override the preset post-SwiGLU (downproj) width; gate+up is 2x",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="override the preset TOTAL expert count across all EP ranks",
    )
    p.add_argument("--top-k", type=int, default=None, help="override the preset top-k")
    p.add_argument(
        "--tokens",
        type=str,
        default=DEFAULT_TOKENS,
        help="comma-separated tokens-per-rank sweep",
    )
    p.add_argument(
        "--max-tokens-per-rank",
        type=int,
        default=None,
        help="pin FleetParams.max_tokens_per_rank for every point (e.g. 8192); "
        "default: the current token count",
    )
    p.add_argument("--arms", type=str, default="cand,b1", help="comma list of cand,b1")
    p.add_argument(
        "--modes", type=str, default="eager,graph", help="comma list of eager,graph"
    )
    p.add_argument(
        "--b1-backend",
        choices=B1_BACKENDS,
        default="cutlass_bf16",
        help="B1 inner compute. cutlass_bf16 (default): the benchmark-local split "
        "kernel wrapping MoELayer(CutlassBf16Config), the SM90-capable BF16 "
        "runner; trtllm_bf16: the stock FusedMoeKernelConfig(TrtllmBf16Config) "
        "spelling (SM100-family runner only; its rows fail on H100).",
    )
    p.add_argument(
        "--b1-transport",
        choices=B1_TRANSPORTS,
        default="ht",
        help="nccl_ep algorithm/layout for B1: ht (default; high-throughput), "
        "ll_expert_major (low-latency, padded expert-major receive buffer; runs "
        "out of memory at large token counts), ll_rank_major (low-latency, "
        "rank-major; whole-token output corruption observed at tokens_per_rank "
        ">= 1024)",
    )
    p.add_argument(
        "--b1-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the first B1 forward of every point inside "
        "flashinfer.autotuner.autotune(True) so the inner CUTLASS tactic is tuned "
        "for that token bucket",
    )
    p.add_argument(
        "--dedup-dispatch", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--capacity-factor", type=float, default=1.0)
    p.add_argument(
        "--combine-wire",
        choices=("prereduced", "prereduced_hilo", "per_route"),
        default=None,
        help="cand combine wire: prereduced (default; one pre-reduced bf16 row per "
        "(token, source rank)) or per_route (one row per route, the round-1 wire "
        "kept for A/B); default follows FLASHINFER_SM90_CAKE_BF16_COMBINE_WIRE",
    )
    p.add_argument(
        "--clamp-limit",
        type=float,
        default=None,
        help="cand FC1 gate/up clamp (fp32, before SwiGLU); default none",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--cooldown-s",
        type=float,
        default=5.0,
        help="idle the GPUs before each timed series so clocks recover; 0 disables",
    )
    p.add_argument("--seed", type=int, default=1234, help="weights and routing seed")
    p.add_argument(
        "--check",
        action="store_true",
        help="before timing, compare the arms with each other and with the "
        "tests/moe_ep/_sm90_bf16_reference.py torch reference at the first "
        "token count (atol=rtol=1e-2)",
    )
    p.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="also write the BENCH_CSV rows to this file (rank 0 only)",
    )
    args = p.parse_args()
    hidden, inter, experts, top_k = GEOMETRIES[args.geometry]
    args.hidden = args.hidden or hidden
    args.intermediate = args.intermediate or inter
    args.num_experts = args.num_experts or experts
    args.top_k = args.top_k or top_k
    args.tokens_list = [int(t) for t in args.tokens.split(",") if t.strip()]
    args.arm_list = _csv_list(args.arms, ARMS, "--arms")
    args.mode_list = _csv_list(args.modes, MODES, "--modes")
    return args


def _flops_per_rank(tokens: int, top_k: int, hidden: int, inter: int) -> int:
    routed = tokens * top_k
    return 2 * routed * hidden * (2 * inter) + 2 * routed * hidden * inter


def _clean(text: str, limit: int = 200) -> str:
    return " ".join(str(text).split()).replace(",", ";")[:limit]


# --------------------------------------------------------------------- data
def _expert_pair(expert: int, *, hidden: int, inter: int, seed: int, device):
    """Canonical bf16 ``(w13 [2I, H] gate|up, w2 [H, I])`` of one global expert."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + expert)
    w13 = torch.randn(2 * inter, hidden, generator=g, device=device) * hidden**-0.5
    w2 = torch.randn(hidden, inter, generator=g, device=device) * inter**-0.5
    return w13.to(torch.bfloat16), w2.to(torch.bfloat16)


def _local_weights(args, rank: int, world: int, device):
    """This rank's expert shard, generated expert by expert (never the full table)."""
    import torch

    local = args.num_experts // world
    start = rank * local
    w13 = torch.empty(
        local, 2 * args.intermediate, args.hidden, dtype=torch.bfloat16, device=device
    )
    w2 = torch.empty(
        local, args.hidden, args.intermediate, dtype=torch.bfloat16, device=device
    )
    for i in range(local):
        a, b = _expert_pair(
            start + i,
            hidden=args.hidden,
            inter=args.intermediate,
            seed=args.seed,
            device=device,
        )
        w13[i].copy_(a)
        w2[i].copy_(b)
    return w13, w2


class _ExpertTable:
    """Duck-typed ``[E, rows, cols]`` weight table for ``reference_moe_bf16``.

    The reference only reads ``.shape`` and ``table[e]``.  Local experts come
    from the exact tensors the layer received; any other expert is regenerated
    from its seed on demand, so the global table (15 GB of ``w13`` for
    geometry A) never exists.  ``index`` selects w13 (0) or w2 (1).
    """

    def __init__(self, args, device, index: int, local_tensor, local_start: int):
        self._args = args
        self._device = device
        self._index = index
        self._local = local_tensor
        self._local_start = local_start
        self._cache: tuple[int, tuple] | None = None
        self.shape = (args.num_experts, *local_tensor.shape[1:])

    def __getitem__(self, expert):
        expert = int(expert)
        offset = expert - self._local_start
        if 0 <= offset < self._local.shape[0]:
            return self._local[offset]
        if self._cache is None or self._cache[0] != expert:
            pair = _expert_pair(
                expert,
                hidden=self._args.hidden,
                inter=self._args.intermediate,
                seed=self._args.seed,
                device=self._device,
            )
            self._cache = (expert, pair)
        return self._cache[1][self._index]


def _make_inputs(args, tokens: int, rank: int, device):
    """bf16 x [T, H], int32 global top-k ids [T, K], fp32 softmax route weights."""
    import torch

    g = torch.Generator(device=device).manual_seed(args.seed + 7919 * rank + tokens)
    x = torch.randn(tokens, args.hidden, generator=g, device=device).to(torch.bfloat16)
    logits = torch.randn(tokens, args.num_experts, generator=g, device=device)
    values, ids = logits.topk(args.top_k, dim=1)
    weights = torch.softmax(values, dim=-1).to(torch.float32)
    return x, ids.to(torch.int32).contiguous(), weights.contiguous()


# ------------------------------------------------------------------- layers
def _layer_kwargs(
    args, world: int, rank: int, pack, max_tokens: int, *, transport: str | None = None
) -> dict:
    """bootstrap / fleet_params / weights shared by both arms (test pattern).

    ``transport`` (b1 only) selects the nccl_ep algorithm/layout in
    ``FleetParams``; HT uses its FLAT receive layout and keeps the default
    ``EXPERT_MAJOR`` tag (``RANK_MAJOR`` is only valid with LL).
    """
    import torch.distributed as dist

    from flashinfer.moe_ep import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams

    group = dist.group.WORLD if world > 1 else None
    fleet_mode: dict = {}
    if transport == "ht":
        fleet_mode = {"algorithm": EpAlgorithm.HIGH_THROUGHPUT}
    elif transport == "ll_rank_major":
        fleet_mode = {"layout": EpLayout.RANK_MAJOR}
    return {
        "bootstrap": BootstrapConfig(world_size=world, rank=rank, process_group=group),
        "fleet_params": FleetParams(
            num_experts=args.num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=args.hidden,
            **fleet_mode,
        ),
        "weights": pack,
    }


def _build_cand(args, world: int, rank: int, pack, max_tokens: int):
    from flashinfer.moe_ep import (
        MegaConfig,
        MoEEpLayer,
        Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig,
    )

    return MoEEpLayer(
        **_layer_kwargs(args, world, rank, pack, max_tokens),
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig(
                intermediate_size=args.intermediate,
                top_k=args.top_k,
                capacity_factor=args.capacity_factor,
                dedup_dispatch=args.dedup_dispatch,
                clamp_limit=args.clamp_limit,
                combine_wire=args.combine_wire,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


def _b1_compute_rows(args, world: int, max_tokens: int) -> int:
    """Rows the inner compute sees per forward: the ``tune_max_num_tokens`` ceiling.

    LL expert_major dispatch returns ``[E_local, max_tokens * world, H]``; HT
    (``[world, max_tokens, H]``) and LL rank_major receive each token once.
    """
    if args.b1_transport == "ll_expert_major":
        return (args.num_experts // world) * max_tokens * world
    return max_tokens * world


# Benchmark-local split kernel for ``--b1-backend cutlass_bf16``.  Registered on
# first use by ``_b1_kernel_config_cls`` (importing ``flashinfer`` registers
# nothing); see the module docstring, "What B1 is exactly, and why".
_B1_KERNEL_NAME = "bench_sm90_cutlass_bf16"
_B1_CONFIG_CLS: Any = None


def _b1_kernel_config_cls() -> Any:
    """Register the CUTLASS BF16 split kernel once and return its config class."""
    global _B1_CONFIG_CLS
    if _B1_CONFIG_CLS is not None:
        return _B1_CONFIG_CLS

    import torch

    from flashinfer.fused_moe.api import QuantConfig, QuantFormat
    from flashinfer.moe_ep import EpAlgorithm, EpLayout
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge import (
        build_activation_pack,
        build_activation_pack_rank_major,
        reshape_for_combine,
    )
    from flashinfer.moe_ep.core.kernel.base import SplitKernelBackend
    from flashinfer.moe_ep.core.kernel.registry import register_split_kernel

    @dataclass(frozen=True)
    class CutlassBf16SplitConfig:
        """``SplitConfig(kernel=...)`` payload; ``kernel_name`` routes it to the backend."""

        intermediate_size: int
        top_k: int
        tune_max_num_tokens: int
        kernel_name: str = _B1_KERNEL_NAME

    @register_split_kernel(_B1_KERNEL_NAME)
    class CutlassBf16SplitBackend(SplitKernelBackend):
        """nccl_ep dispatch rows -> ``MoELayer(CutlassBf16Config)`` over the local experts.

        Mirrors ``FusedMoeSplitKernelBackend.compute`` with ``local_expert_offset
        = 0`` and ``num_experts = E_local`` so the CUTLASS runner (no EP support)
        sees a plain local MoE; the EP routing semantics stay in dispatch/combine.
        """

        def __init__(self, config: CutlassBf16SplitConfig) -> None:
            super().__init__(config)
            self._cfg = config
            self._quant = QuantConfig(
                weight=QuantFormat.BF16, activation=QuantFormat.BF16
            )
            self._local_experts = 0
            self._compute: Any = None

        @classmethod
        def kernel_name(cls) -> str:
            return _B1_KERNEL_NAME

        @staticmethod
        def _received_routing(fleet_params) -> bool:
            """HT and LL rank_major rows carry their received local top-k."""
            return (
                fleet_params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
                or fleet_params.layout is EpLayout.RANK_MAJOR
            )

        def preprocess_weights(self, weights, fleet_params):
            from flashinfer.fused_moe.api import CutlassBf16Config
            from flashinfer.fused_moe.api import MoEWeightPack as FusedMoEWeightPack

            w13, w2 = weights.w13, weights.w2
            local, two_i, hidden = w13.shape
            inter = self._cfg.intermediate_size
            if two_i != 2 * inter:
                raise ValueError(
                    f"w13 has {two_i} rows per expert, expected 2 * {inter}"
                )
            # Canonical (gate | up) rows -> the fused_moe public [up, gate] order
            # (``act(x @ w1[I:].T) * (x @ w1[:I].T)``), so the reference's
            # ``silu(gate) * up`` is what the CUTLASS kernel evaluates.
            up_gate = torch.empty_like(w13)
            up_gate[:, :inter].copy_(w13[:, inter:])
            up_gate[:, inter:].copy_(w13[:, :inter])
            view = CutlassBf16Config.prepare_weights(
                up_gate,
                w2,
                num_local_experts=local,
                hidden_size=hidden,
                intermediate_size=inter,
                device=w13.device,
            )
            pack = FusedMoEWeightPack()
            pack.prepare_for("cutlass_bf16", view)
            self._local_experts = local
            self._transformed_weights = pack
            return pack

        def _ensure_compute(self, fleet_params):
            if self._compute is None:
                from flashinfer.fused_moe.api import (
                    BackendOptions,
                    CutlassBf16Config,
                    ExecutionConfig,
                    ExpertConfig,
                    MoEConfig,
                    RoutingConfig,
                )
                from flashinfer.fused_moe.layer import MoELayer

                top_k = self._cfg.top_k if self._received_routing(fleet_params) else 1
                self._compute = MoELayer(
                    MoEConfig(
                        routing=RoutingConfig(
                            num_experts=self._local_experts, top_k=top_k
                        ),
                        quant=self._quant,
                        experts=ExpertConfig(
                            intermediate_size=self._cfg.intermediate_size
                        ),
                        backend=BackendOptions(candidates=(CutlassBf16Config(),)),
                        execution=ExecutionConfig(
                            tune_max_num_tokens=self._cfg.tune_max_num_tokens
                        ),
                    )
                )
            return self._compute

        def compute(self, ctx):
            fleet_params = ctx.fleet_params
            dim0, dim1, _ = ctx.expert_tensors.shape
            if self._received_routing(fleet_params):
                if ctx.recv_topk_idx is None or ctx.recv_topk_weights is None:
                    raise RuntimeError(
                        "HT / LL rank_major compute requires dispatch to return "
                        "recv_topk_idx / recv_topk_weights; got None."
                    )
                act = build_activation_pack_rank_major(
                    ctx.expert_tensors,
                    ctx.recv_topk_idx,
                    ctx.recv_topk_weights,
                    num_local_experts=self._local_experts,
                    local_expert_offset=0,
                    quant=self._quant,
                    hidden_size=fleet_params.token_hidden_size,
                )
            else:
                act = build_activation_pack(
                    ctx.expert_tensors,
                    local_expert_offset=0,
                    quant=self._quant,
                    hidden_size=fleet_params.token_hidden_size,
                    num_experts=self._local_experts,
                )
            out_2d = self._ensure_compute(fleet_params)(act, self._transformed_weights)
            return reshape_for_combine(out_2d, dim0, dim1)

    _B1_CONFIG_CLS = CutlassBf16SplitConfig
    return CutlassBf16SplitConfig


def _b1_moe_config(args, world: int, rank: int, max_tokens: int):
    """Stock ``--b1-backend trtllm_bf16`` spelling (SM100-family runner)."""
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        TrtllmBf16Config,
    )

    local = args.num_experts // world
    return MoEConfig(
        routing=RoutingConfig(num_experts=args.num_experts, top_k=args.top_k),
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(
            intermediate_size=args.intermediate,
            local_expert_offset=rank * local,
            local_num_experts=local,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(
            tune_max_num_tokens=_b1_compute_rows(args, world, max_tokens)
        ),
    )


def _build_b1(args, world: int, rank: int, pack, max_tokens: int):
    from flashinfer.moe_ep import (
        FusedMoeKernelConfig,
        MoEEpLayer,
        NCCLEPConfig,
        SplitConfig,
    )

    if args.b1_backend == "cutlass_bf16":
        kernel = _b1_kernel_config_cls()(
            intermediate_size=args.intermediate,
            top_k=args.top_k,
            tune_max_num_tokens=_b1_compute_rows(args, world, max_tokens),
        )
    else:
        kernel = FusedMoeKernelConfig(
            moe_config=_b1_moe_config(args, world, rank, max_tokens)
        )
    return MoEEpLayer(
        **_layer_kwargs(
            args, world, rank, pack, max_tokens, transport=args.b1_transport
        ),
        backend=SplitConfig(comm=NCCLEPConfig(), kernel=kernel),
    )


def _tune_b1(arm: "_Arm") -> None:
    """First B1 forward under ``autotune(True)``: the inner ``MoELayer`` profiles
    the CUTLASS tactics for this token bucket and caches the winner for the
    process; one plain forward follows so timing never sees tune mode."""
    import torch

    from flashinfer.autotuner import autotune

    with torch.inference_mode(), autotune(True):
        arm.eager()
        torch.cuda.synchronize()
    arm.eager()
    torch.cuda.synchronize()


class _Arm:
    """A built layer with its static input bundle: eager call and graph capture."""

    def __init__(self, name: str, layer, tensors) -> None:
        self.name = name
        self.layer = layer
        self.t = tensors
        self._graph = None
        self._graph_state = None
        self.graph_output = None

    def eager(self):
        return self.layer.forward(self.t)

    def capture(self, warmup: int):
        """Warm, capture one forward, return the replay callable."""
        import torch
        import torch.distributed as dist

        if self.name == "b1":
            # Split layers capture only through a persistent graph state, and
            # refuse to capture before an eager round trip on the same layer.
            state = self.layer.create_graph_state(self.t)
            self._graph_state = state

            def call():
                return self.layer.forward(self.t, graph_state=state)

            for _ in range(max(1, warmup)):
                call()
        else:
            call = self.eager
            for _ in range(max(1, warmup)):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    call()
                torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.graph_output = call()
        dist.barrier()
        self._graph = graph
        return graph.replay

    def destroy(self) -> None:
        import torch

        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        self._graph = None
        self.graph_output = None
        self._graph_state = None
        with contextlib.suppress(Exception):
            self.layer.destroy()


# ------------------------------------------------------------------- timing
def _time_calls(call, *, warmup: int, iters: int) -> list[float]:
    """Per-rank CUDA-event timings (us) of ``call``, barrier-aligned per iteration."""
    import torch
    import torch.distributed as dist

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        call()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) * 1e3)  # ms -> us
    return samples


def _cooldown(seconds: float) -> None:
    import torch
    import torch.distributed as dist

    if seconds <= 0:
        return
    torch.cuda.synchronize()
    dist.barrier()
    time.sleep(seconds)
    dist.barrier()


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


@dataclass
class _ArmPoint:
    """This rank's outcome of one (arm, tokens) point, keyed by mode."""

    # status values: pass | skip_oom | failed | n/a
    status: dict[str, str] = field(default_factory=dict)
    median_us: dict[str, float] = field(default_factory=dict)
    detail: dict[str, str] = field(default_factory=dict)


def _run_arm_point(
    args, arm_name, tokens, max_tokens, inputs, pack, world, rank, want_output
):
    """Build one arm, optionally keep an eager output, time the modes, destroy."""
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    x, ids, weights = inputs
    res = _ArmPoint()
    arm = None
    output = None
    try:
        if arm_name == "cand":
            layer = _build_cand(args, world, rank, pack, max_tokens)
            arm_ids = ids
        else:
            layer = _build_b1(args, world, rank, pack, max_tokens)
            arm_ids = ids.to(torch.int64)
        arm = _Arm(
            arm_name,
            layer,
            MoEEpTensors(hidden_states=x, topk_ids=arm_ids, topk_weights=weights),
        )
        if arm_name == "b1" and args.b1_autotune:
            _tune_b1(arm)
        if want_output:
            output = arm.eager().clone()
            torch.cuda.synchronize()
        if "eager" in args.mode_list:
            _cooldown(args.cooldown_s)
            samples = _time_calls(arm.eager, warmup=args.warmup, iters=args.iters)
            res.status["eager"] = "pass"
            res.median_us["eager"] = median(samples)
    except Exception as exc:  # noqa: BLE001 - the sweep must survive one bad point
        status = "skip_oom" if _is_oom(exc) else "failed"
        for mode in args.mode_list:
            res.status[mode] = status
            res.detail[mode] = _clean(f"{type(exc).__name__}: {exc}")
    else:
        if "graph" in args.mode_list:
            try:
                replay = arm.capture(args.warmup)
            except Exception as exc:  # noqa: BLE001 - uncapturable arm is a result
                res.status["graph"] = "skip_oom" if _is_oom(exc) else "n/a"
                res.detail["graph"] = _clean(f"{type(exc).__name__}: {exc}")
            else:
                try:
                    _cooldown(args.cooldown_s)
                    samples = _time_calls(replay, warmup=args.warmup, iters=args.iters)
                    res.status["graph"] = "pass"
                    res.median_us["graph"] = median(samples)
                except Exception as exc:  # noqa: BLE001
                    res.status["graph"] = "skip_oom" if _is_oom(exc) else "failed"
                    res.detail["graph"] = _clean(f"{type(exc).__name__}: {exc}")
    finally:
        if arm is not None:
            arm.destroy()
        gc.collect()
        torch.cuda.empty_cache()
    return res, output


def _gather(res: _ArmPoint, world: int) -> list[tuple[dict, dict, dict]]:
    """Collective agreement so every rank emits the same status per point."""
    import torch.distributed as dist

    gathered: list = [None] * world
    dist.all_gather_object(gathered, (res.status, res.median_us, res.detail))
    dist.barrier()
    return gathered


def _emit_rows(
    args, *, arm, tokens, max_tokens, world, gathered, header_done, csv_file
) -> None:
    flops = _flops_per_rank(tokens, args.top_k, args.hidden, args.intermediate)
    prefix = (
        f"{arm},{{mode}},{args.geometry},{tokens},{max_tokens},{args.top_k},{world},"
        f"{args.num_experts},{args.num_experts // world},{args.hidden},"
        f"{args.intermediate},{args.warmup},{args.iters}"
    )
    if not header_done:
        print(CSV_HEADER, flush=True)
        if csv_file is not None:
            csv_file.write(CSV_FIELDS + "\n")
    for mode in args.mode_list:
        statuses = [g[0].get(mode, "failed") for g in gathered]
        if all(s == "pass" for s in statuses):
            meds = [g[1][mode] for g in gathered]
            worst = max(meds)
            row = (
                f"{prefix.format(mode=mode)},pass,{worst:.2f},{min(meds):.2f},"
                f"{fmean(meds):.2f},{';'.join(f'{m:.2f}' for m in meds)},{flops},"
                f"{flops / worst / 1e6:.2f},{tokens * world / (worst * 1e-6):.1f},"
            )
        else:
            if "skip_oom" in statuses:
                status = "skip_oom"
            elif "failed" in statuses:
                status = "failed"
            else:
                status = "n/a"
            details = sorted({g[2][mode] for g in gathered if g[2].get(mode)})
            row = (
                f"{prefix.format(mode=mode)},{status},nan,nan,nan,,{flops},nan,nan,"
                f"{' | '.join(details)[:300]}"
            )
        print(f"BENCH_CSV,{row}", flush=True)
        if csv_file is not None:
            csv_file.write(row + "\n")
            csv_file.flush()


# -------------------------------------------------------------------- check
def _load_reference():
    spec = importlib.util.spec_from_file_location(
        "_sm90_bf16_reference", _REFERENCE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_check(
    args, tokens, outputs, inputs, local_w13, local_w2, world, rank, device
) -> None:
    """Arms vs each other and vs the independent torch reference; prints mismatches."""
    import torch.distributed as dist

    ref = _load_reference()
    x, ids, weights = inputs
    local_start = rank * (args.num_experts // world)
    table13 = _ExpertTable(args, device, 0, local_w13, local_start)
    table2 = _ExpertTable(args, device, 1, local_w2, local_start)
    reference = ref.reference_moe_bf16(
        x, ids, weights, table13, table2, clamp=args.clamp_limit
    )
    mismatches = 0
    parts = []
    for name in args.arm_list:
        out = outputs.get(name)
        if out is None:
            parts.append(f"{name}: unavailable")
            continue
        stats = ref.compare_bf16(out, reference, atol=ATOL, rtol=RTOL)
        mismatches += stats["mismatches"]
        parts.append(f"{name} vs reference: {stats}")
    if all(outputs.get(name) is not None for name in ARMS):
        stats = ref.compare_bf16(outputs["cand"], outputs["b1"], atol=ATOL, rtol=RTOL)
        mismatches += stats["mismatches"]
        parts.append(f"cand vs b1: {stats}")
    print(f"[check] rank{rank} tokens={tokens}: " + "; ".join(parts), flush=True)
    totals: list = [None] * world
    dist.all_gather_object(totals, mismatches)
    if rank == 0:
        verdict = "PASS" if sum(totals) == 0 else "FAIL"
        print(
            f"CHECK_RESULT,{verdict},tokens={tokens},atol={ATOL},rtol={RTOL},"
            f"mismatches_per_rank={';'.join(str(t) for t in totals)}",
            flush=True,
        )


# --------------------------------------------------------------------- main
def main() -> int:
    args = _parse_args()

    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEWeightPack

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_kwargs = {"device_id": device} if args.dist_backend == "nccl" else {}
    # Cold JIT builds at the first forward can exceed torch's default watchdog.
    dist.init_process_group(
        backend=args.dist_backend, timeout=timedelta(minutes=60), **init_kwargs
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.num_experts % world != 0:
        raise SystemExit(
            f"--num-experts ({args.num_experts}) must be divisible by the torchrun "
            f"world size ({world})"
        )
    if rank == 0:
        print(
            f"# geometry={args.geometry} hidden={args.hidden} intermediate={args.intermediate} "
            f"experts={args.num_experts} top_k={args.top_k} world={world} arms={args.arm_list} "
            f"modes={args.mode_list} b1_backend={args.b1_backend} b1_comm=nccl_ep "
            f"b1_transport={args.b1_transport} b1_autotune={args.b1_autotune} "
            f"seed={args.seed}",
            flush=True,
        )
        warning = B1_TRANSPORT_WARNINGS.get(args.b1_transport)
        if warning and "b1" in args.arm_list:
            print(f"# WARNING: {warning}", flush=True)

    local_w13, local_w2 = _local_weights(args, rank, world, device)
    pack = MoEWeightPack(w13=local_w13, w2=local_w2)

    csv_file = None
    if rank == 0 and args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        csv_file = open(args.output_csv, "w")  # noqa: SIM115 - closed in finally
        print(f"# output_csv: {args.output_csv}", flush=True)
    header_done = False
    try:
        for index, tokens in enumerate(args.tokens_list):
            max_tokens = args.max_tokens_per_rank or tokens
            if tokens > max_tokens:
                if rank == 0:
                    print(
                        f"# skip tokens_per_rank={tokens} > --max-tokens-per-rank={max_tokens}"
                    )
                continue
            inputs = _make_inputs(args, tokens, rank, device)
            check_here = args.check and index == 0
            outputs: dict = {}
            for arm in args.arm_list:
                if rank == 0:
                    print(
                        f"# [sweep] arm={arm} tokens_per_rank={tokens} max_tokens={max_tokens}",
                        flush=True,
                    )
                res, out = _run_arm_point(
                    args, arm, tokens, max_tokens, inputs, pack, world, rank, check_here
                )
                outputs[arm] = out
                gathered = _gather(res, world)
                if rank == 0:
                    _emit_rows(
                        args,
                        arm=arm,
                        tokens=tokens,
                        max_tokens=max_tokens,
                        world=world,
                        gathered=gathered,
                        header_done=header_done,
                        csv_file=csv_file,
                    )
                header_done = True
            if check_here:
                _run_check(
                    args,
                    tokens,
                    outputs,
                    inputs,
                    local_w13,
                    local_w2,
                    world,
                    rank,
                    device,
                )
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if csv_file is not None:
            csv_file.close()
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
