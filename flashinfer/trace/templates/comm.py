# Copyright (c) 2025 by FlashInfer team.
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

"""TraceTemplates for distributed communication ops."""

from typing import Any, cast

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _allreduce_fusion_reference(
    input: torch.Tensor,
    workspace,
    pattern: int,
    launch_with_pdl: bool = False,
    trigger_completion_at_end: bool = True,
    output=None,
    residual_out=None,
    norm_out=None,
    quant_out=None,
    scale_out=None,
    residual_in=None,
    rms_gamma=None,
    rms_eps: float = 1e-6,
    weight_bias: float = 0.0,
    **_unused,
):
    """Single-rank reference for allreduce_fusion.

    AllReduce is a no-op in a single-process reference (the kernel under
    test normally sums across ranks). This reference therefore treats
    ``input`` as the already-reduced tensor and models the *fusion* side
    of the op:

    - pattern 0 (kAllReduce): passthrough input.
    - pattern 1 (kARResidualRMSNorm): ``residual_out = input + residual_in``;
      ``norm_out = rmsnorm(residual_out, weight_bias + rms_gamma, rms_eps)``.
      ``weight_bias=0`` is standard RMSNorm; ``weight_bias=1`` is Gemma /
      Qwen3.5 style (``(1 + gamma) * x * rsqrt(...)``).

    Quantized / MoE patterns (>= 2) are outside the single-rank scope —
    this reference raises ``NotImplementedError`` for them and callers
    should exercise the real multi-rank kernel for coverage.
    """
    if pattern == 0:
        out = input.clone()
        if output is not None:
            output.copy_(out)
        return out
    if pattern == 1:
        if residual_in is None or rms_gamma is None:
            raise ValueError(
                "pattern=1 (kARResidualRMSNorm) requires residual_in and rms_gamma"
            )
        pre = input.to(torch.float32) + residual_in.to(torch.float32)
        inv_rms = torch.rsqrt(pre.pow(2).mean(dim=-1, keepdim=True) + float(rms_eps))
        normed = (pre * inv_rms) * (float(weight_bias) + rms_gamma.to(torch.float32))
        pre_dtype = pre.to(input.dtype)
        normed_dtype = normed.to(input.dtype)
        if residual_out is not None:
            residual_out.copy_(pre_dtype)
        if norm_out is not None:
            norm_out.copy_(normed_dtype)
        return normed_dtype
    raise NotImplementedError(
        f"allreduce_fusion reference does not model pattern={pattern} "
        "(quantized / MoE patterns are multi-rank-only)"
    )


def _allreduce_fusion_init(
    *,
    num_tokens: int,
    hidden_dim: int = 4096,
    pattern: int = 1,  # AR + Residual + RMSNorm
    device: str = "cuda",
    seed: int = 0,
):
    """Build per-rank inputs for ``allreduce_fusion``.

    Note: ``workspace`` is an opaque multi-rank IPC handle and is **not**
    initialized here — the caller must construct it (see
    ``tests/comm/`` for end-to-end multi-rank examples). This init returns
    everything else needed for the AR+Residual+RMSNorm fusion path
    (pattern=1).
    """
    torch.manual_seed(seed)
    inp = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(inp)
    rms_gamma = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)
    return {
        "input": inp,
        "residual_in": residual,
        "rms_gamma": rms_gamma,
        "rms_eps": 1e-6,
        "pattern": int(pattern),
    }


allreduce_fusion_trace = TraceTemplate(
    op_type="comm",
    name_prefix="allreduce_fusion",
    description=(
        "TRT-LLM / MNNVL fused AllReduce + (Residual + RMSNorm + optional "
        "FP8/FP4 Quantize). The reference models the fusion side of the op "
        "under the assumption that the input has already been reduced "
        "(AllReduce is a no-op single-rank); multi-rank correctness is "
        "exercised by tests/comm/."
    ),
    axes={
        "num_tokens": Var(description="Token count along dim 0."),
        "hidden_dim": Const(abbrev="h"),
    },
    inputs={
        "input": Tensor(
            ["num_tokens", "hidden_dim"],
            description="Pre-reduction token activations (this rank's shard).",
        ),
        "workspace": Scalar(
            "int64",
            description=(
                "AllReduceFusionWorkspace handle (opaque to the trace; "
                "its shape/content are backend-specific)."
            ),
        ),
        "pattern": Scalar(
            "int32",
            description=(
                "AllReduceFusionPattern enum: 0=AllReduce, "
                "1=AR+Residual+RMSNorm, 2..5=with FP8/FP4 quant, "
                "6=MoE reduction, 7=MoE finalize."
            ),
        ),
        "residual_in": Tensor(
            ["num_tokens", "hidden_dim"],
            optional=True,
            description="Residual to add (patterns 1..5).",
        ),
        "rms_gamma": Tensor(
            ["hidden_dim"],
            optional=True,
            description="RMSNorm weight (patterns 1..5).",
        ),
        "rms_eps": Scalar("float32", optional=True),
        "weight_bias": Scalar(
            "float32",
            optional=True,
            description=(
                "Bias added to rms_gamma before scaling. 0.0 (default) is "
                "standard RMSNorm; 1.0 selects Gemma / Qwen3.5 RMSNorm "
                "((1 + gamma) * x * rsqrt(...))."
            ),
        ),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "hidden_dim"],
            dtype_from="input",
            description="Main output; semantics depend on pattern.",
        ),
    },
    tags=["status:verified", "stage:comm", "fused"],
    reference=_allreduce_fusion_reference,
    init=_allreduce_fusion_init,
)


# ── DCP all-to-all (context-parallel attention reduction) ────────────────────


@torch.no_grad()
def _decode_cp_a2a_alltoall_reference(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    workspace,
    cp_rank: int,
    cp_size: int,
    enable_pdl=None,
    **_unused,
):
    """Single-rank reference for the DCP all-to-all.

    The kernel is a multi-rank exchange: each rank sends its
    ``partial_o[..., peer, :]`` slice to the corresponding peer and
    receives the gathered contributions. In a single-process reference we
    return ``partial_o`` and ``softmax_stats`` unchanged — the trace
    captures the schema; multi-rank correctness is tested under
    ``tests/comm/``.
    """
    return partial_o.clone(), softmax_stats.clone()


def _decode_cp_a2a_alltoall_init(
    *,
    batch_dim: int,
    cp_size: int,
    head_dim: int = 128,
    stats_dim: int = 2,
    ws_elems_per_rank: int = 1,
    cp_rank: int = 0,
    device: str = "cuda",
    seed: int = 0,
):
    """Build per-rank inputs for ``decode_cp_a2a_alltoall``.

    Like ``allreduce_fusion``, the ``workspace`` is a multi-rank IPC
    handle and not constructed here. ``cp_rank``/``cp_size`` default to
    a single-rank dummy invocation.
    """
    torch.manual_seed(seed)
    partial_o = torch.randn(
        batch_dim, cp_size, head_dim, dtype=torch.bfloat16, device=device
    )
    softmax_stats = torch.randn(
        batch_dim, cp_size, stats_dim, dtype=torch.float32, device=device
    )
    workspace = torch.zeros(
        cp_size, ws_elems_per_rank, dtype=torch.int64, device=device
    )
    return {
        "partial_o": partial_o,
        "softmax_stats": softmax_stats,
        "workspace": workspace,
        "cp_rank": int(cp_rank),
        "cp_size": int(cp_size),
    }


decode_cp_a2a_alltoall_trace = TraceTemplate(
    op_type="comm",
    name_prefix="decode_cp_a2a_alltoall",
    description=(
        "Context-parallel attention all-to-all reduction. Each rank ships "
        "its ``partial_o[..., peer, :]`` slice to peer ``peer`` and "
        "receives all peers' contributions in return. Used during paged "
        "decode with context-parallelism. Single-rank reference is a "
        "passthrough; multi-rank correctness is exercised by tests/comm."
    ),
    axes={
        "batch_dim": Var(description="Leading batch dimension(s)."),
        "cp_size": Var(description="Context-parallel group size."),
        "head_dim": Const(abbrev="d"),
        "stats_dim": Const(
            description="Softmax stats trailing dim (>=2, even).", abbrev="s"
        ),
        "ws_elems_per_rank": Var(),
    },
    inputs={
        "partial_o": Tensor(
            ["batch_dim", "cp_size", "head_dim"],
            description="Per-rank partial attention outputs [..., cp_size, D].",
        ),
        "softmax_stats": Tensor(
            ["batch_dim", "cp_size", "stats_dim"],
            description="Per-rank softmax stats [..., cp_size, S].",
        ),
        "workspace": Tensor(["cp_size", "ws_elems_per_rank"], dtype="int64"),
        "cp_rank": Scalar("int32"),
        "cp_size": Scalar("int32"),
    },
    outputs={
        "partial_o_out": Tensor(
            ["batch_dim", "cp_size", "head_dim"], dtype_from="partial_o"
        ),
        "softmax_stats_out": Tensor(
            ["batch_dim", "cp_size", "stats_dim"], dtype_from="softmax_stats"
        ),
    },
    tags=["status:verified", "stage:comm"],
    reference=_decode_cp_a2a_alltoall_reference,
    init=_decode_cp_a2a_alltoall_init,
)


# ── PCIe IPC all-reduce (intra-node, no NVLink) ──────────────────────────────


@torch.no_grad()
def _pcie_ipc_all_reduce_reference(
    inp: torch.Tensor,
    *,
    out: torch.Tensor = None,
    config=None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Single-rank reference: an all-reduce over one rank is the identity.

    Same modelling choice as ``allreduce_fusion`` above -- the trace runs in a
    single process, so the cross-rank reduction cannot be exercised here.
    Multi-rank correctness is covered by
    ``tests/comm/test_pcie_ipc_all_reduce.py``, which compares against NCCL at
    zero tolerance.
    """
    return inp.clone() if out is None else out.copy_(inp)


def _pcie_ipc_all_reduce_init(
    *,
    num_tokens: int,
    hidden_dim: int = 6144,
    device: str = "cuda",
    seed: int = 0,
):
    """Build this rank's input for ``PcieIpcAllReduceWorkspace.all_reduce``.

    The workspace itself is an opaque multi-rank IPC handle bound to ``self``
    and is not built here; see ``tests/comm/test_pcie_ipc_all_reduce.py``.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "inp": torch.randn(
            num_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    }


pcie_ipc_all_reduce_trace = TraceTemplate(
    op_type="comm",
    name_prefix="pcie_ipc_all_reduce",
    description=(
        "Custom all-reduce for intra-node PCIe machines without NVLink. The "
        "launch configuration follows the payload in bytes -- a seed default "
        "until the workspace is tuned on the machine it runs on -- so the "
        "traced axes are the ones that select it."
    ),
    axes={
        "num_tokens": Var(description="Token count along dim 0."),
        "hidden_dim": Const(abbrev="h"),
    },
    inputs={
        "inp": Tensor(
            ["num_tokens", "hidden_dim"],
            description="Pre-reduction token activations (this rank's shard).",
        ),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "hidden_dim"],
            dtype_from="inp",
            description="Reduced activations.",
        ),
    },
    tags=["status:verified", "stage:comm"],
    reference=_pcie_ipc_all_reduce_reference,
    init=_pcie_ipc_all_reduce_init,
)


# ── Low-precision Ulysses A2A (flashinfer.comm.ulysses_lowp) ─────────────────
#
# Quantization primitives around the sequence->head all-to-all of Ulysses
# attention: INT8 Q/K on SageAttention2's global 32/64-token grids, FP8 V per
# channel, packed into a destination-major uint8 payload (ABI v3), and the
# receiver-side unpack into the pre-quantized operands SageAttention consumes.
# Shapes below use the MiniMax-H3 deployment as defaults (56 heads, D=128,
# Ulysses P=8, shard L=4736 = 37 x 128).

_ULYSSES_LOWP_HEAD_DIM = 128


def _ulysses_lowp_amax_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    from flashinfer.trace import default_check

    # Reductions over bf16/fp16 values converted to fp32: only the fp32
    # summation order differs from torch, so the tolerance is a few ULPs of the
    # channel sums (tests/comm/test_ulysses_lowp.py uses the same oracle).
    rtol = 1e-3 if rtol is None else rtol
    atol = 1e-2 if atol is None else atol
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


@torch.no_grad()
def _ulysses_lowp_k_sum_v_amax_reference(k, v, **_unused):
    """Per-(batch, head, channel) fp32 K sum and V |max| over the local shard."""
    return k.float().sum(dim=1), v.float().abs().amax(dim=1)


def _ulysses_lowp_k_sum_v_amax_init(
    *,
    batch: int = 1,
    local_sequence: int = 4736,
    num_heads: int = 56,
    head_dim: int = _ULYSSES_LOWP_HEAD_DIM,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.comm.ulysses_lowp.k_sum_v_amax`` (one
    MiniMax-H3 Ulysses shard at P=8)."""
    torch.manual_seed(seed)
    k = torch.randn(
        batch, local_sequence, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    return {"k": k, "v": torch.randn_like(k)}


ulysses_lowp_k_sum_v_amax_trace = TraceTemplate(
    op_type="comm",
    name_prefix="ulysses_lowp_k_sum_v_amax",
    description=(
        "Local statistics for low-precision Ulysses A2A: per-channel K sum and "
        "V absolute maximum over this rank's sequence shard (two-stage, "
        "deterministic fp32 reduction). The caller AllGathers them to form "
        "the global K mean and V scale."
    ),
    axes={
        "batch": Var(),
        "local_sequence": Var(description="Tokens on this rank's shard."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d", value=_ULYSSES_LOWP_HEAD_DIM),
    },
    inputs={
        "k": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "v": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
    },
    outputs={
        "k_sum": Tensor(["batch", "num_heads", "head_dim"], dtype="float32"),
        "v_amax": Tensor(["batch", "num_heads", "head_dim"], dtype="float32"),
    },
    tags=["status:verified", "stage:comm"],
    reference=_ulysses_lowp_k_sum_v_amax_reference,
    check=_ulysses_lowp_amax_check,
    init=_ulysses_lowp_k_sum_v_amax_init,
)


def _ulysses_lowp_grouped_amax_reference(x, group, rank, world_size, mean=None):
    """Global-grid grouped |x| (or |x - mean|) amax of this rank's shard: slot
    s holds global group ``group_first + s``, floored at 1e-7; untouched slots
    of the ``slots(L, group)`` allocation stay zero."""
    batch, local_sequence, num_heads, _ = x.shape
    slots = (local_sequence + 2 * group - 2) // group
    group_first = (rank * local_sequence) // group
    group_last = (rank * local_sequence + local_sequence - 1) // group
    xf = x.float()
    if mean is not None:
        xf = xf - mean.float().unsqueeze(1)
    out = torch.zeros(batch, num_heads, slots, dtype=torch.float32, device=x.device)
    for slot, g in enumerate(range(group_first, group_last + 1)):
        lo = max(g * group, rank * local_sequence) - rank * local_sequence
        hi = min((g + 1) * group, (rank + 1) * local_sequence) - rank * local_sequence
        out[:, :, slot] = xf[:, lo:hi].abs().amax(dim=(1, 3)).clamp_(min=1e-7)
    return out


@torch.no_grad()
def _ulysses_lowp_q_grouped_amax_reference(q, rank, world_size, **_unused):
    return _ulysses_lowp_grouped_amax_reference(q, 32, rank, world_size)


# The rendered reference source must be self-contained (see
# _render_reference_source): declare the shared helper as a dependency.
cast(Any, _ulysses_lowp_q_grouped_amax_reference)._trace_reference_dependencies = (
    _ulysses_lowp_grouped_amax_reference,
)


def _ulysses_lowp_q_grouped_amax_init(
    *,
    batch: int = 1,
    local_sequence: int = 4736,
    q_slots: int = 0,  # derived
    num_heads: int = 56,
    head_dim: int = _ULYSSES_LOWP_HEAD_DIM,
    rank: int = 0,
    world_size: int = 8,
    device: str = "cuda",
    seed: int = 0,
):
    del q_slots  # derived from local_sequence: slots(L, 32)
    torch.manual_seed(seed)
    q = torch.randn(
        batch, local_sequence, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    return {"q": q, "rank": int(rank), "world_size": int(world_size)}


ulysses_lowp_q_grouped_amax_trace = TraceTemplate(
    op_type="comm",
    name_prefix="ulysses_lowp_q_grouped_amax",
    description=(
        "Per-32-token-group |Q| amax on the GLOBAL sequence grid for this "
        "rank's shard (SageAttention2 per-warp Q scale); slot s is global "
        "group group_first(rank)+s."
    ),
    axes={
        "batch": Var(),
        "local_sequence": Var(),
        "q_slots": Var(description="slots(local_sequence, 32) allocation."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d", value=_ULYSSES_LOWP_HEAD_DIM),
        "world_size": Const(abbrev="p", description="Ulysses group size."),
    },
    inputs={
        "q": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "rank": Scalar("int32"),
        "world_size": Scalar("int32"),
    },
    outputs={
        "amax": Tensor(["batch", "num_heads", "q_slots"], dtype="float32"),
    },
    constraints=["q_slots == (local_sequence + 62) // 32"],
    tags=["status:verified", "stage:comm"],
    reference=_ulysses_lowp_q_grouped_amax_reference,
    check=_ulysses_lowp_amax_check,
    init=_ulysses_lowp_q_grouped_amax_init,
)


@torch.no_grad()
def _ulysses_lowp_k_grouped_amax_reference(
    k, k_mean_global, rank, world_size, used_sequence=None, **_unused
):
    """Grouped |K - k_mean| amax with the API's live-prefix repair: when the
    packed sequence carries a zero tail in rows ``[used_sequence, S)``, the ONE
    64-token group mixing live and padded rows is reduced over its live rows
    only (a zero row would contribute |0 - k_mean|); fully padded groups keep
    the plain value and are never consumed."""
    out = _ulysses_lowp_grouped_amax_reference(
        k, 64, rank, world_size, mean=k_mean_global
    )
    local_sequence = k.shape[1]
    global_sequence = local_sequence * world_size
    if (
        used_sequence is None
        or used_sequence >= global_sequence
        or used_sequence % 64 == 0
    ):
        return out
    tail_group = (used_sequence - 1) // 64
    group_first = (rank * local_sequence) // 64
    group_last = (rank * local_sequence + local_sequence - 1) // 64
    if not group_first <= tail_group <= group_last:
        return out
    lo = max(tail_group * 64, rank * local_sequence) - rank * local_sequence
    hi = min(used_sequence, (rank + 1) * local_sequence) - rank * local_sequence
    if hi > lo:
        kc = k[:, lo:hi].float() - k_mean_global.float().unsqueeze(1)
        out[:, :, tail_group - group_first] = kc.abs().amax(dim=(1, 3)).clamp_(min=1e-7)
    else:
        out[:, :, tail_group - group_first] = 1e-7
    return out


cast(Any, _ulysses_lowp_k_grouped_amax_reference)._trace_reference_dependencies = (
    _ulysses_lowp_grouped_amax_reference,
)


def _ulysses_lowp_k_grouped_amax_init(
    *,
    batch: int = 1,
    local_sequence: int = 4736,
    k_slots: int = 0,  # derived
    num_heads: int = 56,
    head_dim: int = _ULYSSES_LOWP_HEAD_DIM,
    rank: int = 0,
    world_size: int = 8,
    device: str = "cuda",
    seed: int = 0,
):
    del k_slots  # derived from local_sequence: slots(L, 64)
    torch.manual_seed(seed)
    k = torch.randn(
        batch, local_sequence, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k_mean_global = k.float().mean(dim=1).to(k.dtype).contiguous()
    return {
        "k": k,
        "k_mean_global": k_mean_global,
        "rank": int(rank),
        "world_size": int(world_size),
    }


ulysses_lowp_k_grouped_amax_trace = TraceTemplate(
    op_type="comm",
    name_prefix="ulysses_lowp_k_grouped_amax",
    description=(
        "Per-64-token-group |K - k_mean| amax on the GLOBAL sequence grid for "
        "this rank's shard (SageAttention2 smooth-K per-block scale). "
        "``used_sequence`` repairs the one group that mixes live and "
        "zero-padded rows."
    ),
    axes={
        "batch": Var(),
        "local_sequence": Var(),
        "k_slots": Var(description="slots(local_sequence, 64) allocation."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d", value=_ULYSSES_LOWP_HEAD_DIM),
        "world_size": Const(abbrev="p", description="Ulysses group size."),
    },
    inputs={
        "k": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "k_mean_global": Tensor(
            ["batch", "num_heads", "head_dim"],
            description="Global K channel mean in K's dtype.",
        ),
        "rank": Scalar("int32"),
        "world_size": Scalar("int32"),
        "used_sequence": Scalar("int32", optional=True),
    },
    outputs={
        "amax": Tensor(["batch", "num_heads", "k_slots"], dtype="float32"),
    },
    constraints=["k_slots == (local_sequence + 126) // 64"],
    tags=["status:verified", "stage:comm"],
    reference=_ulysses_lowp_k_grouped_amax_reference,
    check=_ulysses_lowp_amax_check,
    init=_ulysses_lowp_k_grouped_amax_init,
)


def _ulysses_lowp_quant_qkv_pack_fused_init(
    *,
    batch: int = 1,
    local_sequence: int = 4736,
    chunk_bytes: int = 0,  # derived
    num_heads: int = 56,
    head_dim: int = _ULYSSES_LOWP_HEAD_DIM,
    rank: int = 0,
    world_size: int = 8,
    device: str = "cuda",
    seed: int = 0,
):
    """One rank's shard plus the AllGathered global statistics it would hold
    (mean / 2.25-scaled amax computed here as a single-rank stand-in)."""
    del chunk_bytes  # derived: payload_spec(...)["chunk_bytes"]
    torch.manual_seed(seed)
    q = torch.randn(
        batch, local_sequence, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k, v = torch.randn_like(q), torch.randn_like(q)
    return {
        "q": q,
        "k": k,
        "v": v,
        "k_mean_global": k.float().mean(dim=1).to(k.dtype).contiguous(),
        "v_scale_global": (v.float().abs().amax(dim=1) / 2.25).contiguous(),
        "rank": int(rank),
        "world_size": int(world_size),
    }


ulysses_lowp_quant_qkv_pack_fused_trace = TraceTemplate(
    op_type="comm",
    name_prefix="ulysses_lowp_quant_qkv_pack_fused",
    description=(
        "Fused amax+quantize+pack of one Ulysses shard into the destination-"
        "major uint8 A2A payload (ABI v3): INT8 Q/K on the global 32/64-token "
        "grids, FP8 V per channel, fp32 group scales, 128-byte zero tail. "
        "ALIGN-128 fast path (local_sequence % 128 == 0). The payload bytes "
        "are a frozen ABI contract validated bit-for-bit against the "
        "SageAttention golden in tests/comm/test_ulysses_lowp.py, so no "
        "torch reference is attached here."
    ),
    axes={
        "batch": Var(),
        "local_sequence": Var(description="Whole number of 128-token blocks."),
        "chunk_bytes": Var(description="payload_spec()['chunk_bytes'] per peer."),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d", value=_ULYSSES_LOWP_HEAD_DIM),
        "world_size": Const(abbrev="p", description="Ulysses group size."),
    },
    inputs={
        "q": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "k": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "v": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "k_mean_global": Tensor(["batch", "num_heads", "head_dim"]),
        "v_scale_global": Tensor(
            ["batch", "num_heads", "head_dim"],
            description="Global per-channel V |max| / 2.25 (fp32).",
        ),
        "rank": Scalar("int32"),
        "world_size": Scalar("int32"),
        "used_sequence": Scalar("int32", optional=True),
    },
    outputs={
        "payload": Tensor(
            ["world_size", "chunk_bytes"],
            dtype="uint8",
            description="Destination-major payload; row d goes to rank d.",
        ),
    },
    constraints=[
        "local_sequence % 128 == 0",
        # chunk_bytes = round_up(3*B*L*(H/P)*D + 4*B*(H/P)*(slots(L,32)+slots(L,64)), 128)
    ],
    tags=["stage:comm", "quantization:float8_e4m3fn"],
    init=_ulysses_lowp_quant_qkv_pack_fused_init,
)


ulysses_lowp_quant_qkv_pack_trace = TraceTemplate(
    op_type="comm",
    name_prefix="ulysses_lowp_quant_qkv_pack",
    description=(
        "Split-path quantize+pack of one Ulysses shard with externally "
        "finalized per-group Q/K amax (stats protocol 2: groups may straddle "
        "ranks and were max-merged / derived across the group). Same payload "
        "ABI v3 as the fused path; byte contract validated in "
        "tests/comm/test_ulysses_lowp.py."
    ),
    axes={
        "batch": Var(),
        "local_sequence": Var(),
        "q_slots": Var(),
        "k_slots": Var(),
        "chunk_bytes": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d", value=_ULYSSES_LOWP_HEAD_DIM),
        "world_size": Const(abbrev="p", description="Ulysses group size."),
    },
    inputs={
        "q": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "k": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "v": Tensor(["batch", "local_sequence", "num_heads", "head_dim"]),
        "k_mean_global": Tensor(["batch", "num_heads", "head_dim"]),
        "q_amax_final": Tensor(["batch", "num_heads", "q_slots"]),
        "k_amax_final": Tensor(["batch", "num_heads", "k_slots"]),
        "v_scale_global": Tensor(["batch", "num_heads", "head_dim"]),
        "rank": Scalar("int32"),
        "world_size": Scalar("int32"),
    },
    outputs={
        "payload": Tensor(["world_size", "chunk_bytes"], dtype="uint8"),
    },
    constraints=[
        "q_slots == (local_sequence + 62) // 32",
        "k_slots == (local_sequence + 126) // 64",
        # chunk_bytes = round_up(3*B*L*(H/P)*D + 4*B*(H/P)*(q_slots+k_slots), 128)
    ],
    tags=["stage:comm", "quantization:float8_e4m3fn"],
)


ulysses_lowp_unpack_for_sage_trace = TraceTemplate(
    op_type="comm",
    name_prefix="ulysses_lowp_unpack_for_sage",
    description=(
        "Receiver side of the low-precision Ulysses A2A: rebuild this rank's "
        "heads over the full logical sequence from the received per-source "
        "chunks -- INT8 Q/K [B,S,h,D], FP8 V in SageAttention's 16-token "
        "permuted [B,D,h,padded_S] layout, and the fp32 Q/K group scales at "
        "the consumer's width. Pure byte movement; validated bit-for-bit in "
        "tests/comm/test_ulysses_lowp.py."
    ),
    axes={
        "batch_size": Var(),
        "local_sequence": Var(),
        "logical_sequence": Var(description="local_sequence * world_size."),
        "padded_sequence": Var(description="ceil(logical_sequence / 64) * 64."),
        "q_scale_width": Var(),
        "k_scale_width": Var(),
        "chunk_bytes": Var(),
        "local_heads": Const(abbrev="h", description="Heads this rank attends."),
        "head_dim": Const(abbrev="d", value=_ULYSSES_LOWP_HEAD_DIM),
        "world_size": Const(abbrev="p", description="Ulysses group size."),
    },
    inputs={
        "recv_u8": Tensor(
            ["world_size", "chunk_bytes"],
            description="Received payload; row s came from source rank s.",
        ),
        "batch_size": Scalar("int32"),
        "local_sequence": Scalar("int32"),
        "local_heads": Scalar("int32"),
        "head_dim": Scalar("int32"),
        "world_size": Scalar("int32"),
        "aligned": Scalar("int32", optional=True),
        "scale_sequence": Scalar("int32", optional=True),
    },
    outputs={
        "q_int8": Tensor(
            ["batch_size", "logical_sequence", "local_heads", "head_dim"],
            dtype="int8",
        ),
        "k_int8": Tensor(
            ["batch_size", "logical_sequence", "local_heads", "head_dim"],
            dtype="int8",
        ),
        "v_fp8_packed": Tensor(
            ["batch_size", "head_dim", "local_heads", "padded_sequence"],
            dtype="float8_e4m3fn",
        ),
        "q_scale": Tensor(
            ["batch_size", "local_heads", "q_scale_width"], dtype="float32"
        ),
        "k_scale": Tensor(
            ["batch_size", "local_heads", "k_scale_width"], dtype="float32"
        ),
    },
    constraints=[
        "logical_sequence == local_sequence * world_size",
        "padded_sequence == (logical_sequence + 63) // 64 * 64",
        "q_scale_width == (logical_sequence + 127) // 128 * 4",
        "k_scale_width == (logical_sequence + 63) // 64",
        # chunk_bytes = round_up(3*B*L*local_heads*D + 4*B*local_heads*(slots(L,32)+slots(L,64)), 128)
    ],
    tags=["stage:comm"],
)
