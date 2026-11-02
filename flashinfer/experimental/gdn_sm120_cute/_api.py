# SPDX-License-Identifier: Apache-2.0
"""Inference forward port of FlashQLA5538c35 SM120; all GPU work is CuTe.

Autocp model and coefficients retain the reference policy, not tuned CuTe values.
Only intra-card CP and D=128. No backward/inter-card collectives/checkpoints.
Cached sequence metadata must remain immutable; pass a new tensor after edits.
"""

from dataclasses import dataclass
import weakref
import torch
from .autocp import decide
from .preprocess import precompute
from .boundary import boundary_states
from .consumer import fused_forward
from .normalize import normalize


@dataclass
class Plan:
    indices: torch.Tensor
    segments: torch.Tensor
    ranges: torch.Tensor
    use_cp: bool
    lengths: tuple
    cu_values: tuple
    index_dtype: torch.dtype
    # Derived from CPU segment metadata; no device-to-host read at launch.
    full_chunks: bool = False


_plans: dict[tuple, tuple] = {}


def build_plan(
    lengths, heads, device, index_dtype=torch.int64, auto_cp=True, force_intra_cp=False
):
    lengths = tuple(int(n) for n in lengths)
    if not lengths or min(lengths) <= 0:
        raise ValueError("positive sequence lengths required")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise ValueError("CuTe FlashQLA forward requires SM120")
    use_cp = False
    local_chunks = max((n + 31) // 32 for n in lengths)
    if auto_cp or force_intra_cp:
        use_cp, chosen = decide(
            num_chunks=[(n + 31) // 32 for n in lengths],
            num_v_heads=heads,
            P=props.multi_processor_count,
            chunk=32,
            is_train=False,
            arch="sm120",
        )
        if force_intra_cp:
            use_cp = True
        if chosen is not None:
            local_chunks = chosen
    indices = []
    segments = []
    ranges = []
    start = 0
    cp_cu = []
    for raw, n in enumerate(lengths):
        end = start + n
        indices.extend((t, end) for t in range(start, end, 32))
        part_start = len(segments)
        step = local_chunks * 32 if use_cp else n
        for left in range(start, end, step):
            right = min(end, left + step)
            segments.append((left, right, raw, int(right == end)))
            cp_cu.append(left)
        ranges.append((part_start, len(segments)))
        start = end
    cp_cu.append(start)
    return Plan(
        torch.tensor(indices, device=device, dtype=index_dtype),
        torch.tensor(segments, device=device, dtype=index_dtype),
        torch.tensor(ranges, device=device, dtype=index_dtype),
        bool(use_cp),
        lengths,
        tuple(cp_cu),
        index_dtype,
        all((right - left) % 32 == 0 for left, right, _, _ in segments),
    )


def get_plan(q, v, cu, auto_cp, force_intra_cp):
    if cu is None:
        key = (
            "dense",
            q.shape[0],
            q.shape[1],
            v.shape[2],
            str(q.device),
            auto_cp,
            force_intra_cp,
        )
        if key not in _plans:
            _plans[key] = (
                None,
                build_plan(
                    [q.shape[1]] * q.shape[0],
                    v.shape[2],
                    q.device,
                    torch.int32,
                    auto_cp and q.shape[0] == 1,
                    force_intra_cp and auto_cp and q.shape[0] == 1,
                ),
            )
        return _plans[key][1]
    try:
        version = cu._version
    except RuntimeError:
        version = None
    key = (id(cu), version, v.shape[2], str(q.device), auto_cp, force_intra_cp)
    cached = _plans.get(key)
    if cached is not None and cached[0]() is cu:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm sequence metadata before CUDA graph capture")
    values = cu.cpu().tolist()
    if values[0] != 0 or values[-1] != q.shape[1]:
        raise ValueError("cu_seqlens must span[0,T]")
    plan = build_plan(
        [e - s for s, e in zip(values, values[1:], strict=False)],
        v.shape[2],
        q.device,
        cu.dtype,
        auto_cp,
        force_intra_cp and auto_cp,
    )
    _plans[key] = (weakref.ref(cu, lambda _, key=key: _plans.pop(key, None)), plan)
    return plan


def _default_consumer_v_tile(
    plan, q, v, g, beta, initial_state, state_v_first, output_final_state, norm
) -> int:
    """Measured dense SM120 band; retain V64 outside the qualified domain."""
    if (
        plan.use_cp
        or norm
        or not state_v_first
        or not output_final_state
        or q.dtype != torch.bfloat16
        or g.dtype != torch.float32
        or beta.dtype != torch.float32
        or initial_state is None
        or initial_state.dtype != torch.float32
        or (len(plan.lengths), q.shape[2], v.shape[2]) not in ((2, 16, 48), (4, 8, 24))
    ):
        return 64
    length = plan.lengths[0]
    # V128 won at 2K/4K/8K, lost at 16K and on ragged inputs in same-process
    # CUPTI/cold-L2 comparisons. These are qualification bounds, not a claim
    # that the exact performance crossover has been located.
    if (
        len(set(plan.lengths)) != 1
        or not 2048 <= length <= 8192
        or length % 32
        or torch.cuda.get_device_properties(q.device).multi_processor_count != 156
    ):
        return 64
    return 128


@torch.inference_mode()
def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    head_first=False,
    state_v_first=False,
    auto_cp=True,
    enable_fwd_cp_cache=True,
    cp_context=None,
    force_intra_cp=False,
    *,
    plan=None,
    warmup_threshold=-10.0,
    consumer_v_tile: int | None = None,
):
    if consumer_v_tile not in (None, 64, 128):
        raise ValueError("consumer_v_tile must be None, 64 or 128")
    with torch.cuda.device(q.device):
        if head_first or cp_context is not None:
            raise ValueError(
                "head-first/inter-card/training CP contexts are not supported by this inference port"
            )
        if (
            q.ndim != 4
            or k.shape != q.shape
            or v.ndim != 4
            or q.shape[:2] != v.shape[:2]
            or q.shape[-1] != 128
            or v.shape[-1] != 128
        ):
            raise ValueError("expected Q/K[B,T,Hq,128],V[B,T,Hv,128]")
        if (
            q.dtype not in (torch.bfloat16, torch.float16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError("QKV must share BF16/FP16 dtype")
        if v.shape[2] % q.shape[2]:
            raise ValueError("value heads must be divisible by QK heads")
        if (
            g.shape != v.shape[:-1]
            or beta.shape != g.shape
            or g.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or beta.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            raise ValueError("native floating log-g/beta[B,T,Hv] required")
        tensors = [q, k, v, g, beta]
        if initial_state is not None:
            tensors.append(initial_state)
        if cu_seqlens is not None:
            if (
                q.shape[0] != 1
                or cu_seqlens.ndim != 1
                or cu_seqlens.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("varlen requiresB1 andint32/int64cu")
            tensors.append(cu_seqlens)
        if any(x.numel() >= 2**31 for x in tensors):
            raise ValueError(
                "CuTe SM120 forward currently requires int32-addressable flat tensors"
            )
        if any(x.device != q.device for x in tensors):
            raise ValueError("inputs must be on one CUDA device")
        # Same input_guard behavior as the reference; copies are included in a
        # call when inputs are strided. Contiguous benchmark inputs do not copy.
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
        if initial_state is not None:
            initial_state = initial_state.contiguous()
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous()
        if plan is None:
            plan = get_plan(q, v, cu_seqlens, auto_cp, force_intra_cp)
        if (
            sum(plan.lengths) != q.shape[0] * q.shape[1]
            or plan.segments.device != q.device
        ):
            raise ValueError("plan/input mismatch")
        if initial_state is not None and (
            initial_state.shape != (len(plan.lengths), v.shape[2], 128, 128)
            or initial_state.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            raise ValueError("invalid initial state shape/dtype")
        if consumer_v_tile is None:
            consumer_v_tile = _default_consumer_v_tile(
                plan,
                q,
                v,
                g,
                beta,
                initial_state,
                state_v_first,
                output_final_state,
                use_qk_l2norm_in_kernel,
            )
        if use_qk_l2norm_in_kernel:
            q, k = normalize(q), normalize(k)
        flat = lambda x: x.view(-1, *x.shape[2:])
        q, k, v, g, beta = map(flat, (q, k, v, g, beta))
        a, prefix = precompute(k, beta, g, plan.indices)
        initial_v_first = state_v_first
        if plan.use_cp:
            initial_state, *_ = boundary_states(
                k,
                v,
                a,
                prefix,
                beta,
                plan.segments,
                plan.ranges,
                initial_state,
                state_v_first=state_v_first,
                threshold=warmup_threshold,
            )
            initial_v_first = True
        forward = fused_forward
        forward_kwargs = {}
        if consumer_v_tile == 128:
            from .consumer_v128 import fused_forward as forward

            forward_kwargs["full_chunks"] = plan.full_chunks
        output, final = forward(
            q,
            k,
            v,
            a,
            prefix,
            beta,
            plan.segments,
            initial_state,
            scale=scale,
            state_v_first=state_v_first,
            initial_v_first=initial_v_first,
            output_final_state=output_final_state,
            num_raw_sequences=len(plan.lengths),
            **forward_kwargs,
        )
        # Input batch is known from plan. Varlen input always uses batch1.
        batch = 1 if cu_seqlens is not None else len(plan.lengths)
        return output.view(batch, -1, output.shape[1], 128), final
