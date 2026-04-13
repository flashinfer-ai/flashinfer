# tests/ep/helpers.py
#
# Shared helper functions for FlashInfer EP tests.
# These are plain functions (not pytest fixtures) and can be imported
# directly by test modules. Fixtures live in conftest.py.

import pytest
import torch

import flashinfer.ep as fep


# ─── Backend / layout constants ─────────────────────────────────────

BACKENDS = [fep.Backend.DEEP_EP, fep.Backend.NCCL_EP]
LAYOUTS = [fep.OutputLayout.FLAT_2D, fep.OutputLayout.EXPERT_MAJOR_3D]


# ─── Synthetic data generation ──────────────────────────────────────


def make_tokens(num_tokens, hidden_dim, num_experts, top_k, device="cuda",
                dtype=torch.bfloat16):
    """Generate random hidden states and routing decisions.

    In real MoE, all ranks share the same routing decisions (replicated
    router). We replicate this by broadcasting topk_idx and topk_weights
    from rank 0 so that the per-expert count invariant
    (all_reduce(local_counts) == num_tokens * top_k) holds exactly.

    Hidden states are per-rank (each rank owns its own copy of the batch,
    which is fine — dispatch scatters them).

    Returns:
        hidden:       [num_tokens, hidden_dim]  (dtype)
        topk_idx:     [num_tokens, top_k]       (int64)
        topk_weights: [num_tokens, top_k]       (float32, softmax-normalized)
    """
    import torch.distributed as dist

    # Generate hidden in BF16, then cast. torch.randn doesn't support FP8.
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        hidden = torch.randn(num_tokens, hidden_dim, device=device,
                             dtype=torch.bfloat16).to(dtype)
    else:
        hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)

    topk_idx = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=torch.float32),
        dim=-1,
    )

    # Broadcast routing decisions from rank 0 so all ranks share the same
    # expert assignments — matches real MoE where the router is replicated.
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(topk_idx, src=0)
        dist.broadcast(topk_weights, src=0)

    return hidden, topk_idx, topk_weights


def identity_expert(recv_hidden, recv_expert_counts):
    """Identity expert: output = input. For roundtrip correctness tests."""
    return recv_hidden.clone()


# ─── GPU architecture helpers ───────────────────────────────────────


def get_gpu_arch():
    """Return 'ampere', 'hopper', or 'blackwell'."""
    cap = torch.cuda.get_device_capability()
    if cap[0] == 8:
        return "ampere"
    elif cap[0] == 9:
        return "hopper"
    elif cap[0] >= 10:
        return "blackwell"
    return "unknown"


def has_fp8_support():
    """True on Hopper+ (SM90+)."""
    return torch.cuda.get_device_capability()[0] >= 9


def skip_if_not_enough_gpus(required):
    """Skip test if fewer than `required` GPUs are available."""
    available = torch.cuda.device_count()
    if available < required:
        pytest.skip(f"Need {required} GPUs, have {available}")
