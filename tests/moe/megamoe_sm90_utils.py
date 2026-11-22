"""Generate deterministic SM90 MXFP4 transport test inputs."""

import torch


def resolve_tokens(tokens, scope, world_size, capacity):
    """Resolve per-rank counts without rounding or padding."""
    if scope not in ("per-rank", "total") or world_size < 1:
        raise ValueError("invalid token scope or world size")
    if any(m < 1 or (scope == "total" and m % world_size) for m in tokens):
        raise ValueError("tokens must be positive; total tokens must divide by EP")
    resolved = [m // world_size if scope == "total" else m for m in tokens]
    if not resolved or max(resolved) > capacity:
        raise ValueError("per-rank tokens must fit the buffer capacity")
    return resolved


def make_weights(first_expert, local_experts, hidden, intermediate, seed, device):
    """Generate MXFP4 weights from global expert IDs."""
    dimensions = ((2 * intermediate, hidden), (hidden, intermediate))
    raw = [
        (
            torch.empty(local_experts, n, k // 2, dtype=torch.uint8, device=device),
            torch.empty(local_experts, n, k // 32, dtype=torch.uint8, device=device),
        )
        for n, k in dimensions
    ]
    for local in range(local_experts):
        for layer, (n, k) in enumerate(dimensions):
            generator = torch.Generator(device="cpu").manual_seed(
                seed + 2 * (first_expert + local) + layer
            )
            packed = torch.randint(
                256, (n, k // 2), generator=generator, dtype=torch.uint8, device="cpu"
            )
            scales = torch.randint(
                3, (n, k // 32), generator=generator, dtype=torch.uint8, device="cpu"
            )
            raw[layer][0][local].copy_(packed)
            raw[layer][1][local].copy_(scales + 122)
    return raw


def make_input(m, rank, hidden, experts, topk, seed, device, hot=None):
    """Generate activation rows and routes from global token IDs."""
    rows = []
    for row in range(rank * m, (rank + 1) * m):
        generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000 + row)
        x = (
            torch.randn(hidden, generator=generator, dtype=torch.float32, device="cpu")
            * 32
        ).to(torch.float8_e4m3fn)
        sf = torch.full((hidden // 128,), 1 / 128, dtype=torch.float32, device="cpu")
        scores = torch.rand(
            experts, generator=generator, dtype=torch.float32, device="cpu"
        )
        if hot is not None:
            scores[hot] += (
                torch.rand(len(hot), generator=generator, device="cpu") < 0.5
            ) * 10
        idx = scores.topk(topk).indices
        if hot is not None and row % 3 == 0:
            idx[-1] = -1
        weights = torch.randn(
            topk, generator=generator, dtype=torch.float32, device="cpu"
        )
        weights = weights.softmax(0) if hot is None else weights * 0.25
        rows.append((x, sf, idx, weights))
    return tuple(torch.stack(values).to(device) for values in zip(*rows, strict=True))
