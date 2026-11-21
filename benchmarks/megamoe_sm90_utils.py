"""Deterministic global-row and global-expert fixtures for SM90 MegaMoE."""

import hashlib

import torch


def resolve_tokens(tokens, scope, world_size, capacity):
    """Resolve uniform per-rank token counts without rounding or padding."""
    if scope not in ("per-rank", "total") or world_size < 1:
        raise ValueError("invalid token scope or world size")
    if any(m < 1 or (scope == "total" and m % world_size) for m in tokens):
        raise ValueError("tokens must be positive; total tokens must divide by EP")
    resolved = [m // world_size if scope == "total" else m for m in tokens]
    if not resolved or max(resolved) > capacity:
        raise ValueError("per-rank tokens must fit the buffer capacity")
    return resolved


def tensor_digest(*tensors):
    """Hash complete tensor bytes, shapes and dtypes outside the timed region."""
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_weights(
    first_expert, local_experts, hidden, intermediate, group, seed, device
):
    """Generate the same expert's packed FP4 codes on every EP partition."""
    dimensions = ((2 * intermediate, hidden), (hidden, intermediate))
    raw = [
        (
            torch.empty(local_experts, n, k // 2, dtype=torch.uint8, device=device),
            torch.empty(local_experts, n, k // group, dtype=torch.uint8, device=device),
        )
        for n, k in dimensions
    ]
    identities = []
    for local in range(local_experts):
        expert = first_expert + local
        hashes = []
        for layer, (n, k) in enumerate(dimensions):
            generator = torch.Generator(device="cpu").manual_seed(
                seed + 2 * expert + layer
            )
            packed = torch.randint(
                256, (n, k // 2), generator=generator, dtype=torch.uint8, device="cpu"
            )
            scales = torch.randint(
                3, (n, k // group), generator=generator, dtype=torch.uint8, device="cpu"
            )
            scales += 16 if group == 16 else 122
            raw[layer][0][local].copy_(packed)
            raw[layer][1][local].copy_(scales)
            hashes.append(tensor_digest(packed, scales))
        identities.append(
            (expert, hashlib.sha256("".join(hashes).encode()).hexdigest())
        )
    return raw, identities


def make_input(
    m,
    rank,
    hidden,
    experts,
    topk,
    seed,
    device,
    routes=None,
    hot=None,
    routing="random",
):
    """Generate rows by global ID, independently of EP size or local batch size."""
    if routing not in ("random", "balanced"):
        raise ValueError("routing must be random or balanced")
    groups = 16
    while experts % groups:
        groups //= 2
    rows = []
    identities = []
    for row in range(rank * m, (rank + 1) * m):
        generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000 + row)
        x = (
            torch.randn(hidden, generator=generator, dtype=torch.float32, device="cpu")
            * 32
        ).to(torch.float8_e4m3fn)
        sf = torch.full((hidden // 128,), 1 / 128, dtype=torch.float32, device="cpu")
        if routing == "balanced" and routes is None:
            position = (row * topk + torch.arange(topk, device="cpu")) % experts
            idx = (position % groups) * (experts // groups) + position // groups
        else:
            scores = torch.rand(
                experts, generator=generator, dtype=torch.float32, device="cpu"
            )
            if hot is not None:
                scores[hot] += (
                    torch.rand(len(hot), generator=generator, device="cpu") < 0.5
                ) * 10
            idx = scores.topk(topk).indices
            if routes is not None:
                idx = routes[row % routes.shape[0]].to(device="cpu", dtype=torch.int64)
            elif hot is not None and row % 3 == 0:
                idx[-1] = -1
        weights = torch.randn(
            topk, generator=generator, dtype=torch.float32, device="cpu"
        )
        weights = weights.softmax(0) if hot is None else weights * 0.25
        rows.append((x, sf, idx, weights))
        identities.append((row, tensor_digest(x, sf, idx, weights)))
    tensors = tuple(
        torch.stack(values).to(device) for values in zip(*rows, strict=True)
    )
    return tensors, identities
