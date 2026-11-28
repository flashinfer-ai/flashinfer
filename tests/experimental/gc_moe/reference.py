# SPDX-License-Identifier: Apache-2.0
"""Pure-Torch references and staging helpers for the GC MoE tests.

Every case stages all-ones scales, so all three architectures decode the same
way: MXFP4 block scales of ``2**0`` and an identity activation scale. That
keeps one decode table and one reference forward valid across SM90, SM100 and
SM120 while the checkpoint *layouts* differ per architecture.
"""

import torch

GEOMETRY = {
    "num_experts": 4,
    "hidden": 512,
    "intermediate_hidden": 256,
    "num_topk": 2,
    "capacity": 64,
}

# E2M1 magnitudes; the sign bit selects the negative half of the table.
FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# Four packed UE8M0 bytes, each 0x7F = 2**0.
UE8M0_ONE_WORD = 0x7F7F7F7F

BACKEND_BY_CAPABILITY = {
    (9, 0): "sm90",
    (10, 0): "sm100",
    (10, 3): "sm100",
    (12, 0): "sm120",
}


def decode_fp4_payload(payload, k):
    """``[..., k // 2]`` int8 MXFP4 -> ``[..., k]`` float32; even K in the low nibble."""
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=payload.device)
    raw = payload.view(torch.uint8).to(torch.int32)
    out = torch.empty(
        payload.shape[:-1] + (k,), dtype=torch.float32, device=payload.device
    )
    out[..., 0::2] = table[raw & 0x0F]
    out[..., 1::2] = table[(raw >> 4) & 0x0F]
    return out


def checkpoint_weights(backend, num_experts, hidden, intermediate, device):
    """Random MXFP4 payloads plus the backend's all-ones checkpoint scales."""
    generator = torch.Generator(device="cpu").manual_seed(20261113)

    def payload(rows, k):
        return torch.randint(
            -128,
            128,
            (num_experts, rows, k // 2),
            dtype=torch.int8,
            generator=generator,
        ).to(device)

    def scales(rows, k):
        if backend == "sm90":
            # FP32 powers of two; 1.0 keeps the Humming exponent clipping inert.
            return torch.ones(
                (num_experts, rows, k // 32), dtype=torch.float32, device=device
            )
        if backend == "sm100":
            # SM100 consumes MN-major packed scales: [E, N, K/128] with stride(-2) == 1.
            return torch.full(
                (num_experts, k // 128, rows),
                UE8M0_ONE_WORD,
                dtype=torch.int32,
                device=device,
            ).transpose(-1, -2)
        return torch.full(
            (num_experts, rows, k // 128),
            UE8M0_ONE_WORD,
            dtype=torch.int32,
            device=device,
        )

    return (
        (payload(2 * intermediate, hidden), scales(2 * intermediate, hidden)),
        (payload(hidden, intermediate), scales(hidden, intermediate)),
    )


def stage_tokens(num_tokens, hidden, num_experts, num_topk, device, seed=7):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(num_tokens, hidden, generator=generator) * 0.5).to(device)
    x = x.to(torch.float8_e4m3fn)
    scores = torch.randn(num_tokens, num_experts, generator=generator)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(device)
    return x, topk_idx.to(torch.int64).to(device), topk_weights


def stage(workspace, x, topk_idx, topk_weights):
    tokens = x.shape[0]
    workspace.x[:tokens].copy_(x)
    if workspace.x_sf.dtype == torch.float32:
        workspace.x_sf[:tokens].fill_(1.0)
    else:
        workspace.x_sf[:tokens].fill_(UE8M0_ONE_WORD)
    workspace.topk_idx[:tokens].copy_(topk_idx)
    workspace.topk_weights[:tokens].copy_(topk_weights)


def reference_forward(x_fp8, topk_idx, topk_weights, w1, w2, intermediate):
    """Exact-arithmetic SwiGLU MoE reference over the decoded weights."""
    tokens, hidden = x_fp8.shape
    x = x_fp8.float()
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x.device)
    for token in range(tokens):
        acc = torch.zeros(hidden, dtype=torch.float32, device=x.device)
        for slot in range(topk_idx.shape[1]):
            expert = int(topk_idx[token, slot])
            linear = x[token] @ w1[expert].T
            gate, up = linear[:intermediate], linear[intermediate:]
            mid = gate * torch.sigmoid(gate) * up
            acc = acc + float(topk_weights[token, slot]) * (mid @ w2[expert].T)
        out[token] = acc
    return out


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b / denominator).item())


def relative_l2(a, b):
    """Normalized L2 distance; alongside cosine it also bounds magnitude errors."""
    a = a.flatten().float()
    b = b.flatten().float()
    scale = float(b.norm())
    if scale == 0.0:
        return 0.0 if float(a.norm()) == 0.0 else float("inf")
    return float((a - b).norm()) / scale


def run_public_case(backend, group, num_tokens=8, geometry=GEOMETRY):
    """Drive the public API end to end and return ``(y, expected)``."""
    import flashinfer

    device = torch.device("cuda", torch.cuda.current_device())
    experts = geometry["num_experts"]
    hidden = geometry["hidden"]
    intermediate = geometry["intermediate_hidden"]
    topk = geometry["num_topk"]
    l1_checkpoint, l2_checkpoint = checkpoint_weights(
        backend, experts, hidden, intermediate, device
    )
    w1 = decode_fp4_payload(l1_checkpoint[0], hidden)
    w2 = decode_fp4_payload(l2_checkpoint[0], intermediate)
    x, topk_idx, topk_weights = stage_tokens(num_tokens, hidden, experts, topk, device)
    with flashinfer.GCMoEWorkspace(
        group, experts, geometry["capacity"], topk, hidden, intermediate
    ) as workspace:
        l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(
            l1_checkpoint, l2_checkpoint
        )
        stage(workspace, x, topk_idx, topk_weights)
        y = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
        torch.cuda.synchronize()
        expected = reference_forward(x, topk_idx, topk_weights, w1, w2, intermediate)
        y = y.clone()
    return y, expected
