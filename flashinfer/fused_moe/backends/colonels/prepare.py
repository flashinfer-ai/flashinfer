import torch

from ...api import SwiGLU
from .activations import activation_kind


def interleave_gate_up(tensor, rep):
    """Convert canonical [up, gate] rows to [gate_0, up_0, gate_1, up_1, ...]."""
    e, n, k = tensor.shape
    up, gate = tensor.chunk(2, dim=1)
    return (
        torch.stack(
            (
                gate.reshape(e, n // (2 * rep), rep, k),
                up.reshape(e, n // (2 * rep), rep, k),
            ),
            dim=2,
        )
        .reshape(e, n, k)
        .contiguous()
    )


def quantize_weights(weight):
    e, n, k = weight.shape
    blocks = weight.float().reshape(e, n // 128, 128, k // 128, 128)
    scales = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-4) / 448.0
    q = (blocks / scales[:, :, None, :, None]).clamp(-448, 448)
    return q.reshape(e, n, k).to(torch.float8_e4m3fn), scales.contiguous()


def prepare_weights(
    w1,
    w2,
    *,
    num_local_experts,
    hidden_size,
    intermediate_size,
    activation=None,
    device=None,
    fp8=False,
    w1_scale=None,
    w2_scale=None,
    expert_map=None,
):
    activation = SwiGLU() if activation is None else activation
    activation_kind(activation)
    if fp8 and activation != SwiGLU():
        raise NotImplementedError("Colonels FP8 supports SwiGLU")
    if min(num_local_experts, hidden_size, intermediate_size) <= 0:
        raise ValueError("Expert and weight dimensions must be positive")
    rows = intermediate_size * (2 if activation.is_gated else 1)
    if hidden_size % (256 if fp8 else 128) or rows % (256 if fp8 else 16):
        raise ValueError(
            "Colonels weights do not meet hidden/intermediate alignment requirements"
        )
    expected = (
        (num_local_experts, rows, hidden_size),
        (num_local_experts, hidden_size, intermediate_size),
    )
    if tuple(w1.shape) != expected[0] or tuple(w2.shape) != expected[1]:
        raise ValueError(f"Expected canonical weight shapes {expected}")
    if w1.device != w2.device or w1.dtype != w2.dtype:
        raise ValueError("Weights must have the same dtype and device")
    if device is not None:
        w1, w2 = w1.to(device=device), w2.to(device=device)
    if fp8:
        if w1.dtype == torch.bfloat16:
            if w1_scale is not None or w2_scale is not None:
                raise ValueError("BF16 source weights must not have FP8 scales")
            w1, w1_scale = quantize_weights(w1)
            w2, w2_scale = quantize_weights(w2)
        elif w1.dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 preparation requires BF16 or E4M3 weights")
        for scale, weight in ((w1_scale, w1), (w2_scale, w2)):
            if (
                scale is None
                or scale.dtype != torch.float32
                or tuple(scale.shape)
                != (weight.shape[0], weight.shape[1] // 128, weight.shape[2] // 128)
            ):
                raise ValueError(
                    "Prequantized weights require FP32 128x128 block scales"
                )
        w1_scale = interleave_gate_up(w1_scale.to(w1.device), 1)
        w2_scale = w2_scale.to(w2.device).contiguous()
    elif w1.dtype != torch.bfloat16 or w1_scale is not None or w2_scale is not None:
        raise ValueError("BF16 preparation requires BF16 weights without scales")
    mapping = {}
    if expert_map is not None:
        if (
            expert_map.ndim != 1
            or expert_map.dtype != torch.int32
            or not expert_map.numel()
        ):
            raise ValueError(
                "expert_map must be a nonempty int32 global-to-local vector"
            )
        # Preparation runs at model load, so value validation may synchronize.
        cpu_map = expert_map.cpu()
        local_ids = cpu_map[cpu_map >= 0]
        if (cpu_map < -1).any() or (local_ids >= num_local_experts).any():
            raise ValueError("expert_map entries must be -1 or local weight indices")
        if local_ids.unique().numel() != local_ids.numel():
            raise ValueError(
                "expert_map must map distinct experts to distinct local rows"
            )
        mapping = {
            "expert_map": expert_map.to(w1.device).contiguous(),
            "expert_map_max": int(local_ids.max()) if local_ids.numel() else -1,
        }
    return {
        **mapping,
        "w1": interleave_gate_up(w1, 8) if activation == SwiGLU() else w1.contiguous(),
        "w2": w2.contiguous(),
        "activation": activation,
        **({"w1_scale": w1_scale, "w2_scale": w2_scale} if fp8 else {}),
    }


def prepare_activations(x):
    """Canonical DeepSeek FP8 payload and transposed FP32 K128 scales."""
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("Expected BF16 [tokens, hidden] with hidden divisible by 128")
    blocks = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
    q = (blocks / scales.unsqueeze(-1)).clamp(-448, 448).reshape(x.shape)
    return q.to(torch.float8_e4m3fn), scales.T.contiguous()
