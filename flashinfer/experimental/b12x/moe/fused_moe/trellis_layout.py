"""Lossless trellis slot assembly and checkpoint-global sign coordinates.

These load-time tensor operations have no kernel/compiler dependency. CPU
execution is useful for format validation; serving prepares onto CUDA before
any execution plan or graph is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .source import TrellisExtent


@dataclass(frozen=True, kw_only=True)
class TrellisStaging:
    """Limits on a temporary uniform-codeword batch, excluding final weights.

    Transfers contain at most ``max_experts`` experts and ``max_bytes`` bytes.
    A single expert larger than the byte limit is rejected before allocation.
    Permuted copy operands can add up to one projection (one third of a batch)
    to this bound. Scale/sign tables and CUDA allocator overhead are separate.
    """

    max_experts: int = 64
    max_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_experts", "max_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"TrellisStaging.{name} must be a positive integer")

    def batch_size(self, *, slots: int, hidden_size: int, bits: int) -> int:
        expert_bytes = slots * 3 * (hidden_size // 16) * 64 * bits
        if expert_bytes <= 0:
            raise ValueError("trellis staging geometry must be positive")
        count = min(self.max_experts, self.max_bytes // expert_bytes)
        if count == 0:
            raise ValueError(
                f"one trellis expert requires {expert_bytes} staging bytes; "
                f"max_bytes={self.max_bytes} is insufficient"
            )
        return count


def assemble_uniform_slots(
    codes: torch.Tensor,
    *,
    num_experts: int,
    hidden_size: int,
    bits: int,
    device: torch.device | str,
    staging: TrellisStaging = TrellisStaging(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble FC1/FC2 int16 carriers without decoding or rounding weights.

    Input rows are slot-major expert bundles, each containing gate, up, down.
    The uint8 payload may have a larger physical row stride, but no logical
    padding columns. CPU payloads are copied in bounded expert batches.
    """

    device = torch.device(device)
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[0] == 0:
        raise ValueError("codes must be nonempty uint8 [slots,payload_bytes]")
    if hidden_size <= 0 or hidden_size % 32 or num_experts <= 0:
        raise ValueError("trellis assembly requires E > 0 and H divisible by 32")
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError("trellis assembly supports symmetric rates K2 through K6")
    if codes.device.type not in ("cpu", "cuda"):
        raise ValueError("trellis payload must reside on CPU or CUDA")
    if codes.device.type == "cuda" and codes.device != device:
        raise ValueError("CUDA trellis payload must reside on its destination device")
    slots = codes.shape[0]
    hidden_tiles = hidden_size // 16
    section = hidden_tiles * 64 * bits
    payload = num_experts * 3 * section
    if codes.shape[1] != payload or codes.stride(1) != 1:
        raise ValueError(f"codes must have {payload} contiguous bytes per logical row")
    batch_size = staging.batch_size(slots=slots, hidden_size=hidden_size, bits=bits)
    w13 = torch.empty(
        (2, num_experts, hidden_tiles, 2 * slots, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.empty(
        (num_experts, 2 * slots, hidden_tiles, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    bundles = codes.unflatten(1, (num_experts, 3 * section))
    for first in range(0, num_experts, batch_size):
        count = min(batch_size, num_experts - first)
        chunk = (
            bundles[:, first : first + count]
            .contiguous()
            .to(device=device)
            .view(torch.int16)
            .reshape(slots, count, 3, 2, hidden_tiles, 16 * bits)
        )
        for matrix in range(2):
            w13[matrix, first : first + count].copy_(
                chunk[:, :, matrix]
                .permute(1, 3, 0, 2, 4)
                .reshape(count, hidden_tiles, 2 * slots, 16 * bits)
            )
        w2[first : first + count].copy_(
            chunk[:, :, 2]
            .permute(1, 0, 2, 3, 4)
            .reshape(count, 2 * slots, hidden_tiles, 16 * bits)
        )
        # Do not retain the previous staging allocation during the next H2D copy.
        del chunk
    return w13, w2


def intermediate_hadamard_signs(
    length: int, *, sign_pattern: int, axis: int
) -> torch.Tensor:
    """Deterministic CPU sign sequence used by the intermediate transform."""

    if length <= 0 or not 0 <= sign_pattern < 8:
        raise ValueError("sign generation requires length > 0 and pattern in 0..7")
    if sign_pattern == 0:
        return torch.ones(length, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (0x6A09E667F3BCC909 * sign_pattern + 0xBB67AE8584CAA73B * axis)
        & ((1 << 63) - 1)
    )
    return torch.randint(0, 2, (length,), generator=generator).mul_(2).sub_(1).float()


def append_intermediate_signs(
    values: torch.Tensor,
    sign_patterns: torch.Tensor,
    *,
    extent: TrellisExtent,
) -> torch.Tensor:
    """Append [pre 2I | post I] signs sliced in checkpoint-global coordinates."""

    local = extent.intermediate_size
    if (
        values.ndim != 2
        or values.shape[1] != 3 * local
        or values.dtype != torch.float16
    ):
        raise ValueError("intermediate values must be fp16 [experts,3*I_local]")
    if sign_patterns.dtype != torch.uint8 or sign_patterns.shape != (values.shape[0],):
        raise ValueError("expert_sign_patterns must be uint8 [experts]")
    patterns = sign_patterns.detach().cpu()
    if bool(torch.any(patterns > 7)):
        raise ValueError("expert_sign_patterns values must be in 0..7")
    signs = torch.empty((values.shape[0], 3 * local), dtype=torch.float16)
    begin = 32 * extent.first_slot
    for pattern in sorted(set(patterns.tolist())):
        rows = torch.nonzero(patterns == pattern, as_tuple=False).flatten()
        pre = intermediate_hadamard_signs(
            2 * extent.global_intermediate_size, sign_pattern=pattern, axis=1
        )[2 * begin : 2 * (begin + local)]
        post = intermediate_hadamard_signs(
            extent.global_intermediate_size, sign_pattern=pattern, axis=2
        )[begin : begin + local]
        signs.index_copy_(
            0, rows, torch.cat((pre, post)).half().expand(rows.numel(), -1)
        )
    return torch.cat((values, signs.to(device=values.device)), dim=1).contiguous()
