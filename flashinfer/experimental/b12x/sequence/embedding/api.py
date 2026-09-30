"""Exact unquantized embedding lookup without output or replay allocation."""
import torch

from ..._lib.gating import default_is_supported
from . import META
from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from ._preparation import plan, query_from_call
from ._tuning import EmbeddingQuery


def _weight_width(weight):
    return weight.shape[1] * 32 if weight.dtype == torch.uint8 else weight.shape[1]


def _check_weight(weight):
    if weight.is_cuda and weight.dtype == torch.uint8:
        if weight.ndim != 3 or weight.shape[1] <= 0 or weight.shape[2] != 34 or not weight.is_contiguous():
            raise ValueError("Q8_0 embedding requires contiguous uint8 [table_rows,K/32,34]")
        return
    if not weight.is_cuda or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("embedding weight must be CUDA BF16 or FP32")
    if weight.ndim != 2 or weight.shape[1] <= 0:
        raise ValueError("embedding weight must be [table_rows, positive width]")
    if weight.stride(1) != 1 or weight.stride(0) < weight.shape[1]:
        raise ValueError("embedding weight must have contiguous nonoverlapping rows")


def _check_tensors(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                   num_rows: torch.Tensor | None = None) -> None:
    _check_weight(weight)
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
        raise TypeError("embedding IDs must be contiguous Int32 or Int64")
    if ids.device != weight.device or out.device != weight.device:
        raise ValueError("embedding tensors must share the CUDA device")
    if out.shape != (*ids.shape, _weight_width(weight)) or out.dtype != (torch.bfloat16 if weight.dtype == torch.uint8 else weight.dtype):
        raise ValueError("embedding output must have ids.shape + (width,) and weight dtype")
    if not out.is_contiguous() or ids.numel() >= 2**31:
        raise ValueError("embedding requires contiguous output and an Int32 launch count")
    if torch._C._overlaps(out, weight) or torch._C._overlaps(out, ids):
        raise ValueError("embedding output must not alias inputs")
    if num_rows is not None:
        if (num_rows.device != weight.device or num_rows.dtype != torch.int32
                or num_rows.numel() != 1 or not num_rows.is_contiguous()):
            raise ValueError("num_rows must be a CUDA Int32 scalar on the weight device")
        if ids.numel() == 0:
            raise ValueError("device-count lookup requires positive output capacity")
        if torch._C._overlaps(out, num_rows):
            raise ValueError("embedding output must not alias num_rows")


@torch.library.custom_op("b12x::embedding_out", mutates_args=("out",))
def _embedding_out(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                   plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    require_prepared(plan_from_handle(plan_handle), "sequence.embedding", weight.device).run(
        weight, ids, out=out, num_rows=num_rows,
    )


@_embedding_out.register_fake
def _embedding_out_fake(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                        plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    return None


def run(weight, ids, *, out, plan: Plan, num_rows=None):
    """Copy exact rows into caller storage through the prepared embedding kernel.

    IDs and the optional device live count remain dynamic during graph replay.
    Only live rows are written. Invalid live IDs or counts trap on the device.
    """
    _embedding_out(weight, ids, out, plan.handle, num_rows)
    return out


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


def clear_caches():
    from ._kernel import compile_embedding
    compile_embedding.cache_clear()
