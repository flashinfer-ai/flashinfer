from concurrent.futures import ThreadPoolExecutor
from functools import cache
import os

import torch

from ...cute_dsl.moe_utils import allocate_moe_sort_buffers, moe_sort
from ....jit.colonels import gen_colonels_module


def precompile_kernels(tactics, activation, arch):
    """Build unique module families in parallel before probing launch support."""
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        # Leave AOT loading and missing-cache errors to build_and_load().
        return
    specs = {}
    for tactic in tactics:
        spec = gen_colonels_module(
            tactic.variant, tactic.compilation_parameters(), activation, arch
        )
        if not spec.is_aot:
            specs[spec.name] = spec
    if not specs:
        return
    workers = os.environ.get(
        "FLASHINFER_JIT_PREBUILD_MAX_JOBS", os.environ.get("MAX_JOBS", "")
    )
    max_workers = int(workers) if workers.isdigit() and int(workers) > 0 else None

    def build(spec):
        # Keep the usual module-local Ninja logs so warm caches stay warm.
        spec.build(need_lock=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(build, specs.values()))


@cache
def load_kernel(tactic, activation, arch):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm the Colonels tactic before CUDA graph capture")
    return gen_colonels_module(
        tactic.variant, tactic.compilation_parameters(), activation, arch
    ).build_and_load()


class Workspace:
    def __init__(
        self,
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        block_sizes,
        device,
        num_sms,
        needs_scratch,
        scratch=None,
    ):
        self.routing = {
            bm: allocate_moe_sort_buffers(
                num_tokens, num_experts, top_k, tile_tokens_dim=bm, device=device
            )
            for bm in block_sizes
        }
        self.scratch = scratch
        if self.scratch is None and needs_scratch:
            self.scratch = torch.empty(
                num_sms * 128 * intermediate_size, dtype=torch.bfloat16, device=device
            )

    def sort(self, ids, weights, num_experts, block_m):
        with torch.cuda.device(ids.device):
            return self._sort_on_device(ids, weights, num_experts, block_m)

    def _sort_on_device(self, ids, weights, num_experts, block_m):
        buffers = self.routing[block_m]
        result = moe_sort(
            ids,
            weights,
            num_experts,
            ids.shape[1],
            tile_tokens_dim=block_m,
            padding=ids.numel(),
            **buffers,
        )
        return result[3], result[0], result[4]


def launch(
    tactic,
    activation,
    x,
    w1,
    w2,
    routing,
    weights,
    out,
    *,
    scratch=None,
    x_scale=None,
    w1_scale=None,
    w2_scale=None,
    scaling_factor=1.0,
    clamp=0.5,
    experts_done=None,
    zero_output=True,
):
    """Launch on prepared tensors; retain buffers until this stream completes.

    ``routing`` is the aligned (sorted ids, expert ids, padded count) triplet;
    GEMV uses top-k ids in its first entry. Completion flags are caller-owned,
    including initialization and coordination with an overlapped consumer.
    """
    device = x.device
    tensors = (
        x,
        w1,
        w2,
        *routing,
        weights,
        out,
        scratch,
        x_scale,
        w1_scale,
        w2_scale,
        experts_done,
    )
    for tensor in tensors:
        if tensor is not None and (
            tensor.device != device or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Colonels launch tensors must be contiguous and on the same device"
            )
    if device.type != "cuda":
        raise ValueError("Colonels requires CUDA")
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("Expected matrix activations and expert weight tensors")
    if out.shape != x.shape or out.dtype != torch.bfloat16:
        raise ValueError("Colonels output must be BF16 with the input shape")
    if not tactic.supports_shape(x.shape[1], w2.shape[2], activation == 0):
        return False
    if (
        weights.dtype != torch.float32
        or weights.ndim != 2
        or weights.shape[0] != x.shape[0]
    ):
        raise ValueError("Routing weights must be FP32 [tokens, top_k]")
    reads = (x, w1, w2, *routing, weights, x_scale, w1_scale, w2_scale)
    writes = (out, scratch, experts_done)
    for index, target in enumerate(writes):
        if target is None or not target.numel():
            continue
        start = target.data_ptr()
        end = start + target.numel() * target.element_size()
        for tensor in reads + writes[index + 1 :]:
            if (
                tensor is not None
                and tensor.numel()
                and (
                    start < tensor.data_ptr() + tensor.numel() * tensor.element_size()
                    and tensor.data_ptr() < end
                )
            ):
                raise ValueError(
                    "Output and mutable workspace must not alias other launch buffers"
                )
    for tensor in (x, w1, w2, out, scratch):
        if tensor is not None and tensor.numel() and tensor.data_ptr() % 16:
            raise ValueError("Colonels operands must be 16-byte aligned")
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        module = load_kernel(tactic, activation, major * 10 + minor)
        if zero_output:
            out.zero_()
        if x.shape[0] == 0:
            return True
        return module.run(
            x,
            w1,
            w2,
            *routing,
            weights,
            out,
            x_scale,
            w1_scale,
            w2_scale,
            scratch,
            experts_done,
            weights.shape[1],
            tactic.stages,
            scaling_factor,
            clamp,
            tactic.cluster_fallback,
            torch.cuda.current_stream(device).cuda_stream,
        )
