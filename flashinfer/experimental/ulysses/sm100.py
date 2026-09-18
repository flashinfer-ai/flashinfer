# SPDX-License-Identifier: Apache-2.0
"""Collective preflight for the pinned, narrowly admitted FA4 backend."""

import inspect

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

BASE_COMMIT = "c2006099f3ff03de187f4e1b27e756fe6df482ba"


def validate_geometry(world, local_seq, used_seqlen, full_nvlink, discard_tail_output):
    for name, number in (
        ("world", world),
        ("local_seq", local_seq),
        ("used_seqlen", used_seqlen),
    ):
        if type(number) is not int or number <= 0:
            raise ValueError(f"{name} must be a positive integer")
    physical = world * local_seq
    if (world, physical) not in ((2, 37888), (4, 37888), (8, 38912)):
        raise ValueError("outside historical SM100 H56/D128 shape allowlist")
    if not 0 < used_seqlen <= 37888:
        raise ValueError("used_seqlen exceeds the validated query extent")
    if full_nvlink is not True:
        raise ValueError("full NVLink topology must be attested before preparation")
    if discard_tail_output is not True:
        raise ValueError("padding-query output must be explicitly disposable")
    return 37888 if world == 8 else None


class PrefixRunner:
    """Freeze request metadata so hot calls cannot silently change rank contracts."""

    def __init__(self, runner, used):
        self._runner, self.used_seqlen = runner, used

    def run(self, query, key, value):
        # Inputs must have been validated identically on all ranks by the
        # framework. Rank-local hot-path failures cannot trigger a safe fallback.
        return self._runner.forward_prepared(
            query,
            key,
            value,
            used_seqlen=self.used_seqlen,
            discard_tail_output=True,
        )


def prepare(*, group, local_seq, used_seqlen, full_nvlink, discard_tail_output):
    if group is None or not dist.is_initialized():
        raise ValueError("an initialized explicit process group is required")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if world <= 0 or rank < 0:
        raise ValueError("caller must belong to the supplied group")
    # Every participant reaches the same control collective, including ranks
    # with unsupported hardware or missing optional imports. No rendezvous or
    # symmetric allocation can occur before agreement.
    error, kernel, extent = None, None, None
    try:
        extent = validate_geometry(
            world, local_seq, used_seqlen, full_nvlink, discard_tail_output
        )
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 0):
            raise ValueError("distributed FA4 requires SM100")
        if str(symm_mem.get_backend(device)).lower() != "nvshmem":
            raise ValueError("caller must configure symmetric memory with NVSHMEM")
        from . import sm100_kernel as kernel

        required = {"sched_used_q", "peer_wrap", "o_tma"}
        if not required.issubset(inspect.signature(kernel.DistArgs).parameters):
            raise ValueError("FA4 patch missing: use the documented pinned checkout")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    # Do not execute repr/equality of caller objects (including CUDA tensors)
    # outside the guarded validation before this control collective.
    signature = tuple(
        (type(x).__name__, x if type(x) in (int, bool) else None)
        for x in (world, local_seq, used_seqlen, full_nvlink, discard_tail_output)
    )
    reports = [None] * world
    dist.all_gather_object(reports, (signature, error), group=group)
    errors = [f"rank {i}: {r[1]}" for i, r in enumerate(reports) if r[1]]
    if errors:
        raise RuntimeError(
            "SM100 preflight rejected before allocation: " + "; ".join(errors)
        )
    if any(r[0] != reports[0][0] for r in reports):
        raise ValueError("SM100 request metadata differs across ranks")
    runner = kernel.DistributedFA4Runner(
        rank,
        world,
        local_seq,
        group=group,
        dtype=torch.bfloat16,
        nheads=56,
        head_dim=128,
        o_tma=True,
        mem_backend="nvshmem",
        tile_n=128,
        sched_used_q=extent,
    )
    return PrefixRunner(runner, used_seqlen)
