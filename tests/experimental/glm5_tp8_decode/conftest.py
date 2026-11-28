# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
from datetime import timedelta
import faulthandler
import os
import traceback

import pytest
import torch
import torch.distributed as dist

from flashinfer.utils import get_compute_capability

# Buffers that unfinished GPU work may still access. Releasing them can
# synchronize with the device (for example, the symmetric-memory destructor
# frees device memory) and would block behind that work, so they stay
# referenced until interpreter shutdown, where that destructor skips its CUDA
# calls.
_KEEP_ALIVE = []


class Collective:
    def __init__(self, rank, device, group):
        self.rank, self.device, self.group = rank, device, group
        # Set when a test cannot confirm that its GPU work finished. Later GPU
        # tests in this process are skipped and teardown does not wait for the
        # device.
        self.incomplete_gpu_work = None

    def gather(self, value):
        result = [None] * 8
        dist.all_gather_object(result, value, group=self.group)
        return result

    def broadcast(self, value, src=0):
        result = [value]
        dist.broadcast_object_list(result, src=src, group=self.group)
        return result[0]

    def sync(self):
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def checked(self, function):
        """Propagate local reference/validation failures to every rank together."""
        error, result = None, None
        try:
            result = function()
        except Exception:
            error = traceback.format_exc()
        errors = self.gather(error)
        assert not any(errors), "\n".join(
            f"rank {rank}:\n{error}" for rank, error in enumerate(errors) if error
        )
        return result

    def mark_incomplete(self, reason):
        """Record that GPU completion could not be confirmed on every rank."""
        self.incomplete_gpu_work = reason

    def keep_alive(self, *objects):
        """Keep buffers that unfinished GPU work may use until process exit."""
        _KEEP_ALIVE.extend(objects)


@pytest.fixture(scope="session")
def collective():
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("run with torchrun --nproc-per-node=8 on eight B300 GPUs")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    count = torch.cuda.device_count()
    if count not in (1, 8):
        pytest.skip("requires one visible GPU per process or eight visible GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # A nonidentity mapping catches code that confuses logical TP rank and CUDA ordinal.
    index = 0 if count == 1 else (local_rank + 3) % count
    device = torch.device("cuda", index)
    if get_compute_capability(device) != (10, 3):
        pytest.skip("GLM TP8 decode is qualified on B300 (SM103)")
    props = torch.cuda.get_device_properties(device)
    if (
        "B300" not in props.name
        or "GB300" in props.name
        or props.multi_processor_count != 148
    ):
        pytest.skip("requires the eight-B300 / 148-SM platform")
    torch.cuda.set_device(device)
    created_world = not dist.is_initialized()
    if created_world:
        dist.init_process_group("gloo", timeout=timedelta(seconds=240))
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=240))
    rank = dist.get_rank(group)
    control = Collective(rank, device, group)
    identities = control.gather(str(props.uuid))
    assert len(set(identities)) == 8, "each logical rank must own a different GPU"
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
        yield control
    finally:
        # After unconfirmed GPU completion, torch.cuda.synchronize could block
        # forever. Skip it, but still destroy the Gloo groups so that pytest
        # can exit with a failure status; torchrun then stops any rank that is
        # still running.
        if control.incomplete_gpu_work is None:
            control.sync()
        dist.destroy_process_group(group)
        if created_world:
            dist.destroy_process_group()
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cudnn.allow_tf32,
        ) = previous[:3]
        torch.set_num_threads(previous[3])


@pytest.fixture
def collective_watchdog(collective):
    if collective.incomplete_gpu_work is not None:
        pytest.skip(
            "device state is unknown because earlier GPU work did not complete: "
            f"{collective.incomplete_gpu_work}"
        )
    # Print every thread's stack if a test blocks, for example in a collective
    # or behind a peer-waiting kernel. The process is not terminated.
    faulthandler.dump_traceback_later(240, repeat=False, exit=False)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
