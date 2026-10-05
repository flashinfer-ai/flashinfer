"""Prepared Ulysses all-to-all fixture shared by the test and the benchmark.

The generated NVLink implementation targets Blackwell SM100/SM103. The public
communicator keeps its topology selection, caller-owned outputs and NCCL
fallback:

.. code-block:: python

    from flashinfer.comm import UlyssesCommunicator

    with UlyssesCommunicator(group, max_bytes=q.nbytes, dtype=q.dtype) as comm:
        q_global = comm.scatter_heads(q)
        k_global = comm.scatter_heads(k)
        v_global = comm.scatter_heads(v)
        output = comm.gather_heads(attention_output)

``SHAPES`` lists 36 small correctness rows and four large BF16 performance rows
covering worlds 2/4/6/8, fp16/bf16/fp32, batch dimensions and scalar tails. A
prepared case allocates its inputs, outputs and workspace once; every measured
call performs three head scatters and one gather with independently supplied
inputs, including the four staging-to-output copies. No attention operation is
timed.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from flashinfer.comm import UlyssesCommunicator

SHAPES: list[dict[str, Any]] = [
    {
        "label": "correctness_w2_fp16_aligned",
        "world_size": 2,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w2_fp16_batch",
        "world_size": 2,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w2_fp16_scalar",
        "world_size": 2,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w2_bf16_aligned",
        "world_size": 2,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w2_bf16_batch",
        "world_size": 2,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w2_bf16_scalar",
        "world_size": 2,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w2_fp32_aligned",
        "world_size": 2,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w2_fp32_batch",
        "world_size": 2,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w2_fp32_scalar",
        "world_size": 2,
        "B": 1,
        "S_local": 5,
        "H": 22,
        "D": 3,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w4_fp16_aligned",
        "world_size": 4,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w4_fp16_batch",
        "world_size": 4,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w4_fp16_scalar",
        "world_size": 4,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w4_bf16_aligned",
        "world_size": 4,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w4_bf16_batch",
        "world_size": 4,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w4_bf16_scalar",
        "world_size": 4,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w4_fp32_aligned",
        "world_size": 4,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w4_fp32_batch",
        "world_size": 4,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w4_fp32_scalar",
        "world_size": 4,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w6_fp16_aligned",
        "world_size": 6,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w6_fp16_batch",
        "world_size": 6,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w6_fp16_scalar",
        "world_size": 6,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w6_bf16_aligned",
        "world_size": 6,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w6_bf16_batch",
        "world_size": 6,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w6_bf16_scalar",
        "world_size": 6,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w6_fp32_aligned",
        "world_size": 6,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w6_fp32_batch",
        "world_size": 6,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w6_fp32_scalar",
        "world_size": 6,
        "B": 1,
        "S_local": 5,
        "H": 30,
        "D": 3,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w8_fp16_aligned",
        "world_size": 8,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w8_fp16_batch",
        "world_size": 8,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w8_fp16_scalar",
        "world_size": 8,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "float16",
        "performance": False,
    },
    {
        "label": "correctness_w8_bf16_aligned",
        "world_size": 8,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w8_bf16_batch",
        "world_size": 8,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w8_bf16_scalar",
        "world_size": 8,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "bfloat16",
        "performance": False,
    },
    {
        "label": "correctness_w8_fp32_aligned",
        "world_size": 8,
        "B": 1,
        "S_local": 8,
        "H": 24,
        "D": 128,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w8_fp32_batch",
        "world_size": 8,
        "B": 2,
        "S_local": 16,
        "H": 24,
        "D": 64,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "correctness_w8_fp32_scalar",
        "world_size": 8,
        "B": 1,
        "S_local": 5,
        "H": 24,
        "D": 3,
        "dtype": "float32",
        "performance": False,
    },
    {
        "label": "perf_w2_bf16_wan",
        "world_size": 2,
        "B": 1,
        "S_local": 16380,
        "H": 40,
        "D": 128,
        "dtype": "bfloat16",
        "performance": True,
    },
    {
        "label": "perf_w4_bf16_wan",
        "world_size": 4,
        "B": 1,
        "S_local": 8190,
        "H": 40,
        "D": 128,
        "dtype": "bfloat16",
        "performance": True,
    },
    {
        "label": "perf_w6_bf16_divisible",
        "world_size": 6,
        "B": 1,
        "S_local": 5460,
        "H": 48,
        "D": 128,
        "dtype": "bfloat16",
        "performance": True,
    },
    {
        "label": "perf_w8_bf16_wan",
        "world_size": 8,
        "B": 1,
        "S_local": 4095,
        "H": 40,
        "D": 128,
        "dtype": "bfloat16",
        "performance": True,
    },
]


def shapes(world_size):
    """Return this world's complete correctness and performance portfolio."""
    return [record for record in SHAPES if record["world_size"] == world_size]


@dataclass
class PreparedCase:
    communicator: UlyssesCommunicator
    inputs: tuple
    outputs: tuple
    workspace: object

    def run(self):
        """Three scatters and an independent gather, including copy-out."""
        for source, destination in zip(self.inputs[:3], self.outputs[:3], strict=False):
            self.communicator.scatter_heads(
                source, out=destination, workspace=self.workspace
            )
        self.communicator.gather_heads(
            self.inputs[3], out=self.outputs[3], workspace=self.workspace
        )
        return self.outputs

    def check(self):
        """Check both directions independently using all-gather references."""
        group = self.communicator.group
        rank = self.communicator.rank
        world = self.communicator.world_size
        self.run()
        for index, (source, actual) in enumerate(
            zip(self.inputs, self.outputs, strict=False)
        ):
            peers = [torch.empty_like(source) for _ in range(world)]
            dist.all_gather(peers, source, group=group)
            if index < 3:
                local_heads = source.shape[2] // world
                expected = torch.cat(
                    [
                        peer[:, :, rank * local_heads : (rank + 1) * local_heads, :]
                        for peer in peers
                    ],
                    dim=1,
                )
            else:
                local_sequence = source.shape[1] // world
                expected = torch.cat(
                    [
                        peer[
                            :,
                            rank * local_sequence : (rank + 1) * local_sequence,
                            :,
                            :,
                        ]
                        for peer in peers
                    ],
                    dim=2,
                )
            if not torch.equal(actual, expected):
                direction = "scatter" if index < 3 else "gather"
                raise AssertionError(
                    f"{direction} mismatch on rank {rank}, operand {index}"
                )

    def close(self):
        self.communicator.close()


def prepare(shape, *, backend="nvlink", group=None, inputs=None, outputs=None):
    """Allocate a case once; reuse its tensors and IPC state for every call."""
    group = dist.group.WORLD if group is None else group
    world = dist.get_world_size(group=group)
    if world != shape["world_size"]:
        raise ValueError("the shape and process group must have the same world size")
    rank = dist.get_rank(group=group)
    batch, sequence, heads, head_dim = (
        shape[key] for key in ("B", "S_local", "H", "D")
    )
    dtype = getattr(torch, shape["dtype"])
    local_shape = (batch, sequence, heads, head_dim)
    global_shape = (batch, sequence * world, heads // world, head_dim)
    if inputs is None:
        generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
        inputs = tuple(
            torch.randn(dimensions, dtype=dtype, device="cuda", generator=generator)
            for dimensions in (local_shape, local_shape, local_shape, global_shape)
        )
    if outputs is None:
        outputs = tuple(
            torch.empty(dimensions, dtype=dtype, device="cuda")
            for dimensions in (global_shape, global_shape, global_shape, local_shape)
        )
    communicator = UlyssesCommunicator(
        group,
        max_bytes=batch * sequence * heads * head_dim * dtype.itemsize,
        dtype=dtype,
        backend=backend,
    )
    if communicator.backend != backend:
        communicator.close()
        raise RuntimeError(f"requested {backend}, selected {communicator.backend}")
    workspace = communicator.create_workspace() if backend == "nccl" else None
    return PreparedCase(communicator, inputs, outputs, workspace)
