"""Four-collective Ulysses example and reproducible input portfolio.

The generated NVLink implementation targets Blackwell SM100/SM103. The public
communicator retains its existing NCCL fallback and output ownership.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator


def shapes(world_size):
    """Return this world's complete correctness and performance portfolio."""
    records = json.loads(Path(__file__).with_name("shapes.json").read_text())
    return [record for record in records if record["world_size"] == world_size]


def scatter_heads(communicator, tensor, *, out=None, workspace=None):
    """Scatter heads through the existing FlashInfer communicator API."""
    return communicator.scatter_heads(tensor, out=out, workspace=workspace)


def gather_heads(communicator, tensor, *, out=None, workspace=None):
    """Gather heads through the existing FlashInfer communicator API."""
    return communicator.gather_heads(tensor, out=out, workspace=workspace)


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
        max_elems=batch * sequence * heads * head_dim,
        dtype=dtype,
        backend=backend,
    )
    if communicator.backend != backend:
        communicator.close()
        raise RuntimeError(f"requested {backend}, selected {communicator.backend}")
    workspace = communicator.create_workspace() if backend == "nccl" else None
    return PreparedCase(communicator, inputs, outputs, workspace)
