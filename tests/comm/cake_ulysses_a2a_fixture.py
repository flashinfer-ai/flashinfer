"""Prepared Ulysses all-to-all fixture shared by the test and the benchmark.

The exported Cake NVLink implementation (``flashinfer.cake_ulysses_backend``)
targets Blackwell SM100/SM103; FlashInfer's portable NVLink kernel serves every
other device. The public communicator keeps its topology selection,
caller-owned outputs and NCCL fallback:

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
inputs, including whatever copies the arm's kernel issues (the portable kernel
stages and copies out; the Cake programs write the output directly). No
attention operation is timed.

``prepare(..., backend="nvlink", nvlink_kernel=...)`` selects the NVLink
kernel: ``"cake"`` (the package route; the device must have a delivered route)
or ``"portable"`` (FlashInfer's own kernel, forced by routing the communicator
away from the package for the duration of its construction).
``prepare(..., registered=True)`` takes the four outputs from the
communicator's ``allocate_output``, so on the Cake route every collective
writes the peers' registered outputs directly.
"""

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from flashinfer.comm import UlyssesCommunicator
from flashinfer.comm import ulysses as ulysses_mod

NVLINK_KERNELS = ("cake", "portable")

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
    # "cake" | "portable" for the NVLink backend, None for NCCL
    nvlink_kernel: object = None
    # whether the outputs come from the communicator's allocate_output
    registered: bool = False

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

    def check(self, run=True):
        """Check both directions independently using all-gather references
        (``run=False`` checks the outputs of calls the caller already issued)."""
        group = self.communicator.group
        rank = self.communicator.rank
        world = self.communicator.world_size
        if run:
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


def cake_route_available(world_size, dtype_names=("float16", "bfloat16", "float32")):
    """Whether the package delivers both collectives of every ``dtype`` for the current device."""
    device_index = torch.cuda.current_device()
    return all(
        ulysses_mod._cake_nvlink_route(device_index, getattr(torch, name), world_size)
        is not None
        for name in dtype_names
    )


def cake_direct_available(world_size, dtype_names=("float16", "bfloat16", "float32")):
    """Whether the package also delivers the direct route (registered outputs) of every ``dtype``."""
    device_index = torch.cuda.current_device()
    for name in dtype_names:
        dtype = getattr(torch, name)
        arch = ulysses_mod._cake_nvlink_route(device_index, dtype, world_size)
        if arch is None or not ulysses_mod._cake_nvlink_direct(
            arch, device_index, dtype, world_size
        ):
            return False
    return True


@contextlib.contextmanager
def _portable_nvlink_kernel():
    """Route communicators constructed inside away from the Cake package."""
    saved = ulysses_mod._cake_nvlink_route
    ulysses_mod._cake_nvlink_route = lambda device_index, dtype, world_size: None
    try:
        yield
    finally:
        ulysses_mod._cake_nvlink_route = saved


def prepare(
    shape,
    *,
    backend="nvlink",
    nvlink_kernel="cake",
    group=None,
    inputs=None,
    outputs=None,
    registered=False,
):
    """Allocate a case once; reuse its tensors and IPC state for every call.

    ``registered`` takes the outputs from ``allocate_output`` (``outputs`` must
    then be ``None``).
    """
    if nvlink_kernel not in NVLINK_KERNELS:
        raise ValueError(
            f"nvlink_kernel must be one of {NVLINK_KERNELS}, got {nvlink_kernel!r}"
        )
    if registered and outputs is not None:
        raise ValueError("a registered case allocates its own outputs")
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
    if outputs is None and not registered:
        outputs = tuple(
            torch.empty(dimensions, dtype=dtype, device="cuda")
            for dimensions in (global_shape, global_shape, global_shape, local_shape)
        )
    route = (
        _portable_nvlink_kernel()
        if backend == "nvlink" and nvlink_kernel == "portable"
        else contextlib.nullcontext()
    )
    with route:
        communicator = UlyssesCommunicator(
            group,
            max_bytes=batch * sequence * heads * head_dim * dtype.itemsize,
            dtype=dtype,
            backend=backend,
        )
    if communicator.backend != backend:
        communicator.close()
        raise RuntimeError(f"requested {backend}, selected {communicator.backend}")
    if backend == "nvlink" and communicator.nvlink_kernel != nvlink_kernel:
        selected = communicator.nvlink_kernel
        communicator.close()
        raise RuntimeError(
            f"requested the {nvlink_kernel} NVLink kernel, selected {selected} "
            "(the package delivers no route for this device and dtype)"
        )
    if registered:
        outputs = tuple(
            communicator.allocate_output(source, op)
            for source, op in zip(
                inputs,
                ("scatter_heads", "scatter_heads", "scatter_heads", "gather_heads"),
                strict=True,
            )
        )
    workspace = communicator.create_workspace() if backend == "nccl" else None
    return PreparedCase(
        communicator,
        inputs,
        outputs,
        workspace,
        communicator.nvlink_kernel if backend == "nvlink" else None,
        registered,
    )
