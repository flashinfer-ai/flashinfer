# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Saved forward tensors for the batch-invariant MegaMoE kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class MoEEpMegaTrainingOutputs:
    """Workspace-owned tensors written by ``save_for_backward=True``.

    ``fc1_preact`` is BF16 ``[P, 2 * intermediate]`` in gate-then-up order,
    after FC1 dequantization and before activation/clamping. ``expert_ids``
    gives the local expert index of each receiver-pool row; ``-1`` marks padding.
    Padding values in ``fc1_preact`` are undefined. Only valid rows of the native
    input and handoff buffers are defined. Their physical
    layouts and route metadata are described by ``layout`` and the metadata
    keys, rather than being interchangeable across kernel providers.

    ``route_metadata_format`` describes the packed Int64 ``token_src_metadata``
    field. ``token32_slot16_rank16`` stores token, top-k slot and source rank in
    bits 0:31, 32:47 and 48:63. ``token16_slot8_rank8`` stores them in bits 0:15,
    16:23 and 24:31, with destination order and route count minus one in bits
    32:39 and 40:47.

    The buffers are borrowed from one workspace and overwritten by its next
    forward. Keep that workspace alive through backward, and use independent
    explicit workspaces for forwards whose saved states must coexist. Reacquire
    this bundle after automatic plan selection replaces workspace storage.
    Token-axis quantized WGrad operands are not produced by this interface.
    """

    fc1_preact: "torch.Tensor"
    expert_ids: "torch.Tensor"
    fc1_input: "torch.Tensor"
    fc1_input_scale: "torch.Tensor | None"
    fc2_input: "torch.Tensor"
    fc2_input_scale: "torch.Tensor | None"
    route_metadata: "dict[str, torch.Tensor]"
    route_metadata_format: Literal["token32_slot16_rank16", "token16_slot8_rank8"]
    layout: str


def _validate_save_for_backward(config) -> None:
    enabled = getattr(config, "save_for_backward", False)
    if type(enabled) is not bool:
        raise ValueError("save_for_backward must be a bool")
    if enabled and not getattr(config, "batch_invariant", False):
        raise ValueError("save_for_backward=True requires batch_invariant=True")


def _torch_dtype(dtype):
    import torch

    name = getattr(dtype, "__name__", str(dtype))
    names = {
        "BFloat16": torch.bfloat16,
        "Float32": torch.float32,
        "Int32": torch.int32,
        "Int64": torch.int64,
        "Uint8": torch.uint8,
        "Uint32": torch.uint32,
        "Float8E4M3FN": torch.float8_e4m3fn,
        "Float8E5M2": torch.float8_e5m2,
        "Float8E8M0FNU": getattr(torch, "float8_e8m0fnu", torch.uint8),
    }
    if int(dtype.width) == 4:
        return torch.uint8
    if name not in names:
        raise TypeError(f"Unsupported saved-forward buffer dtype {name}")
    return names[name]


def _bind_legacy_training_outputs(frontend, inputs, mega) -> None:
    if not frontend.config.save_for_backward:
        return
    import torch

    kernel = mega.kernel

    def view(name):
        spec = kernel._local_region_by_name[name]
        raw = mega.local_workspace.narrow(0, kernel._local_offsets[name], spec.nbytes)
        typed = raw.view(_torch_dtype(spec.cute_dtype))
        shape = tuple(spec.shape)
        if int(spec.cute_dtype.width) == 4:
            shape = (*shape[:-1], shape[-1] // 2)
        return typed.view(shape)

    def optional(name):
        return view(name) if name in kernel._local_region_by_name else None

    payload = view("l1_token_buffer")
    payload = payload.view(inputs.activation.dtype)
    metadata = {
        "token_src_metadata": view("token_src_metadata").view(torch.int64).reshape(-1),
        "topk_weights": view("l1_topk_weights_buffer"),
        "source_topk_idx": inputs.topk_idx,
        "source_topk_weights": inputs.topk_weights,
    }
    per_token_scale = optional("l1_fc1_activation_per_token_scale_buffer")
    if per_token_scale is not None:
        metadata["fc1_input_per_token_scale"] = per_token_scale
    frontend._training_outputs = MoEEpMegaTrainingOutputs(
        fc1_preact=view("mega_moe.fc1_preact"),
        expert_ids=view("mega_moe.fc1_expert_ids"),
        fc1_input=payload,
        fc1_input_scale=optional("l1_sf_buffer"),
        fc2_input=view("fc1_output"),
        fc2_input_scale=optional("fc1_output_sf"),
        route_metadata=metadata,
        route_metadata_format="token32_slot16_rank16",
        layout="legacy receiver pool; packed FP4 and native blocked scale storage; "
        "token_src_metadata: low32=source token, high32=(source rank<<16)|top-k slot",
    )


def _bind_next_training_outputs(frontend, kernel) -> None:
    if not frontend.config.save_for_backward:
        return

    workspace = kernel._device_workspace

    def view(name):
        region = workspace.region(name)
        root = getattr(frontend, f"{region.buffer_space}_workspace")
        raw = root.narrow(0, workspace.offset(name), workspace.nbytes(name))
        typed = raw.view(_torch_dtype(region.dtype))
        shape, stride = tuple(region.shape), tuple(workspace.stride(name))
        if int(region.dtype.width) == 4:
            # FP4 tensors have packed physical bytes, not a Torch scalar dtype.
            return typed
        if all(isinstance(x, int) for x in (*shape, *stride)):
            return typed.as_strided(shape, stride)
        return typed

    tc = kernel.token_comm
    genphase = frontend.config.kernel_variant == "genphase"
    input_region = tc.plain_activation_region if genphase else tc.fc1_activation_region
    scale_region = (
        tc.plain_activation_sf_region if genphase else tc.fc1_activation_sf_region
    )
    metadata = {
        "source_topk_idx": frontend.topk_idx,
        "source_topk_weights": frontend.topk_weights,
    }
    for space in ("local", "shared"):
        for region in workspace.regions(space):
            if "token_comm." not in region.name or region.reset != "data":
                continue
            name = region.name.rsplit(".", 1)[-1]
            if any(
                key in name
                for key in (
                    "metadata",
                    "base",
                    "index",
                    "sizes_by_rank",
                    "topk",
                    "slot",
                    "canonical_activation_sf",
                )
            ):
                metadata[name] = view(region.name)
    blackwell = ".sm100." in type(frontend).__module__
    metadata_layout = (
        "token_src_metadata: token bits0:15, slot16:23, rank24:31, "
        "destination order32:39, route count minus one40:47"
        if blackwell
        else "token_src_metadata: low32=source token, high32=(source rank<<16)|top-k slot"
    )
    frontend._training_outputs = MoEEpMegaTrainingOutputs(
        fc1_preact=view("mega_moe.fc1_preact"),
        expert_ids=view("mega_moe.fc1_expert_ids"),
        fc1_input=view(input_region),
        fc1_input_scale=view(scale_region),
        fc2_input=view(kernel.fc1_output_region),
        fc2_input_scale=view(kernel.fc1_output_sf_region),
        route_metadata=metadata,
        route_metadata_format=(
            "token16_slot8_rank8" if blackwell else "token32_slot16_rank16"
        ),
        layout=("sm100_next " if blackwell else "sm107_next ")
        + (
            "genphase source pool with receiver gather_index"
            if genphase
            else "receiver pool"
        )
        + "; native packed payload and blocked scale storage; "
        + metadata_layout,
    )


def _reset_training_outputs(frontend) -> None:
    if frontend.config.save_for_backward:
        frontend._training_outputs.expert_ids.fill_(-1)


def _training_launch(frontend, compiled):
    """Keep auxiliary initialization in every eager, autotune, and graph launch."""
    if not frontend.config.save_for_backward:
        return compiled

    def launch(**kwargs):
        _reset_training_outputs(frontend)
        return compiled(**kwargs)

    return launch
