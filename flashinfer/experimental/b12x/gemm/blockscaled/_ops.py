"""Opaque prepared packed calls with exact caller-buffer mutation effects."""
from __future__ import annotations

import torch

from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x._lib.utils import cuda_stream_to_int


def _execute(source, values, scales, global_scale, activation_scale, workspace, out,
             global_scale_kind, stream, plan_handle, required_mode, expected_m=None):
    declaration = plan_from_handle(plan_handle) if plan_handle else None
    if declaration is None:
        from ._preparation import heuristic_plan
        from ._tuning import BlockscaledQuery
        declaration = heuristic_plan(BlockscaledQuery(
            recipe="mxfp8" if values.dtype == torch.float8_e4m3fn else "nvfp4",
            num_tokens=source.numel() // source.shape[-1], in_features=source.shape[-1],
            padded_in_features=values.shape[-1] * (2 if values.dtype == torch.uint8 else 1),
            out_features=values.shape[0], global_scale_kind=global_scale_kind, expected_m=expected_m,
            activation_mode="a16" if required_mode == "a16" else "auto",
            activation_scale_available=activation_scale is not None,
            source_contiguous=source.is_contiguous(), source_aligned=source.data_ptr() % 16 == 0,
            output_mode="functional" if out is None else "provided",
            workspace_form="owned" if workspace is None else "provided",
            workspace_nbytes=None if workspace is None else workspace.numel(),
        ), source.device)
    state = require_prepared(declaration, "gemm.blockscaled_precision", source.device)
    resolve = getattr(state, "resolve", None)
    if resolve is not None:
        state = resolve(source)
    if required_mode != "any" and state.config.mode != required_mode:
        raise ValueError("this entry point requires an A16 prepared plan")
    if global_scale_kind != state.query.global_scale_kind:
        raise ValueError("weight scale semantics differ from prepared invocation")
    if state.query.recipe == "mxfp8" and scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e8m0fnu)
    return state.run(source, values, scales, global_scale, activation_scale=activation_scale,
                     workspace=workspace, out=out, stream=stream)


def _metadata(source, values):
    return source.new_empty((*source.shape[:-1], values.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("b12x::blockscaled_bf16", mutates_args=())
def _functional(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                global_scale_kind: str, stream: int | None, plan_handle: int, required_mode: str, expected_m: int | None = None) -> torch.Tensor:
    return _execute(source, values, scales, global_scale, activation_scale, None, None,
                    global_scale_kind, stream, plan_handle, required_mode, expected_m)


@_functional.register_fake
def _functional_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                     global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                     global_scale_kind: str, stream: int | None, plan_handle: int, required_mode: str, expected_m: int | None = None) -> torch.Tensor:
    return _metadata(source, values)


@torch.library.custom_op("b12x::blockscaled_bf16_workspace", mutates_args=("workspace",))
def _functional_workspace(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                          global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                          workspace: torch.Tensor, global_scale_kind: str, stream: int | None,
                          plan_handle: int, required_mode: str, expected_m: int | None = None) -> torch.Tensor:
    return _execute(source, values, scales, global_scale, activation_scale, workspace, None,
                    global_scale_kind, stream, plan_handle, required_mode, expected_m)


@_functional_workspace.register_fake
def _functional_workspace_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                               global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                               workspace: torch.Tensor, global_scale_kind: str, stream: int | None,
                               plan_handle: int, required_mode: str, expected_m: int | None = None) -> torch.Tensor:
    return _metadata(source, values)


@torch.library.custom_op("b12x::blockscaled_bf16_out", mutates_args=("out",))
def _out(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
         global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
         out: torch.Tensor, global_scale_kind: str, stream: int | None,
         plan_handle: int, required_mode: str, expected_m: int | None = None) -> None:
    _execute(source, values, scales, global_scale, activation_scale, None, out,
             global_scale_kind, stream, plan_handle, required_mode, expected_m)


@_out.register_fake
def _out_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
              global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
              out: torch.Tensor, global_scale_kind: str, stream: int | None,
              plan_handle: int, required_mode: str, expected_m: int | None = None) -> None:
    del source, values, scales, global_scale, activation_scale, out, global_scale_kind, stream, plan_handle


@torch.library.custom_op("b12x::blockscaled_bf16_out_workspace", mutates_args=("out", "workspace"))
def _out_workspace(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                   global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                   out: torch.Tensor, workspace: torch.Tensor, global_scale_kind: str,
                   stream: int | None, plan_handle: int, required_mode: str, expected_m: int | None = None) -> None:
    _execute(source, values, scales, global_scale, activation_scale, workspace, out,
             global_scale_kind, stream, plan_handle, required_mode, expected_m)


@_out_workspace.register_fake
def _out_workspace_fake(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor,
                        global_scale: torch.Tensor | None, activation_scale: torch.Tensor | None,
                        out: torch.Tensor, workspace: torch.Tensor, global_scale_kind: str,
                        stream: int | None, plan_handle: int, required_mode: str, expected_m: int | None = None) -> None:
    del source, values, scales, global_scale, activation_scale, out, workspace, global_scale_kind, stream, plan_handle


def linear(source, values, scales, global_scale, *, plan: Plan,
           global_scale_kind: str, activation_global_scale=None, out=None, workspace=None, stream=None, required_mode="any", expected_m=None):
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    args = (source, values, scales, global_scale, activation_global_scale)
    stream_int = cuda_stream_to_int(stream)
    handle = 0 if plan is None else plan.handle
    if out is None:
        if workspace is None:
            return torch.ops.b12x.blockscaled_bf16(*args, global_scale_kind, stream_int, handle, required_mode, expected_m)
        return torch.ops.b12x.blockscaled_bf16_workspace(*args, workspace, global_scale_kind, stream_int, handle, required_mode, expected_m)
    if workspace is None:
        torch.ops.b12x.blockscaled_bf16_out(*args, out, global_scale_kind, stream_int, handle, required_mode, expected_m)
    else:
        torch.ops.b12x.blockscaled_bf16_out_workspace(*args, out, workspace, global_scale_kind, stream_int, handle, required_mode, expected_m)
    return out
