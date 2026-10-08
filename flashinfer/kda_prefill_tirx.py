# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host integration for TIRx KDA with FP32 state."""

import math

import torch

from .kda_prefill import (
    RecurrentKDAPrefillWorkspace,
    _bind_workspace,
    _check_output_does_not_overlap_inputs,
    _get_stream_workspace,
    _storage_ranges_overlap,
)
from .utils import get_compute_capability


def _validate_tensor(name, tensor, shape, dtype, device, alignment=16):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % alignment
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} on {device}, shape {tuple(shape)}, "
            f"with {alignment}-byte alignment for backend='tirx'"
        )


def _tensor_signature(tensor):
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _run_tirx_kda(
    *,
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    beta_is_logit,
    lower_bound,
    cu_seqlens,
    output,
    prefill_workspace,
):
    if not isinstance(q, torch.Tensor) or q.ndim != 4 or not q.is_cuda:
        raise ValueError("q must be a CUDA [B, T, H, 128] tensor")
    if get_compute_capability(q.device) not in ((10, 0), (10, 3)):
        raise ValueError("backend='tirx' requires SM100 or SM103")
    B, T, H, D = q.shape
    if B < 1 or T <= 1 or H < 8 or H % 8 or D != 128:
        raise ValueError("TIRx KDA requires B >= 1, T > 1, H divisible by 8, D=128")
    if torch.cuda.get_device_properties(q.device).multi_processor_count < H:
        raise ValueError("TIRx KDA requires H <= the device SM count")
    if B * T >= 2**21:
        raise ValueError("TIRx KDA requires fewer than 2**21 total tokens")
    if B * T * H * D >= 2**31:
        raise ValueError("TIRx KDA requires B*T*H*128 < 2**31")
    if not (use_qk_l2norm_in_kernel and use_gate_in_kernel and beta_is_logit):
        raise ValueError(
            "TIRx KDA requires fused Q/K normalization, gate and beta sigmoid"
        )
    if lower_bound is None or not -5.0 <= float(lower_bound) < 0.0:
        raise ValueError("TIRx KDA requires lower_bound in [-5, 0)")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
        _validate_tensor(name, tensor, q.shape, torch.bfloat16, q.device)
    _validate_tensor("beta", beta, (B, T, H), torch.bfloat16, q.device)
    _validate_tensor("A_log", A_log, (H,), torch.float32, q.device)
    if dt_bias is None or dt_bias.shape not in ((H * D,), (H, D)):
        raise ValueError("dt_bias must have shape [H*128] or [H, 128]")
    _validate_tensor("dt_bias", dt_bias, dt_bias.shape, torch.float32, q.device)
    if cu_seqlens is not None:
        if (
            B != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("packed KDA requires B=1 and int32/int64 cu_seqlens[N+1]")
        _validate_tensor(
            "cu_seqlens",
            cu_seqlens,
            cu_seqlens.shape,
            cu_seqlens.dtype,
            q.device,
            cu_seqlens.element_size(),
        )
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if N >= 65536:
        raise ValueError("TIRx KDA requires fewer than 65536 sequences")
    if N * H * D * D >= 2**31:
        raise ValueError("TIRx KDA requires N*H*128*128 < 2**31")
    if initial_state is not None:
        _validate_tensor(
            "initial_state", initial_state, (N, H, D, D), torch.float32, q.device, 32
        )
    if output is not None:
        _validate_tensor("output", output, q.shape, q.dtype, q.device)
    scale_value = D**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError("scale must be finite")

    # Both compilation and launch must use the input device's current stream.
    with torch.cuda.device(q.device):
        return _launch_tirx_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale_value,
            initial_state,
            output_final_state,
            float(lower_bound),
            cu_seqlens,
            output,
            prefill_workspace,
        )


def _launch_tirx_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    output_final_state,
    lower_bound,
    cu_seqlens,
    output,
    prefill_workspace,
):
    try:
        from .kda_kernels.tirx.host import make_plan
        import tvm
    except ImportError as exc:
        raise RuntimeError(
            "backend='tirx' requires CUDA-enabled apache-tvm with TIRx and "
            "tirx_kernels.tirx_lite; see the TIRx installation instructions in docs/api/kda.rst"
        ) from exc
    if not tvm.runtime.enabled("cuda"):
        raise RuntimeError("backend='tirx' requires a CUDA-enabled TVM build")

    capturing = torch.cuda.is_current_stream_capturing()
    explicit = prefill_workspace is not None
    if capturing and (not explicit or output is None):
        raise RuntimeError(
            "TIRx KDA capture requires an explicit warmed workspace and output"
        )
    if explicit and not isinstance(prefill_workspace, RecurrentKDAPrefillWorkspace):
        raise TypeError("prefill_workspace must be a RecurrentKDAPrefillWorkspace")
    workspace = prefill_workspace if explicit else _get_stream_workspace(q.device)
    out = torch.empty_like(v) if output is None else output
    _check_output_does_not_overlap_inputs(
        output, q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state
    )
    for name, tensor in (
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("cu_seqlens", cu_seqlens),
    ):
        if tensor is not None and _storage_ranges_overlap(out, tensor):
            raise ValueError(f"output must not overlap {name}")
    if initial_state is not None:
        for tensor in (q, k, v, g, beta, A_log, dt_bias, cu_seqlens):
            if tensor is not None and _storage_ranges_overlap(initial_state, tensor):
                raise ValueError("initial_state must not overlap other inputs")
    tensors = (q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, out)
    cu_version = (
        None if cu_seqlens is None or cu_seqlens.is_inference() else cu_seqlens._version
    )
    signature = (
        tuple(map(_tensor_signature, tensors)),
        cu_version,
        scale,
        lower_bound,
        output_final_state,
    )
    with workspace._lock:
        _bind_workspace(
            workspace,
            device=q.device,
            stream_ptr=int(torch.cuda.current_stream(q.device).cuda_stream),
            capturing=capturing,
            explicit=explicit,
        )
        entry = getattr(workspace, "_tirx_kda", None)
        if capturing:
            if entry is None or entry["signature"] != signature:
                raise RuntimeError(
                    "TIRx KDA workspace must be warmed with the exact capture tensors"
                )
        else:
            B, T, H, D = q.shape
            offsets = (
                list(range(0, B * T + 1, T))
                if cu_seqlens is None
                else cu_seqlens.tolist()
            )
            if (
                offsets[0] != 0
                or offsets[-1] != B * T
                or any(
                    end <= start
                    for start, end in zip(offsets, offsets[1:], strict=False)
                )
            ):
                raise ValueError(
                    "cu_seqlens must start at 0, end at total tokens, and be strictly increasing"
                )
            plan_key = (tuple(offsets), signature)
            if entry is None or entry["plan_key"] != plan_key:
                zero_state = (
                    torch.zeros(
                        len(offsets) - 1, H, D, D, dtype=torch.float32, device=q.device
                    )
                    if initial_state is None
                    else initial_state
                )
                state = (
                    torch.empty_like(zero_state)
                    if initial_state is None
                    else initial_state
                )
                data = {
                    "q": q,
                    "k": k,
                    "v": v,
                    "g": g,
                    "beta": beta,
                    "A_log": A_log,
                    "dt_bias": dt_bias,
                    "scale": scale,
                    "lower_bound": lower_bound,
                    "initial_state": zero_state,
                    "final_state": state,
                    "output": out,
                }
                entry = {
                    "plan_key": plan_key,
                    "plan": make_plan(data, offsets, B == 1 and cu_seqlens is None),
                    "state": state,
                    "data": data,
                }
                # An internally allocated output never matches a later call,
                # so caching that plan would only pin its tensors and scratch.
                if explicit or output is not None:
                    workspace.__dict__["_tirx_kda"] = entry
            entry["signature"] = signature
            entry["tensors"] = tensors
        state = entry["state"]
        entry["plan"]["launch"]()
        if capturing:
            workspace._captured = True
        # Implicit eager scratch may be reused on the next call. Give callers
        # ownership of the final state when they supplied no initial state.
        if output_final_state and initial_state is None and not explicit:
            state = state.clone()
    return out, state if output_final_state else None
