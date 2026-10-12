# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider-independent GDN gate conventions."""

import torch


def validate_gate_inputs(
    q, v, g, beta, gate_domain, use_gate_in_kernel, A_log, dt_bias, beta_is_logit
):
    if gate_domain not in ("linear", "log"):
        raise ValueError("gate_domain must be 'linear' or 'log'")
    if use_gate_in_kernel and (g is None or A_log is None or dt_bias is None):
        raise ValueError("use_gate_in_kernel requires g, A_log and dt_bias")
    if not use_gate_in_kernel and (A_log is not None or dt_bias is not None):
        raise ValueError("A_log and dt_bias require use_gate_in_kernel=True")
    if beta_is_logit and beta is None:
        raise ValueError("beta_is_logit requires beta")
    if not (use_gate_in_kernel or beta_is_logit or gate_domain == "log"):
        return
    heads = max(q.shape[1], v.shape[1])
    for name, tensor, shape in (
        ("g", g, (q.shape[0], heads)),
        ("beta", beta, (q.shape[0], heads)),
        ("A_log", A_log, (heads,)),
        ("dt_bias", dt_bias, (heads,)),
    ):
        if tensor is None:
            continue
        if tensor.shape != shape or tensor.device != q.device:
            raise ValueError(f"{name} must have shape {shape} on q's device")
        dtypes = (
            (torch.float32, q.dtype)
            if name == "beta"
            else (torch.float32, torch.float16, torch.bfloat16)
        )
        if tensor.dtype not in dtypes:
            raise ValueError(f"unsupported {name} dtype: {tensor.dtype}")


def materialize_gates(
    g, beta, gate_domain, use_gate_in_kernel, A_log, dt_bias, beta_is_logit
):
    """Adapt new conventions for kernels taking linear alpha and FP32 beta."""
    if use_gate_in_kernel:
        log_alpha = -torch.exp(A_log.float()) * torch.nn.functional.softplus(
            g.float() + dt_bias.float()
        )
        g = torch.exp(log_alpha)
    elif gate_domain == "log" and g is not None:
        g = torch.exp(g.float())
    if beta_is_logit:
        # FE's fused sigmoid preserves the input gate's storage rounding.
        beta = torch.sigmoid(beta.float()).to(beta.dtype).float()
    return (
        g.float() if g is not None else None,
        beta.float() if beta is not None else None,
    )
