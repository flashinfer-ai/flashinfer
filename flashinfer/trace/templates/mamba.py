# Copyright (c) 2025 by FlashInfer team.
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

"""TraceTemplates for Mamba SSM ops."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _selective_state_update_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z=None,
    dt_bias=None,
    dt_softplus: bool = False,
    **_unused,
) -> torch.Tensor:
    """Reference for Mamba selective state update (decode / single-token form).

    Implements the discrete recurrence:
        dt'    = softplus(dt + dt_bias) if dt_softplus else (dt + dt_bias)
        dA     = exp(dt' * A)
        dB     = dt' * B
        state  = state * dA + dB * x.unsqueeze(-1)
        y      = (state @ C.unsqueeze(-1)).squeeze(-1) + D * x
        if z is not None: y = y * silu(z)

    Mutates ``state`` in-place. Skips the optional state-cache routing
    (state_batch_indices, intermediate_states_buffer, etc.) — those are
    runtime plumbing that the trace JSON captures via input shapes only.
    Multi-head / multi-token forms are normalized to the 3-D state path.
    """
    # Minimal contract: state [batch, dim, dstate], x [batch, dim].
    if state.dim() == 4:
        # [B, H, D, S] → flatten heads.
        b, h, d, s = state.shape
        state = state.reshape(b * h, d, s)
    if x.dim() == 3:
        # [B, H, D]
        x = x.reshape(-1, x.shape[-1])
    if dt.dim() == 3:
        dt = dt.reshape(-1, dt.shape[-1])

    state_f = state.to(torch.float32)
    x_f = x.to(torch.float32)
    dt_f = dt.to(torch.float32)
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_f = C.to(torch.float32)
    D_f = D.to(torch.float32)
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.to(torch.float32)
    if dt_softplus:
        dt_f = torch.nn.functional.softplus(dt_f)
    # dA: [batch, dim, dstate]
    dA = torch.exp(dt_f.unsqueeze(-1) * A_f)
    # dB: [batch, dim, dstate]
    dB = (
        dt_f.unsqueeze(-1) * B_f.unsqueeze(1)
        if B_f.dim() == 2
        else dt_f.unsqueeze(-1) * B_f
    )
    state_new = state_f * dA + dB * x_f.unsqueeze(-1)
    # y = state @ C; C is [batch, dstate].
    if C_f.dim() == 2:
        y = (state_new * C_f.unsqueeze(1)).sum(dim=-1)
    else:
        y = (state_new * C_f).sum(dim=-1)
    y = y + D_f * x_f
    if z is not None:
        z_f = z.to(torch.float32).reshape(y.shape)
        y = y * (z_f * torch.sigmoid(z_f))
    state.copy_(state_new.to(state.dtype))
    return y.to(x.dtype)


selective_state_update_trace = TraceTemplate(
    op_type="mamba",
    name_prefix="selective_state_update",
    description=(
        "Mamba SSM selective-state-update kernel (decode phase). Updates "
        "the per-sequence state in-place and returns the per-token output. "
        "The trace captures the most common single-token shapes; the "
        "kernel itself supports many additional layouts (multi-head, "
        "varlen multi-token, FP8 state cache) which are all variants of "
        "the same SSM recurrence."
    ),
    axes={
        "batch_size": Var(),
        "dim": Const(abbrev="d"),
        "dstate": Const(abbrev="s"),
    },
    inputs={
        "state": Tensor(
            ["batch_size", "dim", "dstate"],
            description="Recurrent SSM state (mutated in-place).",
        ),
        "x": Tensor(["batch_size", "dim"]),
        "dt": Tensor(["batch_size", "dim"]),
        "A": Tensor(["dim", "dstate"]),
        "B": Tensor(["batch_size", "dstate"]),
        "C": Tensor(["batch_size", "dstate"]),
        "D": Tensor(["dim"]),
        "z": Tensor(["batch_size", "dim"], optional=True),
        "dt_bias": Tensor(["dim"], optional=True),
        "dt_softplus": Scalar("int32", optional=True),
    },
    outputs={
        "out": Tensor(["batch_size", "dim"], dtype_from="x"),
    },
    tags=["status:verified", "mamba"],
    reference=_selective_state_update_reference,
)
