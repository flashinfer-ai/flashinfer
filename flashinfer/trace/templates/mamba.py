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


def _selective_state_update_init(
    *,
    batch_size: int,
    dim: int = 64,
    dstate: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mamba.selective_state_update``.

    Distribution sourced from
    ``tests/mamba/test_selective_state_update_stp.py`` (normalized to the
    3-D state path the trace describes — single-head, no z-gating).
    Defaults match the test's "base" config: ``dim=64``, ``dstate=128``.
    """
    torch.manual_seed(seed)
    state = torch.randn(batch_size, dim, dstate, dtype=torch.bfloat16, device=device)
    x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device=device)
    dt = torch.randn(batch_size, dim, dtype=torch.float32, device=device)
    # A is sampled from -[1, 2] in the unit test; keeps decay positive after exp.
    A = -torch.rand(dim, dstate, dtype=torch.float32, device=device) - 1.0
    B = torch.randn(batch_size, dstate, dtype=torch.bfloat16, device=device)
    C = torch.randn(batch_size, dstate, dtype=torch.bfloat16, device=device)
    D = torch.randn(dim, dtype=torch.float32, device=device)
    dt_bias = torch.rand(dim, dtype=torch.float32, device=device) - 4.0
    return {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "dt_bias": dt_bias,
        "dt_softplus": 1,
    }


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
    init=_selective_state_update_init,
)


class _SSDCombinedTraceTemplate(TraceTemplate):
    """SSDCombined schema with the public state-dtype precedence rule."""

    def build_fi_trace_fn(self, fi_api: str):
        build_definition = super().build_fi_trace_fn(fi_api)

        def fi_trace(save_dir=None, name=None, **kwargs):
            kwargs = dict(kwargs)
            state_dtype_source = kwargs.get("initial_states")
            if state_dtype_source is None:
                state_dtype_source = kwargs.get("checkpoint_states")
            kwargs["_ssd_state_dtype_source"] = state_dtype_source
            return build_definition(save_dir=save_dir, name=name, **kwargs)

        return fi_trace


_SSD_COMBINED_COMMON_AXES = {
    "batch_size": Var(description="Input batch size."),
    "seqlen": Var(description="Tokens per batch row."),
    "nheads": Const(abbrev="h", description="Number of SSD heads."),
    "headdim": Const(abbrev="d", description="Per-head feature dimension."),
    "ngroups": Const(abbrev="g", description="Number of B/C state groups."),
    "dstate": Const(abbrev="s", description="Recurrent state dimension."),
    "nchunks": Var(description="Physical chunks per batch row."),
    "chunk_size": Const(value=128, abbrev=""),
    "num_checkpoints": Var(description="Rows in caller-owned checkpoint storage."),
}


def _make_ssd_combined_trace(
    mode: str,
    d_layout: str,
    return_final_states: bool,
) -> TraceTemplate:
    axes = dict(_SSD_COMBINED_COMMON_AXES)
    inputs = {
        "x": Tensor(["batch_size", "seqlen", "nheads", "headdim"]),
        "dt": Tensor(["batch_size", "seqlen", "nheads"]),
        "A": Tensor(["nheads"], dtype="float32"),
        "B": Tensor(["batch_size", "seqlen", "ngroups", "dstate"]),
        "C": Tensor(["batch_size", "seqlen", "ngroups", "dstate"]),
    }
    if d_layout == "vector":
        inputs["D"] = Tensor(["nheads"], dtype="bfloat16")
    elif d_layout == "matrix":
        inputs["D"] = Tensor(["nheads", "headdim"], dtype="bfloat16")
    elif d_layout != "none":
        raise ValueError(f"unsupported SSDCombined D layout: {d_layout}")

    inputs.update(
        {
            "z": Tensor(
                ["batch_size", "seqlen", "nheads", "headdim"],
                dtype="bfloat16",
                optional=True,
            ),
            "dt_bias": Tensor(["nheads"], dtype="float32", optional=True),
            "dt_softplus": Scalar("bool", optional=True),
        }
    )

    if mode == "varlen":
        axes.update(
            {
                "num_sequences": Var(description="Packed sequence count."),
                "num_logical_chunks": Var(
                    description="Number of logical packed segments."
                ),
                "seq_chunk_cumsum_size": Var(
                    description="Length of the optional prefix-sum buffer."
                ),
            }
        )
        state_batch_axis = "num_sequences"
        inputs.update(
            {
                "initial_states": Tensor(
                    ["num_sequences", "nheads", "headdim", "dstate"],
                    optional=True,
                ),
                "seq_idx": Tensor(["batch_size", "seqlen"]),
                "chunk_indices": Tensor(
                    ["num_logical_chunks"], dtype="int32"
                ),
                "chunk_offsets": Tensor(
                    ["num_logical_chunks"], dtype="int32"
                ),
                "seq_chunk_cumsum": Tensor(
                    ["seq_chunk_cumsum_size"], dtype="int32", optional=True
                ),
                "update_seq_chunk_cumsum": Scalar("bool", optional=True),
            }
        )
    elif mode == "batched":
        state_batch_axis = "batch_size"
        inputs["initial_states"] = Tensor(
            ["batch_size", "nheads", "headdim", "dstate"],
            optional=True,
        )
    else:
        raise ValueError(f"unsupported SSDCombined trace mode: {mode}")

    inputs.update(
        {
            "checkpoint_token_indices": Tensor(
                [state_batch_axis], dtype="int32", optional=True
            ),
            "checkpoint_state_slots": Tensor(
                [state_batch_axis], dtype="int32", optional=True
            ),
            "checkpoint_states": Tensor(
                ["num_checkpoints", "nheads", "headdim", "dstate"],
                optional=True,
            ),
            "out": Tensor(
                [
                    "batch_size",
                    "nheads",
                    "headdim",
                    "nchunks",
                    "chunk_size",
                ],
                dtype="bfloat16",
                optional=True,
                description="Optional caller-owned physical output storage.",
            ),
            "return_final_states": Scalar("bool", optional=True),
        }
    )

    outputs = {
        "out": Tensor(
            ["batch_size", "seqlen", "nheads", "headdim"],
            dtype_from="x",
            description=(
                "Token-major return view. The optional out parameter has a "
                "distinct physical layout, so this remains a value-returning "
                "trace output rather than a destination binding."
            ),
        ),
        "checkpoint_states": Tensor(
            ["num_checkpoints", "nheads", "headdim", "dstate"],
            param="checkpoint_states",
            dtype="bfloat16",
            dtype_from="checkpoint_states",
            optional=True,
            description="Caller-owned selective checkpoint state output.",
        ),
    }
    if mode == "varlen":
        outputs["seq_chunk_cumsum"] = Tensor(
            ["seq_chunk_cumsum_size"],
            param="seq_chunk_cumsum",
            dtype="int32",
            optional=True,
            description="Caller-owned prefix-sum buffer when update is requested.",
        )
    if return_final_states:
        outputs["final_states"] = Tensor(
            [state_batch_axis, "nheads", "headdim", "dstate"],
            dtype="bfloat16",
            dtype_from="_ssd_state_dtype_source",
        )

    final_suffix = "final" if return_final_states else "no_final"
    return _SSDCombinedTraceTemplate(
        op_type="mamba_ssd_combined",
        name_prefix=f"ssd_combined_{mode}_d_{d_layout}_{final_suffix}",
        description=(
            f"Mamba2 SSDCombined {mode} forward pass with {d_layout} D layout "
            f"and return_final_states={return_final_states}."
        ),
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        constraints=[
            "seqlen % chunk_size == 0",
            "nheads % ngroups == 0",
            "chunk_size == 128",
            "headdim == 64",
            "dstate == 128",
        ],
        tags=["status:verified", "mamba", "backend:cake", f"mode:{mode}"],
    )


_SSD_COMBINED_TRACE_BY_CONFIG = {
    (mode, d_layout, return_final_states): _make_ssd_combined_trace(
        mode,
        d_layout,
        return_final_states,
    )
    for mode in ("batched", "varlen")
    for d_layout in ("none", "vector", "matrix")
    for return_final_states in (False, True)
}


def ssd_combined_trace_dispatch(**kwargs):
    """Select the exact SSDCombined schema for mode, D rank, and tuple output."""

    mode = "varlen" if kwargs.get("seq_idx") is not None else "batched"
    D = kwargs.get("D")
    if D is None:
        d_layout = "none"
    elif D.ndim == 1:
        d_layout = "vector"
    elif D.ndim == 2:
        d_layout = "matrix"
    else:
        raise ValueError("SSDCombined trace requires D to be rank 1 or rank 2")
    return _SSD_COMBINED_TRACE_BY_CONFIG[
        (mode, d_layout, bool(kwargs.get("return_final_states", True)))
    ]


ssd_combined_trace_dispatch.templates = tuple(  # type: ignore[attr-defined]
    _SSD_COMBINED_TRACE_BY_CONFIG.values()
)
