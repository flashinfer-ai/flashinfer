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


def _ssd_canonical_flags(kwargs):
    """``kwargs`` with bool ``initial_state_flags`` replaced by their uint8
    view (zero-copy, the bytes the kernels read), so bool and uint8 flags
    trace to one definition and reach its solution in one dtype. Any other
    value is left for the API to check."""

    flags = kwargs.get("initial_state_flags")
    if isinstance(flags, torch.Tensor) and flags.dtype == torch.bool:
        kwargs = dict(kwargs)
        kwargs["initial_state_flags"] = flags.view(torch.uint8)
    return kwargs


class _SSDCombinedTraceTemplate(TraceTemplate):
    """SSDCombined schema with the public state-dtype precedence rule."""

    # ssd_combined_fwd returns ``(out, final_states | None)``; Trace Apply
    # binds a returned tuple to these outputs by name, independent of the
    # positional destination order of ``outputs``.
    return_outputs = ("out", "final_states")

    def canonical_inputs(self, namespace):
        """The Apply namespace in the form the traced definition records."""
        return _ssd_canonical_flags(namespace)

    def build_fi_trace_fn(self, fi_api: str):
        build_definition = super().build_fi_trace_fn(fi_api)

        def fi_trace(save_dir=None, name=None, **kwargs):
            kwargs = dict(_ssd_canonical_flags(kwargs))
            state_dtype_source = kwargs.get("initial_states")
            if state_dtype_source is None:
                state_dtype_source = kwargs.get("final_states")
            if state_dtype_source is None:
                state_dtype_source = kwargs.get("checkpoint_states")
            kwargs["_ssd_state_dtype_source"] = state_dtype_source
            return build_definition(save_dir=save_dir, name=name, **kwargs)

        return fi_trace


_SSD_COMBINED_COMMON_AXES: dict[str, Var | Const] = {
    "batch_size": Var(description="Input batch size."),
    "seqlen": Var(description="Tokens per batch row."),
    "nheads": Const(abbrev="h", description="Number of SSD heads."),
    "headdim": Const(abbrev="d", description="Per-head feature dimension."),
    "ngroups": Const(abbrev="g", description="Number of B/C state groups."),
    "dstate": Const(abbrev="s", description="Recurrent state dimension."),
    "chunk_size": Const(value=128, abbrev=""),
    "num_checkpoints": Var(description="Rows in caller-owned checkpoint storage."),
}

# State dtype name token of the slot schemas.
_SSD_STATE_DTYPE_KEYS = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}

# seq_idx dtype name token of the slot schemas. A runner accepts only the
# seq_idx dtype it was built for; a call without seq_idx (the cu_seqlens form)
# runs the int64 runner.
_SSD_SEQ_IDX_DTYPE_KEYS = {torch.int32: "i32", torch.int64: "i64"}
_SSD_SEQ_IDX_DTYPES = {"i32": "int32", "i64": "int64"}


def _make_ssd_combined_trace(
    mode: str,
    d_layout: str,
    return_final_states: bool,
    initial_slots: bool = False,
    final_slots: bool = False,
    state_dtype_key: str | None = None,
    seq_idx_dtype_key: str | None = None,
) -> TraceTemplate:
    """One SSDCombined schema.  ``initial_slots`` / ``final_slots`` (packed
    varlen only): ``initial_states`` / ``final_states`` is a pool with its
    own row axis, addressed per sequence through ``initial_state_slots`` /
    ``final_state_slots`` (possibly strided vectors); the other side keeps
    one dense row per sequence.  A call with either slot vector also takes
    ``checkpoint_state_slots`` into a checkpoint pool, optional per-sequence
    ``initial_state_flags`` and a ``null_slot_id``, and accepts either
    packed-varlen form (``cu_seqlens`` or the ``seq_idx`` triple).
    ``state_dtype_key`` (``bf16`` / ``f16`` / ``f32``, slot schemas only)
    names the state dtype, so pools of different dtypes route to different
    definitions; ``seq_idx_dtype_key`` (``i32`` / ``i64``, slot schemas only)
    names the seq_idx dtype in the same way."""

    state_slots = initial_slots or final_slots
    if state_slots and mode != "varlen":
        raise ValueError("SSDCombined state slots require the varlen trace mode")
    if (state_dtype_key is not None) != state_slots:
        raise ValueError("SSDCombined slot schemas, and only they, name a state dtype")
    if (seq_idx_dtype_key is not None) != state_slots:
        raise ValueError(
            "SSDCombined slot schemas, and only they, name a seq_idx dtype"
        )
    if final_slots and not return_final_states:
        raise ValueError("SSDCombined final state slots require final states")
    axes: dict[str, Var | Const] = dict(_SSD_COMBINED_COMMON_AXES)
    inputs: dict[str, Tensor | Scalar] = {
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
        initial_rows_axis = "num_sequences"
        if state_slots:
            axes["cu_seqlens_size"] = Var(
                description="Length of the optional cu_seqlens vector."
            )
        if initial_slots:
            axes["initial_state_rows"] = Var(
                description="Rows of the initial-state pool."
            )
            initial_rows_axis = "initial_state_rows"
        if final_slots:
            axes["final_state_rows"] = Var(description="Rows of the final-state pool.")
        inputs.update(
            {
                # A slot vector requires its pool.
                "initial_states": Tensor(
                    [initial_rows_axis, "nheads", "headdim", "dstate"],
                    optional=not initial_slots,
                ),
                "seq_idx": (
                    Tensor(
                        ["batch_size", "seqlen"],
                        dtype=_SSD_SEQ_IDX_DTYPES[seq_idx_dtype_key],
                        optional=True,
                    )
                    if state_slots
                    else Tensor(["batch_size", "seqlen"])
                ),
                "chunk_indices": Tensor(
                    ["num_logical_chunks"], dtype="int32", optional=state_slots
                ),
                "chunk_offsets": Tensor(
                    ["num_logical_chunks"], dtype="int32", optional=state_slots
                ),
                "seq_chunk_cumsum": Tensor(
                    ["seq_chunk_cumsum_size"], dtype="int32", optional=True
                ),
                "update_seq_chunk_cumsum": Scalar("bool", optional=True),
            }
        )
        if state_slots:
            inputs["cu_seqlens"] = Tensor(
                ["cu_seqlens_size"], dtype="int32", optional=True
            )
            inputs["null_slot_id"] = Scalar(
                "int32",
                optional=True,
                description="Slot value that addresses no pool row.",
            )
        if initial_slots:
            inputs.update(
                {
                    "initial_state_slots": Tensor(
                        ["num_sequences"],
                        dtype="int32",
                        optional=True,
                        description="Initial-state pool row of each sequence.",
                    ),
                    "initial_state_flags": Tensor(
                        ["num_sequences"],
                        dtype="uint8",
                        optional=True,
                        description=(
                            "uint8 view of the caller's bool / uint8 "
                            "per-sequence flags: 0 starts the sequence from "
                            "zero state."
                        ),
                    ),
                }
            )
        if final_slots:
            inputs["final_state_slots"] = Tensor(
                ["num_sequences"],
                dtype="int32",
                optional=True,
                description="Final-state pool row of each sequence.",
            )
    elif mode == "batched":
        state_batch_axis = "batch_size"
        inputs["initial_states"] = Tensor(
            ["batch_size", "nheads", "headdim", "dstate"],
            optional=True,
        )
    else:
        raise ValueError(f"unsupported SSDCombined trace mode: {mode}")

    final_rows_axis = "final_state_rows" if final_slots else state_batch_axis
    inputs.update(
        {
            "checkpoint_token_indices": Tensor(
                [state_batch_axis], dtype="int32", optional=True
            ),
            "checkpoint_state_slots": Tensor(
                [state_batch_axis],
                dtype="int32",
                optional=True,
                description=(
                    "Checkpoint-state pool row of each sequence (may be "
                    "strided); skipped when < 0, >= the pool's rows or "
                    "equal to null_slot_id."
                    if state_slots
                    else ""
                ),
            ),
            "checkpoint_states": Tensor(
                ["num_checkpoints", "nheads", "headdim", "dstate"],
                optional=True,
                description=(
                    "Checkpoint state pool rows addressed through "
                    "checkpoint_state_slots."
                    if state_slots
                    else ""
                ),
            ),
            "final_states": Tensor(
                [final_rows_axis, "nheads", "headdim", "dstate"],
                optional=not final_slots,
                description=(
                    "Caller-owned final-state pool."
                    if final_slots
                    else "Optional caller-owned final-state storage."
                ),
            ),
            "out": Tensor(
                ["batch_size", "seqlen", "nheads", "headdim"],
                dtype="bfloat16",
                optional=True,
                description=(
                    "Optional caller-owned contiguous token-major output "
                    "storage; returned as out."
                ),
            ),
            "return_final_states": Scalar("bool", optional=True),
            "state_dtype": Scalar("dtype", optional=True),
        }
    )

    # Positional destination order (main's dense order, shared by the slot
    # schemas): out, checkpoint_states, seq_chunk_cumsum, final_states. A
    # value-returning solution's ``(out, final_states)`` binds by name through
    # ``_SSDCombinedTraceTemplate.return_outputs``.
    outputs: dict[str, Tensor | Scalar] = {
        "out": Tensor(
            ["batch_size", "seqlen", "nheads", "headdim"],
            dtype_from="x",
            description=(
                "Token-major output [batch, seqlen, nheads, headdim]; the "
                "optional out parameter, when given, has this same layout and "
                "is returned."
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
    if final_slots:
        outputs["final_states"] = Tensor(
            [final_rows_axis, "nheads", "headdim", "dstate"],
            param="final_states",
            dtype="bfloat16",
            dtype_from="_ssd_state_dtype_source",
            dtype_from_scalar="state_dtype",
            description="Final states, stored into the caller's pool rows.",
        )
    elif return_final_states:
        # Bound to the caller's dense final_states when given (an unbound
        # output would be derived to the out parameter).
        outputs["final_states"] = Tensor(
            [state_batch_axis, "nheads", "headdim", "dstate"],
            param="final_states",
            dtype="bfloat16",
            dtype_from="_ssd_state_dtype_source",
            dtype_from_scalar="state_dtype",
        )

    constraints = [
        "nheads % ngroups == 0",
        "chunk_size == 128",
        "headdim == 64",
        "dstate == 128",
    ]
    if not state_slots:
        # A state-slot call runs on the Cake backend only, which takes any
        # positive sequence length (a partial trailing chunk included).
        constraints.insert(0, "seqlen % chunk_size == 0")
    final_suffix = "final" if return_final_states else "no_final"
    # Both state sides pools, or only the initial / only the final side.
    mode_name, slot_text = mode, "."
    if initial_slots and final_slots:
        mode_name += "_slots"
        slot_text = ", states read and written through state-pool slots"
    elif initial_slots:
        mode_name += "_islots"
        slot_text = ", initial states read through state-pool slots"
    elif final_slots:
        mode_name += "_fslots"
        slot_text = ", final states written through state-pool slots"
    if state_dtype_key is not None:
        # The dense schema names stay without a state or seq_idx dtype.
        mode_name += f"_{state_dtype_key}_sidx_{seq_idx_dtype_key}"
        slot_text += (
            f", {state_dtype_key} states, "
            f"{_SSD_SEQ_IDX_DTYPES[seq_idx_dtype_key]} seq_idx."
        )
    return _SSDCombinedTraceTemplate(
        op_type="mamba_ssd_combined",
        name_prefix=f"ssd_combined_{mode_name}_d_{d_layout}_{final_suffix}",
        description=(
            f"Mamba2 SSDCombined {mode} forward pass with {d_layout} D layout "
            f"and return_final_states={return_final_states}{slot_text}"
        ),
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        tags=["status:verified", "mamba", "backend:cake", f"mode:{mode}"],
    )


_SSD_COMBINED_TRACE_BY_CONFIG = {
    (
        mode,
        d_layout,
        return_final_states,
        initial_slots,
        final_slots,
        state_dtype_key,
        seq_idx_dtype_key,
    ): _make_ssd_combined_trace(
        mode,
        d_layout,
        return_final_states,
        initial_slots,
        final_slots,
        state_dtype_key,
        seq_idx_dtype_key,
    )
    for mode, initial_slots, final_slots in (
        ("batched", False, False),
        ("varlen", False, False),
        ("varlen", True, True),
        ("varlen", True, False),
        ("varlen", False, True),
    )
    for d_layout in ("none", "vector", "matrix")
    for return_final_states in (False, True)
    # The final-state pool is written only with return_final_states=True.
    if return_final_states or not final_slots
    # One slot schema per state dtype and seq_idx dtype; the dense schemas
    # name neither.
    for state_dtype_key in (
        _SSD_STATE_DTYPE_KEYS.values() if initial_slots or final_slots else (None,)
    )
    for seq_idx_dtype_key in (
        _SSD_SEQ_IDX_DTYPE_KEYS.values() if initial_slots or final_slots else (None,)
    )
}


def ssd_combined_trace_dispatch(**kwargs):
    """Select the exact SSDCombined schema for mode, D rank, tuple output,
    which state side is a slot-indexed pool (a call with a slot vector is
    packed varlen) and, for a slot call, the state and seq_idx dtypes."""

    initial_slots = kwargs.get("initial_state_slots") is not None
    final_slots = kwargs.get("final_state_slots") is not None
    slots = initial_slots or final_slots
    # The API rejects a slot vector without its pool before any write, so
    # such a call must reach it rather than match a slot definition.
    for slots_name, pool_name in (
        ("initial_state_slots", "initial_states"),
        ("final_state_slots", "final_states"),
    ):
        if kwargs.get(slots_name) is not None and kwargs.get(pool_name) is None:
            raise ValueError(f"SSDCombined trace: {slots_name} requires {pool_name}")
    mode = "varlen" if slots or kwargs.get("seq_idx") is not None else "batched"
    D = kwargs.get("D")
    if D is None:
        d_layout = "none"
    elif D.ndim == 1:
        d_layout = "vector"
    elif D.ndim == 2:
        d_layout = "matrix"
    else:
        raise ValueError("SSDCombined trace requires D to be rank 1 or rank 2")
    return_final_states = bool(kwargs.get("return_final_states", True))
    if final_slots and not return_final_states:
        raise ValueError(
            "SSDCombined trace: final_state_slots requires return_final_states=True"
        )
    state_dtype_key = seq_idx_dtype_key = None
    if slots:
        # The state dtype rule of ssd_combined_fwd: state_dtype, else the
        # first state tensor given, else bfloat16; every state tensor given
        # must have that dtype.
        state_tensors = {
            name: kwargs.get(name)
            for name in ("initial_states", "final_states", "checkpoint_states")
        }
        state_dtype = kwargs.get("state_dtype")
        if state_dtype is None:
            state_dtype = next(
                (
                    states.dtype
                    for states in state_tensors.values()
                    if states is not None
                ),
                torch.bfloat16,
            )
        state_dtype_key = _SSD_STATE_DTYPE_KEYS.get(state_dtype)
        if state_dtype_key is None:
            raise ValueError(
                f"SSDCombined trace: unsupported state dtype {state_dtype}"
            )
        for name, states in state_tensors.items():
            if states is not None and states.dtype != state_dtype:
                raise ValueError(
                    f"SSDCombined trace: {name} dtype {states.dtype} doesn't "
                    f"match state_dtype {state_dtype}"
                )
        # The runner's seq_idx dtype: seq_idx's own, else int64.
        seq_idx = kwargs.get("seq_idx")
        seq_idx_dtype = torch.int64 if seq_idx is None else seq_idx.dtype
        seq_idx_dtype_key = _SSD_SEQ_IDX_DTYPE_KEYS.get(seq_idx_dtype)
        if seq_idx_dtype_key is None:
            raise ValueError(
                f"SSDCombined trace: unsupported seq_idx dtype {seq_idx_dtype}"
            )
    return _SSD_COMBINED_TRACE_BY_CONFIG[
        (
            mode,
            d_layout,
            return_final_states,
            initial_slots,
            final_slots,
            state_dtype_key,
            seq_idx_dtype_key,
        )
    ]


ssd_combined_trace_dispatch.templates = tuple(  # type: ignore[attr-defined]
    _SSD_COMBINED_TRACE_BY_CONFIG.values()
)
