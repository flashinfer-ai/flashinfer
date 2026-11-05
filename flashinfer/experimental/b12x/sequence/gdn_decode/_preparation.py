"""Resolved GDN/KDA launches with dynamic packed sequence metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import triton

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._impl import Binding, Caps, KdaBinding, _bind, _bind_kda, _scratch_layout
from ._tuning import GdnConfig, GdnQuery, TUNING, _OPERANDS, aligned_operands


def _alignment(tensor):
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def _kda_strides(mixed_qkv, a, b, dt_bias, state, output):
    return (
        mixed_qkv.stride(0),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        dt_bias.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        output.stride(0),
        output.stride(1),
    )


def invocation_from_tensors(caps: Caps, *, scratch=None, **tensors) -> FrozenMapping:
    """Describe immutable dtype/layout metadata; never read tensor contents.

    The scratch buffer carries no invocation metadata and is accepted only so
    callers can pass their complete binding arguments.
    """
    del scratch
    if not isinstance(caps, Caps):
        raise TypeError("invocation metadata requires GDN Caps")
    kda = caps.key_heads == caps.value_heads
    required = set(_OPERANDS)
    if kda:
        required = (required - {"a", "b"}) | {"raw_g", "raw_beta"}
    missing = required - tensors.keys()
    if missing:
        raise ValueError(
            f"GDN invocation is missing tensor metadata: {', '.join(sorted(missing))}"
        )
    if kda:
        tensors = {**tensors, "a": tensors["raw_g"], "b": tensors["raw_beta"]}
    fields = {
        "a_log_dtype": str(tensors["A_log"].dtype).removeprefix("torch."),
        "dt_bias_dtype": str(tensors["dt_bias"].dtype).removeprefix("torch."),
        "norm_weight_dtype": str(tensors["norm_weight"].dtype).removeprefix("torch."),
        "state_indices_dtype": str(tensors["state_indices"].dtype).removeprefix(
            "torch."
        ),
        "kda_strides": _kda_strides(
            tensors["mixed_qkv"],
            tensors["a"],
            tensors["b"],
            tensors["dt_bias"],
            tensors["recurrent_state"],
            tensors["output"],
        )
        if kda
        else None,
        "pointer_alignments": {
            name: _alignment(tensors[name]) for name in _OPERANDS if name in tensors
        },
    }
    return FrozenMapping(fields)


def _query_from_caps(caps, invocation):
    allowed = {
        "a_log_dtype",
        "dt_bias_dtype",
        "norm_weight_dtype",
        "state_indices_dtype",
        "kda_strides",
        "pointer_alignments",
    }
    if set(invocation) - allowed:
        raise ValueError("unknown GDN invocation metadata")
    return GdnQuery(
        gate_activation=caps.gate_activation,
        qk_l2norm=caps.qk_l2norm,
        state_dtype=str(caps.state_dtype).removeprefix("torch."),
        key_heads=caps.key_heads,
        value_heads=caps.value_heads,
        max_seqs=caps.max_seqs,
        max_tokens=caps.max_tokens,
        state_index_columns=caps.state_index_columns,
        max_state_slots=caps.max_state_slots,
        null_state_index=caps.null_state_index,
        **dict(invocation),
        recover_speculative_state=caps.recover_speculative_state,
    )


def _caps(query, ordinal):
    return Caps(
        device=torch.device("cuda", ordinal),
        max_tokens=query.max_tokens,
        max_seqs=query.max_seqs,
        max_state_slots=query.max_state_slots,
        key_heads=query.key_heads,
        value_heads=query.value_heads,
        state_index_columns=query.state_index_columns,
        state_dtype=getattr(torch, query.state_dtype),
        gate_activation=query.gate_activation,
        qk_l2norm=query.qk_l2norm,
        null_state_index=query.null_state_index,
        recover_speculative_state=query.recover_speculative_state,
    )


@dataclass(frozen=True)
class _TensorMetadata:
    dtype: torch.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    device: torch.device
    alignment: int

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def data_ptr(self):
        return self.alignment


def _metadata_binding(query, layout):
    caps = layout.caps
    m, n, h, columns = (
        caps.max_tokens,
        caps.max_seqs,
        caps.value_heads,
        caps.state_index_columns,
    )
    kda = caps.key_heads == h

    def tensor(name, shape, dtype=torch.bfloat16, strides=None):
        if strides is None:
            stride, result = 1, []
            for size in reversed(shape):
                result.append(stride)
                stride *= size
            strides = tuple(reversed(result))
        return _TensorMetadata(
            dtype,
            tuple(shape),
            tuple(strides),
            caps.device,
            query.pointer_alignments[name],
        )

    strides = query.kda_strides
    data = dict(
        mixed_qkv=tensor(
            "mixed_qkv",
            (m, caps.packed_qkv_width),
            strides=(strides[0], 1) if kda else None,
        ),
        z=tensor("z", (m, h, 128)),
        A_log=tensor("A_log", (h,), getattr(torch, query.a_log_dtype)),
        dt_bias=tensor(
            "dt_bias",
            (h, 128) if kda else (h,),
            getattr(torch, query.dt_bias_dtype),
            (strides[5], 1) if kda else None,
        ),
        norm_weight=tensor(
            "norm_weight", (128,), getattr(torch, query.norm_weight_dtype)
        ),
        recurrent_state=tensor(
            "recurrent_state",
            (caps.max_state_slots, h, 128, 128),
            caps.state_dtype,
            (*strides[6:9], 1) if kda else None,
        ),
        query_start_loc=tensor("query_start_loc", (n + 1,), torch.int32),
        num_accepted_tokens=tensor("num_accepted_tokens", (n,), torch.int32),
        state_indices=tensor(
            "state_indices", (n, columns), getattr(torch, query.state_indices_dtype)
        ),
        num_seqs=tensor("num_seqs", (1,), torch.int32),
        num_tokens=tensor("num_tokens", (1,), torch.int32),
        output=tensor(
            "output", (m, h, 128), strides=(*strides[9:], 1) if kda else None
        ),
    )
    if kda:
        data.update(
            raw_g=tensor("a", (m, h, 128), strides=(strides[1], strides[2], 1)),
            raw_beta=tensor("b", (m, h), strides=(strides[3], strides[4])),
        )
        cls = KdaBinding
    else:
        data.update(a=tensor("a", (m, h)), b=tensor("b", (m, h)))
        cls = Binding
    scratch = _TensorMetadata(torch.uint8, (1,), (1,), caps.device, 16)
    return cls(_state=layout, scratch=scratch, **data)


@program_cache(scope="preparation")
def compile_decode(query_payload, config_payload, ordinal):
    from . import _kernels as kernels

    query = GdnQuery(**dict(query_payload))
    config = GdnConfig.from_config(config_payload)
    layout = _scratch_layout(_caps(query, ordinal), config=config)
    b = _metadata_binding(query, layout)
    caps = layout.caps
    kda = caps.key_heads == caps.value_heads
    with torch.cuda.device(ordinal):
        commit = None
        if query.recover_speculative_state:
            from ._recovery import compile_recovery

            recurrent, commit = compile_recovery(query, config)
        elif kda:
            n, columns = b.state_indices.shape
            recurrent = kernels._packed_sequential_kda_decode_kernel.warmup(
                b.mixed_qkv,
                b.raw_g,
                b.raw_beta,
                b.A_log,
                b.dt_bias,
                b.recurrent_state,
                b.query_start_loc,
                b.num_accepted_tokens,
                b.state_indices,
                b.num_seqs,
                b.output,
                float(caps.key_head_dim**-0.5),
                -5.0,
                n,
                columns,
                stride_mixed_token=b.mixed_qkv.stride(0),
                stride_a_token=b.raw_g.stride(0),
                stride_a_head=b.raw_g.stride(1),
                stride_b_token=b.raw_beta.stride(0),
                stride_b_head=b.raw_beta.stride(1),
                stride_dt_bias_head=b.dt_bias.stride(0),
                stride_state_slot=b.recurrent_state.stride(0),
                stride_state_head=b.recurrent_state.stride(1),
                stride_state_v=b.recurrent_state.stride(2),
                stride_indices_request=b.state_indices.stride(0),
                stride_indices_column=b.state_indices.stride(1),
                stride_output_token=b.output.stride(0),
                stride_output_head=b.output.stride(1),
                MAX_SEQS=caps.max_seqs,
                KEY_HEADS=caps.key_heads,
                VALUE_HEADS=caps.value_heads,
                KEY_HEAD_DIM=128,
                VALUE_HEAD_DIM=128,
                STATE_INDEX_COLUMNS=caps.state_index_columns,
                BLOCK_V=config.recurrent_block_v,
                QK_L2NORM=caps.qk_l2norm,
                HAS_NULL_STATE_INDEX=caps.null_state_index is not None,
                NULL_STATE_INDEX=int(caps.null_state_index or 0),
                num_warps=layout.recurrent_num_warps,
                num_stages=3,
                grid=(triton.cdiv(128, config.recurrent_block_v), n * caps.value_heads),
            )
        else:
            from ._cute_kernels import _compile

            recurrent = _compile(b)[1]
        norm = kernels._gated_rmsnorm_kernel.warmup(
            b.output,
            b.z,
            b.norm_weight,
            b.num_tokens,
            1e-6,
            b.output.shape[0],
            stride_output_token=b.output.stride(0),
            stride_output_head=b.output.stride(1),
            stride_z_token=b.z.stride(0),
            stride_z_head=b.z.stride(1),
            VALUE_HEADS=caps.value_heads,
            VALUE_HEAD_DIM=128,
            SIGMOID_GATE=caps.gate_activation == "sigmoid",
            NORM_WEIGHT_FP32=b.norm_weight.dtype == torch.float32,
            KDA_NORM_FP32=kda,
            num_warps=layout.norm_num_warps,
            num_stages=1,
            grid=(b.output.shape[0] * caps.value_heads,),
        )
        return {"recurrent": recurrent, "norm": norm, "commit": commit}


def _binding_tensors(binding):
    kda = isinstance(binding, KdaBinding)
    return (
        binding.mixed_qkv,
        binding.raw_g if kda else binding.a,
        binding.raw_beta if kda else binding.b,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        binding.recurrent_state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        binding.output,
    )


@dataclass(frozen=True)
class _GdnState:
    query: GdnQuery
    layout: object
    recurrent: object
    norm: object
    commit: object = None
    _kda: bool = field(init=False)
    _has_null: bool = field(init=False)
    _null: int = field(init=False)
    _value_tiles: int = field(init=False)
    _sigmoid_gate: bool = field(init=False)
    _norm_fp32: bool = field(init=False)
    _alignments: tuple[int | None, ...] = field(init=False)
    _parameter_dtypes: tuple = field(init=False)

    def __post_init__(self):
        q = self.query
        values = {
            "_kda": q.key_heads == q.value_heads,
            "_has_null": q.null_state_index is not None,
            "_null": int(q.null_state_index or 0),
            "_value_tiles": triton.cdiv(128, self.layout.recurrent_block_v),
            "_sigmoid_gate": q.gate_activation == "sigmoid",
            "_norm_fp32": q.norm_weight_dtype == "float32",
            "_alignments": tuple(
                q.pointer_alignments[name]
                if name in aligned_operands(q.key_heads == q.value_heads)
                else None
                for name in _OPERANDS
            ),
            "_parameter_dtypes": tuple(
                (index, getattr(torch, dtype))
                for index, dtype in (
                    (4, q.a_log_dtype),
                    (5, q.dt_bias_dtype),
                    (6, q.norm_weight_dtype),
                    (7, q.state_dtype),
                    (10, q.state_indices_dtype),
                )
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def bind(self, **kwargs):
        if self._kda:
            raise ValueError("KDA plan requires bind_kda")
        binding = _bind(self.layout, **kwargs)
        self._check(_binding_tensors(binding))
        return binding

    def bind_kda(self, **kwargs):
        if not self._kda:
            raise ValueError("Qwen plan requires bind")
        binding = _bind_kda(self.layout, **kwargs)
        self._check(_binding_tensors(binding))
        return binding

    def _check(self, tensors):
        q = self.query
        for name, tensor, alignment in zip(
            _OPERANDS, tensors, self._alignments, strict=True
        ):
            if tensor.device != self.layout.caps.device:
                raise ValueError(
                    f"GDN {name} device {tensor.device} differs from the prepared {self.layout.caps.device}"
                )
            if (
                alignment is not None
                and not (q.recover_speculative_state and name == "b")
                and _alignment(tensor) != alignment
            ):
                raise ValueError(
                    f"GDN {name} pointer alignment {_alignment(tensor)} differs from the prepared {alignment}"
                )
        for index, dtype in self._parameter_dtypes:
            if tensors[index].dtype != dtype:
                raise ValueError(
                    f"GDN {_OPERANDS[index]} dtype {tensors[index].dtype} differs from the prepared {dtype}"
                )
        if self._kda:
            strides = _kda_strides(
                tensors[0],
                tensors[1],
                tensors[2],
                tensors[5],
                tensors[7],
                tensors[13],
            )
            actual, expected = strides, q.kda_strides
            if q.recover_speculative_state:
                actual = (*strides[:3], *strides[5:])
                expected = (*q.kda_strides[:3], *q.kda_strides[5:])
            if actual != expected:
                raise ValueError(
                    f"KDA compile-time strides {strides} differ from the prepared {q.kda_strides}"
                )

    def run(
        self, binding, *, eps=1e-6, scale=None, lower_bound=-5.0, apply_output_norm=True
    ):
        if binding._state != self.layout:
            raise ValueError("binding belongs to another GDN layout")
        eps, scale = float(eps), float(128**-0.5 if scale is None else scale)
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("GDN epsilon and scale must be finite and positive")
        if self._kda and (not math.isfinite(lower_bound) or lower_bound >= 0):
            raise ValueError("KDA lower bound must be finite and negative")
        if self.query.recover_speculative_state:
            self.recurrent(binding, scale=scale, lower_bound=float(lower_bound))
            if apply_output_norm:
                self._run_norm(
                    binding.output,
                    binding.z,
                    binding.norm_weight,
                    binding.num_tokens,
                    eps,
                )
            return binding.output
        self.run_tensors(
            *_binding_tensors(binding),
            eps=eps,
            scale=scale,
            lower_bound=float(lower_bound),
        )
        return binding.output

    def run_tensors(
        self,
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        recurrent_state,
        query_start_loc,
        num_accepted_tokens,
        state_indices,
        num_seqs,
        num_tokens,
        output,
        *,
        eps,
        scale,
        lower_bound,
    ):
        if self.query.recover_speculative_state:
            raise ValueError("KDA recovery requires a binding with record buffers")
        self._check(
            (
                mixed_qkv,
                a,
                b,
                z,
                A_log,
                dt_bias,
                norm_weight,
                recurrent_state,
                query_start_loc,
                num_accepted_tokens,
                state_indices,
                num_seqs,
                num_tokens,
                output,
            )
        )
        q, layout = self.query, self.layout
        n, columns = map(int, state_indices.shape)
        stride_r, stride_c = state_indices.stride()
        kda = self._kda
        if kda:
            self.recurrent[(self._value_tiles, n * q.value_heads, 1)](
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                recurrent_state,
                query_start_loc,
                num_accepted_tokens,
                state_indices,
                num_seqs,
                output,
                float(scale),
                float(lower_bound),
                n,
                columns,
                *q.kda_strides[:9],
                stride_r,
                stride_c,
                *q.kda_strides[9:],
                q.max_seqs,
                q.key_heads,
                q.value_heads,
                128,
                128,
                q.state_index_columns,
                layout.recurrent_block_v,
                q.qk_l2norm,
                self._has_null,
                self._null,
            )
        else:
            self.recurrent(
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                recurrent_state,
                query_start_loc,
                num_accepted_tokens,
                state_indices,
                num_seqs,
                output,
                float(scale),
            )
        self._run_norm(output, z, norm_weight, num_tokens, eps)

    def _run_norm(self, output, z, norm_weight, num_tokens, eps):
        q, m, kda = self.query, output.shape[0], self._kda
        self.norm[(m * q.value_heads, 1, 1)](
            output,
            z,
            norm_weight,
            num_tokens,
            float(eps),
            m,
            int(output.stride(0)),
            int(output.stride(1)),
            int(z.stride(0)),
            int(z.stride(1)),
            q.value_heads,
            128,
            self._sigmoid_gate,
            self._norm_fp32,
            kda,
        )


def plan(
    caps: Caps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: GdnConfig | None = None,
) -> Plan:
    if not isinstance(caps, Caps):
        raise TypeError("plan requires GDN Caps")
    invocation = FrozenMapping(invocation)
    query = _query_from_caps(caps, invocation)

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.sequence.gdn_decode._preparation:compile_decode",
                query.to_dict(),
                TUNING.encode_config(config),
                device.ordinal,
            ),
        )

    def memory(config, device):
        return MemoryRequirements(
            scratch=_scratch_layout(caps, config=config).scratch_specs()
        )

    def materialize(selection, device):
        config = selection.config
        programs = compile_decode(
            query.to_dict(), TUNING.encode_config(config), device.ordinal
        )
        return _GdnState(query, _scratch_layout(caps, config=config), **programs)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )
