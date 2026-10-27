# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The SDPA launch binder of the vendored cuDNN Frontend tree, in Python.

cuDNN Frontend binds its FROST SDPA launches through a nanobind extension (``cudnn._compiled_module._SdpaDenseBinder`` /
``_SdpaThdBinder``): per call it observes the operands' facts (address, dtype, device, span, shape, strides), validates them
against the plan's immutable specialization and assembles the compiled kernel's positional frame.  The vendored tree ships
no extension, so this module is that binder in Python.  The frame logic is cuDNN Frontend's own retired Python
implementation, vendored byte-faithfully as ``_cudnn_frost/sdpa/fwd/_binding_reference.py`` (``bind_dense``,
``bind_dense_split``, ``bind_thd``: the dense and packed operands, the paged pools and tables, the split combine); this file
adds what that implementation predates -- the per-call THD epilogue gate the EXTEND form binds -- plus the operand pack the
FE helpers read facts from and the two binder classes the vendored ``_compiled_module`` shim hands out.

It is the explicit, documented and only binder of the vendored tree (never a silent fallback, cuDNN Frontend's own rule),
and it serves the half (bf16 / f16) plans the Qwen 3.5 block declares; a quantized plan is a typed decline, and so is a split
plan with an attention sink: cuDNN Frontend's dense binder (its ``sdpa_dense_binding.cpp`` since the Rubin decode levers) folds
the sink in the split combine and appends the fp32 ``[H]`` sink pointer and, over padded query lengths, the int32 ``[B]``
Q-length address to the combine frame (``DenseLaunchSpec.combine_q_lens``); this binder binds neither slot and says so by
name (``supports_combine_q_lens`` is False, which the vendored ``prepared.py`` reads before it builds such a plan).  The Qwen
3.5 block has no attention sink, and on cc 10.7 the adapter keeps the sink-free dense split contract, so the decline is never
reached on the served device.

Each paged pool binds its OWN declared in-page layout (``paged_hnd`` for K, ``paged_v_hnd`` for V: cuDNN Frontend's independent
K / V physical layouts, which its native binders read the same way): the vendored frame implementation takes the V flag off the
spec, so a plan whose K pool is HND and V pool NHD (or the reverse) binds each pool by its own strides, and a runtime pool in the
other layout is refused by name (``compiled for HND / NHD``) before anything launches -- ``tests/experimental/qwen_3_5_block/
test_binder.py`` pins both directions for the dense and the packed frame.  FlashInfer's paged cache spells one ``kv_layout`` for
both pools, so the adapter declares the two pools alike; the binder serves either declaration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ._cudnn_frost.frost import buffers as _buffers
from ._cudnn_frost.sdpa.fwd import _binding_reference as _ref
from ._cudnn_frost.sdpa.fwd import prepared as _prepared

BufferFacts = _prepared.BufferFacts
_DLPACK_CUDA = _prepared._DLPACK_CUDA
_ALIGN_TMA = _prepared._ALIGN_TMA

__all__ = [
    "OperandPack",
    "SdpaDenseBinder",
    "SdpaThdBinder",
    "facts_of",
    "native_pack_from_facts",
    "read_buffer_sequence",
]


def _stream_int(stream) -> int:
    """The integer handle of a launch stream (``cuda.CUstream``, an int, a ``torch.cuda.Stream``); 0 = the default stream."""
    if stream is None:
        return 0
    try:
        return int(stream)
    except TypeError:
        return int(getattr(stream, "cuda_stream", 0))


def _compact_strides(shape: Sequence[int]) -> tuple:
    strides: List[int] = []
    acc = 1
    for n in reversed(list(shape)):
        strides.append(acc)
        acc *= int(n)
    return tuple(reversed(strides))


def facts_of(buf) -> Optional[Any]:
    """The :class:`BufferFacts` of one operand: a torch tensor (observed through ``facts_of_tensor``), the vendored shim's
    ``OperandBuffer`` (a workspace carve), or any DLPack / CUDA-array-interface producer; ``None`` stays ``None``."""
    if buf is None:
        return None
    if isinstance(buf, torch.Tensor):
        return _prepared.facts_of_tensor(buf)
    if (
        hasattr(buf, "ptr")
        and hasattr(buf, "shape")
        and hasattr(buf, "code")
        and hasattr(buf, "bits")
    ):
        shape = tuple(int(s) for s in buf.shape)
        n = 1
        for s in shape:
            n *= s
        dtype = _prepared._DTYPE_BY_CODE.get((int(buf.code), int(buf.bits)), "")
        return BufferFacts(
            int(buf.ptr),
            dtype,
            (_DLPACK_CUDA, int(buf.device_id)),
            n,
            shape,
            _compact_strides(shape),
        )
    if hasattr(buf, "__cuda_array_interface__"):
        return _prepared.facts_of_tensor(torch.as_tensor(buf))
    if hasattr(buf, "__dlpack__"):
        return _prepared.facts_of_tensor(torch.from_dlpack(buf))
    raise TypeError(
        f"an SDPA operand must be a torch.Tensor, a DLPack / CUDA-array-interface producer or a workspace carve; got {type(buf).__name__}"
    )


class OperandPack:
    """The standalone operand pack of one SDPA launch: the facts of every role, in role order, read once per call.

    The Python stand-in for the extension's ``VariantPackNative`` on the standalone route: ``_facts_as`` / ``is_filled`` /
    ``set_operand`` are what the vendored ``prepared.py`` helpers call on it."""

    __slots__ = ("_facts",)

    def __init__(self, facts: Sequence[Any]):
        self._facts: List[Any] = list(facts)

    def __len__(self) -> int:
        return len(self._facts)

    def is_filled(self, index: int) -> bool:
        i = int(index)
        return 0 <= i < len(self._facts) and self._facts[i] is not None

    def _facts_as(self, indices, cls=BufferFacts, by_code=None) -> List[Optional[Any]]:
        out = []
        for i in indices:
            f = self._facts[int(i)]
            out.append(None if f is None or cls is BufferFacts else cls(*f)) if (
                f is None or cls is not BufferFacts
            ) else out.append(f)
        return out

    def set_operand(
        self,
        index,
        ptr,
        shape,
        strides,
        code,
        bits,
        lanes,
        nbytes,
        device_type,
        device_id,
    ) -> None:
        dtype = _prepared._DTYPE_BY_CODE.get((int(code), int(bits)), "")
        width = max(1, (int(bits) + 7) // 8)
        span = -1 if nbytes is None or int(nbytes) < 0 else int(nbytes) // width
        self._facts[int(index)] = BufferFacts(
            int(ptr),
            dtype,
            (int(device_type), int(device_id)),
            span,
            tuple(int(s) for s in shape),
            tuple(int(s) for s in strides),
        )

    def role_facts(self, indices, roles) -> Dict[str, Any]:
        """``{role: facts}`` for the filled roles (an index of -1 or an unfilled slot = an absent optional role)."""
        out: Dict[str, Any] = {}
        for role, index in zip(roles, indices, strict=False):
            if index is None or int(index) < 0 or not self.is_filled(index):
                continue
            out[role] = self._facts[int(index)]
        return out


def read_buffer_sequence(buffers):
    """``(pack, unread)`` for a sequence of operands: every buffer is observed here, so ``unread`` is empty."""
    return OperandPack([facts_of(b) for b in buffers]), ()


def native_pack_from_facts(facts, roles, dtypes=None, itemsizes=None):
    """A pack from already-observed facts (padded to the role count)."""
    facts = list(facts)
    facts += [None] * (len(roles) - len(facts))
    return OperandPack(facts)


def _half_plan_only(spec, what: str) -> None:
    if getattr(spec, "quant", None) is not None:
        raise NotImplementedError(
            f"{what}: the Python SDPA binder of the vendored cuDNN Frontend tree serves the half (bf16 / f16) plans only; "
            "a quantized (FP8 / MXFP8) plan has no binder here"
        )


class SdpaDenseBinder:
    """The dense (padded) launch binder: ``execute(pack, indices, stream, scale_softmax_log2=None, workspace=0)``.

    Roles in ``prepared._NATIVE_DENSE_ROLES`` order; a split plan binds its fp32 partials in ``workspace`` and runs the
    combine (``spec.combine.fn``, which on a gate-in-combine plan is the FE ``CombineGate`` bound by the caller).  Every
    check the extension makes per call is made here by the vendored frame implementation, which shares the FE's cached
    geometry helpers (``_dense_role_layout``, ``_paged_pool_layout``, ...)."""

    supports_epilogue_gate = True
    supports_paged_epilogue_gate = True
    # The split attention-sink combine's frame slots (the extension appends the fp32 [H] sink pointer and, over padded query
    # lengths, the int32 [B] Q-length address: ``DenseLaunchSpec.combine_q_lens``) are not bound here: a split plan with a sink is
    # the typed decline below, and the vendored ``prepared.py`` reads this flag to decline the padded-Q form by name before it
    # builds the plan.  The vendored shim's ``_SdpaDenseBinder`` carries the same value (``vendor.py``, ``PYTHON_BINDER_FLAGS``).
    supports_combine_q_lens = False
    # The native D576x512 dense form of the extension's binder (the 7301fad21 pin) is not validated through this binder (the block's forms
    # are d256): False here and in the shim (``PYTHON_BINDER_FLAGS``), so the vendored ``prepared.py`` declines such a plan by name.
    supports_native_d576 = False
    # The quantized (fp8) split attention-sink combine of the extension's binder (the 977757763 pin): this binder declines a split plan
    # with a sink altogether (``supports_combine_q_lens`` above; the block declares no sink), so False here and in the shim, and the
    # vendored ``prepared.py`` / ``api_dsl.py`` decline such a plan by name.
    supports_quantized_split_sink = False

    def __init__(self, spec):
        _half_plan_only(spec, "SdpaDenseBinder")
        if getattr(spec, "ragged", False):
            raise NotImplementedError(
                "SdpaDenseBinder: the ragged-Q decode leg is not served by the Python binder"
            )
        if int(spec.split) < 1:
            raise ValueError("cudnn.sdpa: a dense plan needs split >= 1")
        if int(spec.split) > 1 and spec.combine is None:
            raise ValueError("cudnn.sdpa: a split plan needs a combine artifact")
        if int(spec.split) > 1 and (
            bool(getattr(spec, "has_sink", False))
            or bool(getattr(spec, "combine_q_lens", False))
        ):
            raise NotImplementedError(
                "SdpaDenseBinder: a split plan with an attention sink is not served by the Python binder (the split combine's "
                "sink pointer and per-batch Q-length frame slots are the native binder's); the Qwen 3.5 block declares no sink, "
                "and cc 10.7 dense plans keep the sink-free split contract, so this form is never declared on the served device"
            )
        self.spec = spec

    def execute(
        self, pack, indices, stream, scale_softmax_log2=None, workspace=0
    ) -> bool:
        spec = self.spec
        facts = pack.role_facts(indices, _prepared._NATIVE_DENSE_ROLES)
        stream_int = _stream_int(stream)
        combine_args = None
        if int(spec.split) > 1:
            bound = _ref.bind_dense_split(
                spec, facts, int(workspace), stream, stream_int
            )
            if bound is None:
                return False
            frame, combine_args = bound
        else:
            frame = _ref.bind_dense(spec, facts, stream, stream_int)
        if scale_softmax_log2 is not None:
            slot = (
                "scale_softmax_log2"
                if "scale_softmax_log2" in spec.index
                else "scale_softmax"
            )
            frame[spec.index[slot]] = scale_softmax_log2
        spec.fn(*frame)
        if combine_args is not None:
            spec.combine.fn(*combine_args)
        return True


class SdpaThdBinder:
    """The packed (THD) launch binder: ``execute(pack, indices, workspace, stream, scale=None, gate=-1)``.

    Roles in ``prepared._NATIVE_THD_ROLES`` order; the epilogue gate G of a gated plan arrives OUTSIDE the role pack by its
    own pack index (``gate``; -1 = none) and is bound like Q / O: Q's dtype, the plan's device, a 16-byte-aligned base, the
    declared ``(token, head, elem)`` strides in the packed ``(T, H, D)`` or the graph's ``(B, H, S, D)`` form, and a capacity
    covering every packed Q token the launch addresses.  A gate on an ungated plan, or none on a gated one, is refused."""

    supports_stats_stride_override = True
    supports_paged_packed_split = True
    supports_paged_split_sink = True
    supports_paged_d64_packed_split = True
    supports_paged_d256_packed_split = True
    supports_nonpaged_d256_packed_split = True
    supports_nonpaged_packed_split = True
    supports_nonpaged_d128_packed_split = True
    supports_epilogue_gate = True
    supports_paged_epilogue_gate = True
    # The 7301fad21 pin's THD binder flags: the wave policy / geometry (``ThdLaunchSpec.wave_policy`` / ``wave_geometry``) are host
    # arithmetic on ``n_thd_units`` inside the vendored frame implementation ``bind_thd`` delegates to, so they are served; the per-role
    # PHYSICAL ORIGINS (``ThdLaunchSpec.origin_specs``, bound from the ``thd_origins.ROLES`` offset operands) are not gathered by this
    # binder (``execute`` packs ``_NATIVE_THD_ROLES`` only), so every origins arm reads False here and in the shim (``PYTHON_BINDER_FLAGS``)
    # and the vendored ``prepared.py`` declines a plan with origins by name; the block's packed declarations carry none.
    supports_initial_waves = True
    supports_initial_wave_geometry = True
    supports_physical_origins = False
    supports_paged_physical_origins = False
    supports_paged_split_physical_origins = False
    supports_paged_fp8_physical_origins = False
    # The native D576x512 PACKED forms of the extension's THD binder (the 977757763 pin: the unsplit form, the packed split): not validated
    # through this binder (the block's packed forms are d256), so False here and in the shim (``PYTHON_BINDER_FLAGS``) and the vendored
    # ``engines.py`` / ``api_dsl.py`` decline such a plan by name, as the dense binder's ``supports_native_d576`` does.
    supports_native_d576 = False
    supports_native_d576_packed_split = False

    def __init__(self, spec):
        _half_plan_only(spec, "SdpaThdBinder")
        self.spec = spec
        self._gate_decl = None
        if spec.gate_expect is not None:
            decl = spec.decl.get("gate")
            if decl is None:
                raise ValueError(
                    "cudnn.sdpa: a gated THD plan declares no gate geometry"
                )
            if spec.gate_expect != spec.expect["q"]:
                raise ValueError(
                    "cudnn.sdpa: the THD epilogue gate is bound in Q's dtype"
                )
            if getattr(spec, "split_workspace", None) is not None:
                raise ValueError(
                    "cudnn.sdpa: the THD epilogue gate is not bound on a packed split plan (the partials bypass the gated epilogue)"
                )
            self._gate_decl = tuple(int(x) for x in decl)

    def execute(self, pack, indices, workspace, stream, scale=None, gate=-1) -> bool:
        spec = self.spec
        facts = pack.role_facts(indices, _prepared._NATIVE_THD_ROLES)
        g = pack._facts[int(gate)] if int(gate) >= 0 and pack.is_filled(gate) else None
        has_gate = spec.gate_expect is not None
        if has_gate and g is None:
            raise ValueError(
                "cudnn.sdpa: gate is required by this compiled specialization (constructed with sample_gate)"
            )
        if not has_gate and g is not None:
            raise ValueError(
                "cudnn.sdpa: this specialization was compiled without an epilogue gate"
            )
        gate_geo = self._gate_geometry(facts, g) if has_gate else None
        stream_int = _stream_int(stream)
        frame = _ref.bind_thd(spec, facts, int(workspace), stream, stream_int)
        if frame is not None and has_gate:
            ts, hs, row = gate_geo
            t_q = int(frame[spec.index["problem_size"]][3])
            cap = 0 if g.span < row else (g.span - row) // ts + 1
            if cap < t_q:
                raise ValueError(
                    "cudnn.sdpa: gate must cover the packed Q tokens: its buffer holds fewer rows than the bound Q / O"
                )
            frame[spec.index["gate_ptr"]] = g.ptr
            frame[spec.index["gate_strides"]] = (ts, ts, hs)
        _ref.initialize_thd_stats(spec, facts, stream_int)
        if frame is None:
            return False
        if scale is not None:
            frame[spec.index["scale_softmax_log2"]] = scale
        spec.fn(*frame)
        return True

    def _gate_geometry(self, facts, g: Any):
        """``(token stride, head stride, row span)`` of the gate under the plan's declared gate geometry (the C++ binder's
        ``resolve_declared``), validated before any launch."""
        spec = self.spec
        if g.dtype != spec.gate_expect:
            raise ValueError(
                f"cudnn.sdpa: gate: runtime buffer dtype {g.dtype} does not match its declaration (the gate is Q's dtype {spec.gate_expect})"
            )
        _ref._on_plan_device(spec, "gate", g)
        if g.ptr % _ALIGN_TMA:
            raise ValueError(
                "cudnn.sdpa: gate: runtime buffer base address must be 16-byte aligned (TMA global-address rule)"
            )
        if g.span < 0:
            raise ValueError(
                "cudnn.sdpa: gate was passed as a bare address; a ragged operand needs a sized buffer"
            )
        h, d, dts, dhs, des = self._gate_decl[:5]
        b = _ref.resolve_thd_geometry(spec, facts).b
        sh, st = tuple(int(x) for x in g.shape), tuple(int(x) for x in g.strides)
        if g.numel == 0:
            ts, hs, es = dts, dhs, des
        elif len(sh) == 4:
            if sh[0] != b or sh[1] != h or sh[3] != d:
                raise ValueError(
                    f"cudnn.sdpa: gate: effective shape {sh} must be ({b}, {h}, S, {d}) for this plan"
                )
            ts, hs, es = st[2], st[1], st[3]
        elif len(sh) == 3:
            if sh[1] != h or sh[2] != d:
                raise ValueError(
                    f"cudnn.sdpa: gate: a packed THD buffer is (T, {h}, {d}) for this plan; got {sh}"
                )
            ts, hs, es = st[0], st[1], st[2]
        else:
            raise ValueError(
                f"cudnn.sdpa: gate: a THD operand is (T, H, D) or the graph's (B, H, S, D); got rank {len(sh)}"
            )
        align = _ALIGN_TMA // _buffers.DTYPE_ITEMSIZE[spec.gate_expect]
        if es != 1:
            raise ValueError(
                f"cudnn.sdpa: gate: the head dim must be contiguous (elem stride 1); got {es}"
            )
        if hs < d or hs % align:
            raise ValueError(
                f"cudnn.sdpa: gate: head stride {hs} must cover {d} elements and be a 16-byte multiple"
            )
        row = (h - 1) * hs + d
        if ts < row or ts % align:
            raise ValueError(
                f"cudnn.sdpa: gate: token stride {ts} must cover the {h} heads and be a 16-byte multiple"
            )
        return ts, hs, row
