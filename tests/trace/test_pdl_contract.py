"""PDL ordering contract of the FlashInfer APIs.

With PDL a kernel may start while the preceding kernel still runs, so every
kernel behind an API must read its tensor parameters after
``griddepcontrol.wait`` (RAW) and write its buffers after it (WAR). The
exceptions are the parameters an API lists in
``@flashinfer_api(pdl_static=...)``, such as weights, which the preceding
kernel never writes; they are skipped. Set ``FLASHINFER_PDL_ASSUME_PRODUCED=1``
to test them too, for example to list the kernels that read weights before
the wait.

The registry below only says how to call each API; which operands are tested
comes from the API's signature. The harness is the one #6799 added for the
norms.
"""

import inspect
import os
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import pytest
import torch

import flashinfer
from flashinfer.trace.template import Tensor, TraceTemplate
from flashinfer.trace.templates import gemm as gemm_templates
from flashinfer.trace.templates import norm as norm_templates
from flashinfer.trace.templates import quantize as quantize_templates
from flashinfer.trace.templates.page import tgv_gemm_sm100_trace
from flashinfer.trace.templates.sampling import softmax_trace
from flashinfer.utils import BackendSupportedError
from tests.test_helpers.pdl_race import (
    PDL_REPEATS,
    PDL_SPIN_CYCLES,
    assert_bitwise,
    pdl_launcher,
    pdl_race_module,
    skip_unless_pdl_overlaps,
)

_ASSUME_PRODUCED = os.environ.get("FLASHINFER_PDL_ASSUME_PRODUCED", "0") not in (
    "",
    "0",
)


@dataclass(frozen=True)
class PdlApi:
    """How to call one API under PDL. The operands and their roles come from
    the API itself: its tensor parameters and ``@flashinfer_api(pdl_static=...)``."""

    name: str
    api: Callable[..., Any]
    # Builds the inputs (``init``) and names the in-place outputs.
    template: TraceTemplate
    axes: Tuple[Dict[str, int], ...]
    variants: Tuple[Dict[str, Any], ...] = ({},)
    # Tensor operands the template's init does not build, by parameter:
    # optional ones such as a bias, and scalars that may be passed as tensors.
    extra_inputs: Dict[str, Callable[[Dict[str, Any]], torch.Tensor]] = field(
        default_factory=dict
    )
    # Optional parameters these calls require.
    required: Tuple[str, ...] = ()

    @property
    def params(self):
        return inspect.signature(self.api, eval_str=True).parameters

    @property
    def pdl_param(self):
        (pdl_param,) = [p for p in self.params if "pdl" in p]
        return pdl_param

    @property
    def out_param(self):
        return "out" if "out" in self.params else None

    def operands(self):
        """Tensor parameters, each with its declared PDL role."""
        return {
            p: "static" if p in self.api.pdl_static else "produced"
            for p, param in self.params.items()
            if _takes_tensor(param.annotation) and p != self.out_param
        }

    def optional_operands(self):
        """Optional operands these calls can pass."""
        provided = {d.param or n for n, d in self.template.inputs.items()}
        provided |= self.extra_inputs.keys()
        return [
            p
            for p in self.operands()
            if self.params[p].default is None
            and p in provided
            and p not in self.required
        ]

    def written_params(self):
        """Buffers the call writes that exist before it: ``out`` and in-place
        operands."""
        written = [self.out_param] if self.out_param else []
        return written + [p for p in self.operands() if p in self.template.outputs]


def _takes_tensor(annotation):
    return annotation is torch.Tensor or torch.Tensor in typing.get_args(annotation)


def _bias(inputs, n_dim):
    n = inputs[n_dim[0]].shape[n_dim[1]]
    return torch.randn(n, dtype=torch.bfloat16, device="cuda")


def _scale(inputs):
    return torch.rand(1, dtype=torch.float32, device="cuda") + 0.5


def _row_scale(inputs):
    batch_size = inputs["logits"].shape[0]
    return torch.rand(batch_size, dtype=torch.float32, device="cuda") + 0.5


_LOW_M = (dict(M=1), dict(M=8), dict(M=32))

_PDL_APIS = [
    PdlApi(
        "mm_bf16",
        flashinfer.mm_bf16,
        gemm_templates.mm_bf16_trace,
        axes=tuple(
            dict(M=m, N=n, K=k)
            for m, n, k in ((1, 768, 7168), (8, 4096, 4096), (32, 1024, 7168))
        ),
        # The backends whose docs say they honor pdl.
        variants=tuple({"backend": b} for b in ("tgv", "tinygemm", "cute-dsl")),
        extra_inputs={"bias": lambda inputs: _bias(inputs, ("b", 1))},
    ),
    PdlApi(
        "mm_fp4",
        flashinfer.mm_fp4,
        gemm_templates.mm_fp4_trace,
        axes=tuple(dict(M=m, N=2048, K=4096) for m in (8, 128)),
        variants=tuple({"backend": b} for b in ("cudnn", "cutlass", "cute-dsl")),
        extra_inputs={"alpha": _scale},
    ),
    *(
        PdlApi(
            template.name_prefix,
            flashinfer.mm_bf16_fp4,
            template,
            axes=_LOW_M,
        )
        for template in (
            gemm_templates.mm_bf16_fp4_cudnn_trace,
            gemm_templates.mm_bf16_fp4_cute_dsl_trace,
            gemm_templates.mm_bf16_fp4_cute_dsl_sm100_trace,
            gemm_templates.mm_bf16_fp4_native_trace,
        )
    ),
    PdlApi(
        "tinygemm_bf16",
        flashinfer.gemm.tinygemm_bf16,
        gemm_templates.tinygemm_bf16_trace,
        axes=tuple(dict(M=m, N=4096, K=4096) for m in (1, 8)),
    ),
    PdlApi(
        "tgv_gemm_sm100",
        flashinfer.tgv_gemm_sm100,
        tgv_gemm_sm100_trace,
        axes=_LOW_M,
    ),
    PdlApi(
        "mm_M1_16_K7168_N256",
        flashinfer.gemm.mm_M1_16_K7168_N256,
        gemm_templates.mm_M1_16_K7168_N256_trace,
        axes=(dict(M=1), dict(M=16)),
    ),
    PdlApi(
        "mm_M1_16_K6144_N256",
        flashinfer.gemm.mm_M1_16_K6144_N256,
        gemm_templates.mm_M1_16_K6144_N256_trace,
        axes=(dict(M=1), dict(M=16)),
    ),
    PdlApi(
        "fp4_quantize",
        flashinfer.fp4_quantize,
        quantize_templates.fp4_quantize_trace,
        axes=tuple(dict(M=m, K=4096) for m in (1, 128)),
        # Optional only for UE8M0 scales.
        required=("global_scale",),
    ),
    PdlApi(
        "nvfp4_quantize",
        flashinfer.nvfp4_quantize,
        quantize_templates.nvfp4_quantize_trace,
        axes=tuple(dict(M=m, K=4096) for m in (1, 128)),
    ),
    PdlApi(
        "rmsnorm_quant",
        flashinfer.rmsnorm_quant,
        norm_templates.rmsnorm_quant_trace,
        axes=tuple(dict(batch_size=b, hidden_size=4096) for b in (1, 33)),
        extra_inputs={"scale": _scale},
    ),
    PdlApi(
        "fused_add_rmsnorm_quant",
        flashinfer.fused_add_rmsnorm_quant,
        norm_templates.fused_add_rmsnorm_quant_trace,
        axes=tuple(dict(batch_size=b, hidden_size=4096) for b in (1, 33)),
        extra_inputs={"scale": _scale},
    ),
    PdlApi(
        "softmax",
        flashinfer.softmax,
        softmax_trace,
        axes=tuple(dict(batch_size=b, vocab_size=32000) for b in (1, 8)),
        extra_inputs={"temperature": _row_scale},
    ),
]

# Kernels known to break the declared contract: (api, backend, operand) -> why.
# Strict, so a fix shows up as XPASS and its entry has to go.
_KNOWN_RACES = {
    ("mm_bf16", "tinygemm", "bias"): "tinygemm2 compute warps skip the wait",
    ("tinygemm_bf16", None, "bias"): "tinygemm2 compute warps skip the wait",
    ("mm_bf16", "cute-dsl", "bias"): "fixed by #7591",
    ("mm_fp4", "cutlass", "alpha"): "NVIDIA/cutlass#3786",
    ("mm_bf16_fp4_cute_dsl_sm100", None, "alpha"): "fixed by #7288",
}
# Does not reproduce on SM120 with these shapes.
_KNOWN_RACES_SM10X = {
    ("mm_fp4", "cute-dsl", "alpha"): "fixed by #7288",
}


@dataclass(frozen=True)
class Case:
    spec: PdlApi
    variant: Dict[str, Any]
    axes: Dict[str, int]
    with_optional: bool

    @property
    def backend(self):
        return self.variant.get("backend")

    @property
    def id(self):
        parts = [self.spec.name, *map(str, self.variant.values())]
        parts.append("".join(f"{k}{v}" for k, v in self.axes.items()))
        parts.append("all" if self.with_optional else "required")
        return "-".join(parts)


def _cases():
    for spec in _PDL_APIS:
        for variant in spec.variants:
            for axes in spec.axes:
                for with_optional in (
                    (False, True) if spec.optional_operands() else (True,)
                ):
                    yield Case(spec, variant, axes, with_optional)


_CASES = list(_cases())


def _storage_bytes(t):
    """The whole storage behind ``t`` as one contiguous uint8 tensor, so that
    strided views such as a column-major weight can be poisoned and copied."""
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())


class _Call:
    """One API call whose every run starts from the same inputs."""

    def __init__(self, case):
        spec = case.spec
        self.case = case
        self.inputs = spec.template.init(**case.axes)
        for param, build in spec.extra_inputs.items():
            self.inputs[param] = build(self.inputs)
        if not case.with_optional:
            for param in spec.optional_operands():
                self.inputs.pop(param, None)
        self._initial = {
            p: v.clone() for p, v in self.inputs.items() if isinstance(v, torch.Tensor)
        }
        self.result = spec.api(**self.inputs, **case.variant)
        if spec.out_param is not None and self.inputs.get(spec.out_param) is None:
            self.inputs[spec.out_param] = self.result
        self.written = [self.inputs[p] for p in spec.written_params()]
        torch.cuda.synchronize()

    def __call__(self):
        spec = self.case.spec
        self.result = spec.api(
            **self.inputs, **self.case.variant, **{spec.pdl_param: True}
        )

    def reset(self):
        """Restore in-place operands and clear ``out``, so that a run which
        does not write it fails."""
        for param, value in self._initial.items():
            self.inputs[param].copy_(value)
        if self.case.spec.out_param is not None:
            _storage_bytes(self.inputs[self.case.spec.out_param]).fill_(0x11)

    def outputs(self):
        returned = self.result if isinstance(self.result, tuple) else (self.result,)
        return self.written + [t for t in returned if isinstance(t, torch.Tensor)]


def _call_or_skip(case):
    try:
        return _Call(case)
    # TypeError: a backend whose binding requires an operand the template
    # marks optional, such as alpha for the CUTLASS FP4 GEMM.
    except (BackendSupportedError, NotImplementedError, TypeError, ValueError) as exc:
        pytest.skip(f"{case.id} unavailable here: {exc}")


def _expect_known_race(request, case, operand):
    races = _KNOWN_RACES
    if torch.cuda.get_device_capability()[0] == 10:
        races = races | _KNOWN_RACES_SM10X
    reason = races.get((case.spec.name, case.backend, operand))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))


def _read_cases():
    for case in _CASES:
        optional = case.spec.optional_operands()
        for param in case.spec.operands():
            if param in optional and not case.with_optional:
                continue
            yield pytest.param(case, param, id=f"{case.id}-{param}")


@pytest.mark.parametrize("spec", _PDL_APIS, ids=lambda spec: spec.name)
def test_pdl_contract_operands_match_declarations(spec):
    """A tensor parameter without a tensor annotation would go untested, and a
    misspelled ``pdl_static`` entry would declare nothing."""
    operands = spec.operands().keys()
    tensors = {
        d.param or name
        for name, d in spec.template.inputs.items()
        if isinstance(d, Tensor)
    }
    assert tensors - {spec.out_param} <= operands
    assert spec.api.pdl_static <= operands


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_pdl_contract_probe_sees_overlap(mode):
    """Calibration: the tests below can only fail where a PDL dependent starts
    while the producer spins."""
    skip_unless_pdl_overlaps(mode)


@pytest.mark.parametrize(("case", "operand"), list(_read_cases()))
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_pdl_contract_waits_before_reading(request, case, operand, mode):
    """RAW: a produced operand written late by the preceding kernel must not
    be read early."""
    if case.spec.operands()[operand] == "static" and not _ASSUME_PRODUCED:
        pytest.skip(f"{operand} is declared static; it may be read before the wait")
    skip_unless_pdl_overlaps(mode)
    call = _call_or_skip(case)
    if call.inputs.get(operand) is None:
        pytest.skip(f"{operand} is not passed in this case")
    module = pdl_race_module()

    call.reset()
    call()
    torch.cuda.synchronize()
    expected = [t.clone() for t in call.outputs()]

    call.reset()
    dst = _storage_bytes(call.inputs[operand])
    src = dst.clone()

    def fn():
        module.pdl_trigger_spin_copy([dst], [src], PDL_SPIN_CYCLES)
        call()

    run = pdl_launcher(mode, fn)
    _expect_known_race(request, case, operand)
    for rep in range(PDL_REPEATS):
        call.reset()
        # 0xFF bytes are NaN in every floating-point and scale format.
        dst.fill_(0xFF)
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        for actual, want in zip(call.outputs(), expected, strict=True):
            assert_bitwise(actual, want, f"{case.id} {operand} read early, run {rep}")


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_pdl_contract_waits_before_writing(case, mode):
    """WAR: a buffer the preceding kernel reads late must not be overwritten
    early."""
    if not case.spec.written_params():
        pytest.skip(f"{case.spec.name} writes only buffers it allocates")
    skip_unless_pdl_overlaps(mode)
    call = _call_or_skip(case)
    module = pdl_race_module()

    buffers = [_storage_bytes(t) for t in call.written]
    snapshots = [torch.empty_like(b) for b in buffers]

    def fn():
        module.pdl_trigger_spin_copy(snapshots, buffers, PDL_SPIN_CYCLES)
        call()

    run = pdl_launcher(mode, fn)
    for rep in range(PDL_REPEATS):
        call.reset()
        before = [b.clone() for b in buffers]
        for snapshot in snapshots:
            snapshot.zero_()
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        for snapshot, want in zip(snapshots, before, strict=True):
            assert_bitwise(snapshot, want, f"{case.id} overwritten early, run {rep}")
