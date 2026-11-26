"""Test specs of the specialized MoE backend's kernels (data, not test modules).

Every kernel registered under
``flashinfer/fused_moe/backends/specialized/kernels/<name>/`` has a module
``<name>.py`` here that exports ``SPEC``, a :class:`KernelSpec` subclass
instance. ``tests/moe/test_unified_moe_specialized.py`` discovers the specs with
:func:`load_specs` (as the registry discovers the kernel subpackages) and runs
every contract test of the backend on each kernel; a spec's ``checks`` adds the
kernel's internal checks (layout, permutation, tie-breaking, ...). A new kernel
adds its spec module here and no line to the shared test file.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Check:
    """A kernel-internal check, run by ``test_kernel_check``.

    ``fn(spec, ctx)`` gets the spec and a context with ``monkeypatch`` and the
    entries of one ``params`` item as attributes. ``needs_kernel`` skips the
    check where the kernel does not run (``spec.available()``); ``skip``
    returns a reason to skip on this host, or ``None``.
    """

    name: str
    fn: Callable[..., None]
    params: Tuple[Tuple[str, Dict[str, Any]], ...] = (("", {}),)
    needs_kernel: bool = True
    skip: Optional[Callable[[], Optional[str]]] = None


@dataclass
class KernelSpec:
    """What the shared tests need to know about one kernel.

    Declarations are the expected values of the kernel class; the served path
    (``weights`` / ``inputs`` / ``pack``) builds the kernel's served
    configuration; the ``check_*`` hooks hold its tolerances and the facts
    that differ between kernels.
    """

    name: str
    kernel_module: ModuleType  # .../kernels/<name>/kernel.py
    package: ModuleType  # .../kernels/<name>/
    kernel_class: type
    sibling: str  # the weight view it reads = the reference backend's key
    sibling_config: Any  # that backend's config instance
    quant_variants: Tuple[Tuple[Any, Any], ...]
    routing_modes: Tuple[Any, ...]
    activation_classes: Tuple[type, ...]
    compute_capabilities: Tuple[int, ...]
    skip_reason: str
    # the module the kernel imports its programs from on first use, if lazy
    kernel_module_path: Optional[str] = None
    # host checks
    config_cases: Dict[str, Tuple[Dict[str, Any], Optional[str]]] = field(
        default_factory=dict
    )
    call_cases: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], str]] = field(
        default_factory=dict
    )
    meta_tokens: int = 1  # a served token count for meta calls
    meta_rejected_tokens: int = 3  # a token count outside the allowlist
    # the served path on the kernel's device
    numerics_cases: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    lazy_rejected_tokens: int = 3
    lazy_admitted_tokens: int = 1
    graph_tokens: int = 1
    compact_padded_tokens: Tuple[int, ...] = ()
    perf_cases: Tuple[Tuple[str, int], ...] = ()
    min_speedup: float = 1.2
    checks: Tuple[Check, ...] = ()

    # -- hooks every spec implements -------------------------------------
    def available(self) -> bool:
        raise NotImplementedError

    def config(self, **overrides) -> Any:
        """The served ``MoEConfig`` (default backend list) with overrides."""
        raise NotImplementedError

    def pin_device(self, monkeypatch, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Pin the device facts a config case names (keys starting with ``_``)
        and return the remaining config overrides."""
        raise NotImplementedError

    def meta_call(self, num_tokens: Optional[int] = None):
        """A served call (activation-pack kwargs, weight view) on meta tensors."""
        raise NotImplementedError

    def contract_violation(self) -> Dict[str, Any]:
        """Activation-pack overrides that ``check_contract`` rejects with
        ``input_layout``."""
        raise NotImplementedError

    def check_allowlist(self, allowlist: Mapping[str, Any], payload: Mapping) -> None:
        """The packaged allowlist (``workloads.json``) as the kernel loads it."""
        raise NotImplementedError

    def check_aot(self, monkeypatch) -> None:
        """What the kernel hands to ``flashinfer.aot``."""
        raise NotImplementedError

    def weights(self):
        """(canonical tensors, ``MoEWeightPack`` with the sibling's view only)."""
        raise NotImplementedError

    def inputs(self, num_tokens: int, kind: str = "serving", seed: int = 1):
        raise NotImplementedError

    def pack(self, *inputs) -> Any:
        raise NotImplementedError

    def check_numerics(self, inputs, out, stock, layered, on_kernel: bool) -> None:
        """The kernel alone (``out``), the reference backend alone (``stock``)
        and the default layer (``layered``, run on the kernel when
        ``on_kernel``) against the kernel's reference."""
        raise NotImplementedError

    def check_replay(self, inputs, replayed, eager, on_kernel: bool) -> None:
        """A CUDA-graph replay on ``inputs`` against the same backend run eagerly."""
        raise NotImplementedError

    # -- optional hooks -----------------------------------------------------
    def check_build_stats(self, stats: Mapping[str, Any]) -> None:
        """Extra facts about the stats after the first build."""

    def check_compact_padded(self, num_tokens: int) -> None:
        raise NotImplementedError


def load_specs() -> Dict[str, KernelSpec]:
    """Every kernel spec of this package, by kernel name."""
    specs: Dict[str, KernelSpec] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        spec = module.SPEC
        assert spec.name == info.name, (spec.name, info.name)
        specs[spec.name] = spec
    return specs


def spec_names() -> List[str]:
    return list(load_specs())
