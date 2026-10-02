"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Callable, Mapping

# Compute capability -> exported architecture name. The generated programs are
# exact per-architecture builds; no other capability maps onto them.
ARCHES: Mapping[tuple[int, int], str] = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

# A route's SM count can be read from the route record itself (``config.num_sms``
# or ``num_sms``) or, for families whose route key encodes it, through a
# caller-supplied ``route_num_sms(key, route)``.
RouteNumSms = Callable[[str, Mapping[str, Any]], int]


class UnsupportedDevice(NotImplementedError):
    """The catalog has no route for this device: wrong architecture or SM count.

    Every generated route is specialized for one exact physical SM count, so a
    SKU with another count (for example a cut-down part of the same
    architecture) has no program to run. The message names the device so the
    caller can act on it.
    """


def nvcc_flags(arch: str) -> list[str]:
    """The exact nvcc flag set of ``arch``."""
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


def jit_spec_name(name: str, arch: str) -> str:
    """JIT spec name of program ``name`` built for ``arch``.

    Per-architecture program names already carry the architecture
    (``..._sm100a_<id>``) and are kept verbatim so existing JIT caches stay
    valid; an architecture-neutral program name gets the ``_<arch>`` suffix so
    two architectures never share one cached library.
    """
    if arch.replace("_", "") in name:
        return name
    return f"{name}_{arch}"


class Catalog:
    """One family's exported-program catalog.

    Two layouts are read: the per-architecture layout
    (``{"schema", "arches": {arch: {"programs", "routes"}}}``) and the
    architecture-merged layout (``{"schema", "programs": {name: {...,
    "arches": [...]}}, "routes": {arch: {...}} | {...}}``) where one source
    compiles for every listed architecture. Nothing is hashed or re-verified at
    import; ``closure_sha256`` is a build receipt, not a load-time gate.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        label: str,
        path: Path | None = None,
        route_num_sms: RouteNumSms | None = None,
    ):
        if "schema" not in data:
            raise ValueError(f"{label} catalog has no 'schema' field")
        self._data = data
        self.label = label
        self.path = path
        self._route_num_sms = route_num_sms

    @property
    def schema(self) -> str:
        return self._data["schema"]

    @property
    def merged(self) -> bool:
        """True for the architecture-merged layout."""
        return "programs" in self._data

    @property
    def arches(self) -> tuple[str, ...]:
        if self.merged:
            arches: set[str] = set()
            for record in self._data["programs"].values():
                arches.update(record["arches"])
            return tuple(sorted(arches))
        return tuple(sorted(self._data["arches"]))

    def programs(self, arch: str) -> Mapping[str, Mapping[str, Any]]:
        """Program records compiled for ``arch`` (name -> record)."""
        self._require_arch(arch)
        if self.merged:
            return {
                name: record
                for name, record in self._data["programs"].items()
                if arch in record["arches"]
            }
        return self._data["arches"][arch]["programs"]

    def program(self, arch: str, name: str) -> Mapping[str, Any]:
        try:
            return self.programs(arch)[name]
        except KeyError as error:
            raise KeyError(
                f"{self.label} catalog has no program {name!r} for {arch}"
            ) from error

    def routes(self, arch: str) -> Mapping[str, Mapping[str, Any]]:
        """Route records of ``arch`` (route key -> route)."""
        self._require_arch(arch)
        if self.merged:
            routes = self._data["routes"]
            # Either keyed by architecture or shared by every architecture.
            return routes.get(arch, routes)
        return self._data["arches"][arch]["routes"]

    def route(self, arch: str, key: str, *, options: Any = None) -> Mapping[str, Any]:
        """The route under ``key``; ``UnsupportedDevice`` names the gap when absent."""
        try:
            return self.routes(arch)[key]
        except KeyError as error:
            shown = key if options is None else options
            raise UnsupportedDevice(
                f"No exported {self.label} program for {shown} on {arch}; "
                f"catalogued SM counts: {list(self.supported_num_sms(arch))}"
            ) from error

    def supported_num_sms(self, arch: str) -> tuple[int, ...]:
        """Physical SM counts with catalogued routes for ``arch``."""
        counts: set[int] = set()
        for key, route in self.routes(arch).items():
            counts.add(self._num_sms(key, route))
        return tuple(sorted(counts))

    def _num_sms(self, key: str, route: Mapping[str, Any]) -> int:
        if self._route_num_sms is not None:
            return int(self._route_num_sms(key, route))
        config = route.get("config")
        if isinstance(config, Mapping) and "num_sms" in config:
            return int(config["num_sms"])
        if "num_sms" in route:
            return int(route["num_sms"])
        raise ValueError(
            f"{self.label} route {key!r} carries no SM count; pass route_num_sms "
            "to read it from the route key"
        )

    def arch_for_capability(self, capability: tuple[int, int]) -> str:
        """Exported architecture of ``capability`` (RuntimeError when none is catalogued)."""
        arch = ARCHES.get(capability)
        if arch is None or arch not in self.arches:
            raise RuntimeError(
                f"{self.label} has no exported programs for compute capability "
                f"{capability}; catalogued architectures: {list(self.arches)}"
            )
        return arch

    def device_arch(self, device) -> str:
        """Exact architecture of the CUDA ``device`` (RuntimeError when none is catalogued)."""
        import torch

        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError(f"{self.label} requires a CUDA device")
        return self.arch_for_capability(tuple(torch.cuda.get_device_capability(device)))

    def device_num_sms(self, device, arch: str | None = None) -> int:
        """The device's physical SM count; ``UnsupportedDevice`` names the SKU when uncatalogued."""
        import torch

        device = torch.device(device)
        arch = self.device_arch(device) if arch is None else arch
        properties = torch.cuda.get_device_properties(device)
        sms = int(properties.multi_processor_count)
        supported = self.supported_num_sms(arch)
        if sms not in supported:
            raise UnsupportedDevice(
                f"{self.label}: the exported {arch} programs are specialized for "
                f"{list(supported)} physical SMs; {properties.name} ({device}) has {sms}"
            )
        return sms

    def jit_spec(self, arch: str, name: str):
        """JIT build specification of program ``name`` for the exact ``arch``."""
        from flashinfer.jit import env
        from flashinfer.jit.core import gen_jit_spec

        record = self.program(arch, name)
        return gen_jit_spec(
            name=jit_spec_name(name, arch),
            sources=[
                env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/")
                for p in record["sources"]
            ],
            extra_cuda_cflags=[
                *nvcc_flags(arch),
                *record["compile_flags"],
                "--device-entity-has-hidden-visibility=false",
            ],
            extra_ldflags=["-lcuda"],
            extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
            use_fast_math=False,  # only the recorded compile flags select math modes
        )

    def load_program(self, arch: str, name: str):
        """Build and load program ``name`` for ``arch``; cached per process."""
        return _load_program(self, arch, name)

    def _require_arch(self, arch: str) -> None:
        if arch not in self.arches:
            raise KeyError(
                f"{self.label} catalog has no architecture {arch!r}; "
                f"catalogued: {list(self.arches)}"
            )


@functools.cache
def _load_program(catalog: Catalog, arch: str, name: str):
    spec = catalog.jit_spec(arch, name)
    module = spec.build_and_load()
    return module, {
        **catalog.program(arch, name),
        "library_path": str(spec.get_library_path()),
    }


@functools.cache
def load_catalog(
    path: str | Path, *, label: str, route_num_sms: RouteNumSms | None = None
) -> Catalog:
    """Read the catalog at ``path`` once per process."""
    path = Path(path)
    return Catalog(
        json.loads(path.read_text()),
        label=label,
        path=path,
        route_num_sms=route_num_sms,
    )
