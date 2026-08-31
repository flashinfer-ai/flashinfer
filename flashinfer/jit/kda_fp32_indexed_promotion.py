"""Fail-closed runtime for a promoted FP32 indexed KDA prefill program.

The checked-in manifest selects exactly one representation: generated CUDA
sources or target-specific cubins.  Both paths verify the complete declared
file closure before making a program available.  An explicit pending envelope
keeps source-only development fail closed when no promoted payload is installed.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Literal

from . import env as jit_env

PromotionMode = Literal["cuda", "cubin"]
PromotionTarget = Literal["sm100a", "sm103a"]

MANIFEST_KIND = "flashinfer.kda_fp32_indexed_promotion"
PACK_KIND = "flashinfer.generated_program_pack"
MANIFEST_FILENAME = "kda_fp32_indexed_promotion_manifest.json"
SCHEMA_VERSION = 1
TARGETS: tuple[PromotionTarget, ...] = ("sm100a", "sm103a")
DISPATCHER_ABI = "flashinfer.generated_program_dispatcher.v1"
DISPATCHER_ABI_ATTRIBUTE = "FLASHINFER_DISPATCHER_ABI"
DISPATCHER_MODULE_IDS_ATTRIBUTE = "FLASHINFER_MODULE_IDS"
DISPATCHER_BIND_ENTRYPOINT = "bind_loaded_modules"
DISPATCHER_SELECT_ARGUMENTS = (
    "gpu_arch",
    "sm_count",
    "fixed_layout",
    "sequence_lengths",
    "num_heads",
    "use_initial_state",
    "store_final_state",
)
DISPATCHER_RUN_ARGUMENTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "A_log",
    "dt_bias",
    "scale",
    "initial_state",
    "output_final_state",
    "lower_bound",
    "cu_seqlens",
    "output",
    "seq_order",
    "prefill_workspace",
    "state_indices",
    "state_checkpoints",
    "checkpoint_cu_starts",
    "checkpoint_every_n_tokens",
)

RUNTIME_CONTRACT = {
    "A_log_dtype": "float32",
    "beta_dtype": "bfloat16",
    "beta_is_logit": True,
    "checkpoint_mode": "none",
    "dt_bias_dtype": "float32",
    "gate_kind": "softplus",
    "head_dim": 128,
    "head_relationship": "equal_q_kv",
    "initial_state": "indexed_float32_pool",
    "operation": "recurrent_kda_prefill",
    "output_dtype": "bfloat16",
    "qkv_dtype": "bfloat16",
    "targets": ["sm100a", "sm103a"],
}

_COMPUTE_CAPABILITY_TO_TARGET: dict[tuple[int, int], PromotionTarget] = {
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}
_ROOT_KEYS = {
    "contract_denominators",
    "entries",
    "kind",
    "mode",
    "name",
    "schema_version",
}
_PENDING_ROOT_KEYS = {
    "arguments",
    "contract",
    "entries",
    "entry_point",
    "kind",
    "mode",
    "module_ident",
    "schema_version",
    "status",
}
_ENTRY_KEYS = {
    "architecture",
    "artifact_root",
    "artifacts",
    "route_count",
    "route_denominator_sha256",
    "runtime_inventory",
    "runtime_inventory_identity",
    "target",
}
_PACK_ARTIFACT_KEYS = {"executable", "id", "kind", "path", "sha256", "size_bytes"}
_INVENTORY_KEYS = {
    "architecture",
    "contract",
    "dispatcher",
    "dispatcher_seed_identity",
    "mode",
    "modules",
    "routes",
    "schema_version",
    "seeds",
}
_DISPATCHER_KEYS = {
    "artifact_id",
    "run_entrypoint",
    "select_entrypoint",
    "sha256",
}
_MODULE_KEYS = {
    "build_output",
    "cubin",
    "entry_point",
    "host",
    "id",
    "module_ident",
    "recipe",
    "shared_library",
    "source",
}
_REF_KEYS = {"artifact_id", "sha256"}
_BUILD_OUTPUT_KEYS = {"id", "path", "sha256", "size_bytes"}
_SEED_KEYS = {"id"}
_ROUTE_KEYS = {"id", "module_ids", "seed_id", "selector"}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class PromotionManifestError(RuntimeError):
    """The installed runtime manifest or one of its artifacts is invalid."""


@dataclass(frozen=True)
class _InstalledArtifact:
    artifact_id: str
    kind: str
    path: Path
    sha256: str
    size_bytes: int
    executable: bool


@dataclass(frozen=True)
class PromotionModuleSpec:
    """One exact runtime module within a target program."""

    build_output: dict[str, object] | None
    cubin: _InstalledArtifact | None
    entry_point: str
    host: _InstalledArtifact | dict[str, str]
    module_id: str
    mode: PromotionMode
    module_ident: str
    recipe: _InstalledArtifact | None
    shared_library: _InstalledArtifact | dict[str, str]
    source: _InstalledArtifact | None


@dataclass(frozen=True)
class PromotionTargetSpec:
    """Verified dispatcher, module closure, and routes for one architecture."""

    artifact_root: Path
    dispatcher: _InstalledArtifact
    dispatcher_run_entrypoint: str
    dispatcher_select_entrypoint: str
    identity: str
    mode: PromotionMode
    modules: tuple[PromotionModuleSpec, ...]
    routes: tuple[dict[str, object], ...]
    seeds: tuple[dict[str, object], ...]
    target: PromotionTarget


@dataclass(frozen=True)
class _LoadedProgram:
    dispatcher: _BoundDispatcher
    modules: Mapping[str, Callable[..., object]]
    spec: PromotionTargetSpec


@dataclass(frozen=True)
class _BoundDispatcher:
    prepare: Callable[..., object]
    select: Callable[..., object]


class Prepared:
    """One bound dispatcher closure whose launch path takes no arguments."""

    __slots__ = ("_close", "_closed", "_launch")

    def __init__(self, value: object) -> None:
        try:
            launch = value.launch
            close = value.close
        except Exception as exc:
            raise PromotionManifestError(
                "invalid FP32 indexed KDA promotion manifest: dispatcher prepare "
                "result does not expose launch/close"
            ) from exc
        self._launch = _require_exact_keyword_callable(
            launch,
            argument_names=(),
            label="prepared dispatcher launch",
        )
        self._close = _require_exact_keyword_callable(
            close,
            argument_names=(),
            label="prepared dispatcher close",
        )
        self._closed = False

    def launch(self) -> object:
        """Launch the already-selected, already-bound program closure."""

        if self._closed:
            raise RuntimeError("prepared FP32 indexed KDA promotion is closed")
        return self._launch()

    def close(self) -> None:
        """Release retained resources; repeated closes are harmless."""

        if self._closed:
            return
        close = self._close
        self._closed = True
        self._launch = None
        self._close = None
        close()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionManifestError(
            f"invalid FP32 indexed KDA promotion manifest: {message}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionManifestError(
                f"invalid FP32 indexed KDA promotion manifest: duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer KDA CUDA sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _manifest_path() -> Path:
    return _get_csrc_dir() / MANIFEST_FILENAME


def _is_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and (value[0].isalpha() or value[0] == "_")
        and all(character.isalnum() or character == "_" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    return _get_csrc_dir().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: object, label: str) -> str:
    _require(_is_identifier(value), f"{label} must be a C identifier")
    assert isinstance(value, str)
    return value


def _safe_id(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None,
        f"{label} must be a safe identifier",
    )
    assert isinstance(value, str)
    return value


def _sha256_value(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be one full lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _installed_artifact(
    value: object, *, artifact_root: Path, label: str
) -> _InstalledArtifact:
    _require(
        isinstance(value, dict) and set(value) == _PACK_ARTIFACT_KEYS,
        f"{label} envelope is invalid",
    )
    assert isinstance(value, dict)
    artifact_id = _safe_id(value.get("id"), f"{label} id")
    raw_path = value.get("path")
    _require(isinstance(raw_path, str), f"{label} path must be a string")
    assert isinstance(raw_path, str)
    relative = PurePosixPath(raw_path)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == raw_path
        and ".." not in relative.parts,
        f"{label} path must be normalized and relative",
    )
    root_relative = PurePosixPath(
        artifact_root.relative_to(_repository_root()).as_posix()
    )
    _require(
        relative.parts[: len(root_relative.parts)] == root_relative.parts,
        f"{label} escapes its target root",
    )
    path = _repository_root().joinpath(*relative.parts)
    _require(
        path.is_file() and not path.is_symlink(), f"{label} is missing or a symlink"
    )
    digest = _sha256_value(value.get("sha256"), f"{label} sha256")
    size = value.get("size_bytes")
    executable = value.get("executable")
    kind = value.get("kind")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label} size is invalid",
    )
    _require(isinstance(executable, bool), f"{label} executable is invalid")
    _require(isinstance(kind, str) and bool(kind), f"{label} kind is invalid")
    _require(
        (_sha256(path), path.stat().st_size) == (digest, size), f"{label} bytes drifted"
    )
    _require(
        bool(path.stat().st_mode & 0o111) == executable,
        f"{label} executable mode drifted",
    )
    return _InstalledArtifact(
        artifact_id=artifact_id,
        kind=kind,
        path=path,
        sha256=digest,
        size_bytes=size,
        executable=executable,
    )


def _artifact_ref(
    value: object,
    *,
    artifacts: dict[str, _InstalledArtifact],
    label: str,
    installed: bool,
) -> _InstalledArtifact | dict[str, str]:
    _require(
        isinstance(value, dict) and set(value) == _REF_KEYS,
        f"{label} reference is invalid",
    )
    assert isinstance(value, dict)
    artifact_id = _safe_id(value.get("artifact_id"), f"{label} artifact id")
    digest = _sha256_value(value.get("sha256"), f"{label} sha256")
    artifact = artifacts.get(artifact_id)
    if installed:
        _require(
            artifact is not None and artifact.sha256 == digest,
            f"{label} is not an installed exact artifact",
        )
        assert artifact is not None
        return artifact
    if artifact is not None:
        _require(artifact.sha256 == digest, f"{label} installed identity differs")
    return {"artifact_id": artifact_id, "sha256": digest}


def _optional_ref(
    value: object,
    *,
    artifacts: dict[str, _InstalledArtifact],
    label: str,
    installed: bool,
) -> _InstalledArtifact | dict[str, str] | None:
    if value is None:
        return None
    return _artifact_ref(value, artifacts=artifacts, label=label, installed=installed)


def _parse_target(
    value: object, *, mode: PromotionMode, index: int
) -> PromotionTargetSpec:
    label = f"entries[{index}]"
    _require(
        isinstance(value, dict) and set(value) == _ENTRY_KEYS,
        f"{label} envelope is invalid",
    )
    assert isinstance(value, dict)
    target = value.get("target")
    _require(target in TARGETS, f"{label}.target must be one of {list(TARGETS)}")
    assert target in TARGETS
    expected_architecture = "sm_" + target.removeprefix("sm")
    _require(
        value.get("architecture") == expected_architecture,
        f"{label} architecture differs from target",
    )
    raw_root = value.get("artifact_root")
    _require(isinstance(raw_root, str), f"{label} artifact_root must be a path")
    assert isinstance(raw_root, str)
    root_relative = PurePosixPath(raw_root)
    _require(
        raw_root not in ("", ".", "..")
        and "\\" not in raw_root
        and not root_relative.is_absolute()
        and root_relative.as_posix() == raw_root
        and ".." not in root_relative.parts,
        f"{label} artifact_root must be a normalized relative path",
    )
    artifact_root = _repository_root().joinpath(*root_relative.parts)
    _require(
        artifact_root.is_dir() and not artifact_root.is_symlink(),
        f"{label} artifact root is unavailable",
    )
    raw_artifacts = value.get("artifacts")
    _require(
        isinstance(raw_artifacts, list) and bool(raw_artifacts),
        f"{label} artifact inventory is empty",
    )
    artifacts_list = [
        _installed_artifact(
            item, artifact_root=artifact_root, label=f"{label} artifact {item_index}"
        )
        for item_index, item in enumerate(raw_artifacts)
    ]
    artifacts = {artifact.artifact_id: artifact for artifact in artifacts_list}
    _require(len(artifacts) == len(artifacts_list), f"{label} artifact ids repeat")

    inventory = value.get("runtime_inventory")
    _require(
        isinstance(inventory, dict) and set(inventory) == _INVENTORY_KEYS,
        f"{label} runtime inventory is invalid",
    )
    assert isinstance(inventory, dict)
    _require(
        value.get("runtime_inventory_identity") == _content_identity(inventory),
        f"{label} inventory identity is invalid",
    )
    _require(
        inventory.get("schema_version") == 1
        and inventory.get("architecture") == expected_architecture
        and inventory.get("mode") == mode
        and inventory.get("contract") == RUNTIME_CONTRACT,
        f"{label} runtime identity differs from the selected public contract",
    )

    dispatcher = inventory.get("dispatcher")
    _require(
        isinstance(dispatcher, dict) and set(dispatcher) == _DISPATCHER_KEYS,
        f"{label} dispatcher is invalid",
    )
    assert isinstance(dispatcher, dict)
    dispatcher_artifact = _artifact_ref(
        {key: dispatcher[key] for key in _REF_KEYS},
        artifacts=artifacts,
        label=f"{label} dispatcher",
        installed=True,
    )
    assert isinstance(dispatcher_artifact, _InstalledArtifact)
    _require(
        dispatcher_artifact.path.suffix == ".py",
        f"{label} dispatcher must be Python source",
    )
    raw_modules = inventory.get("modules")
    _require(
        isinstance(raw_modules, list) and bool(raw_modules),
        f"{label} modules must be non-empty",
    )
    modules: list[PromotionModuleSpec] = []
    for module_index, raw_module in enumerate(raw_modules):
        module_label = f"{label} module {module_index}"
        _require(
            isinstance(raw_module, dict) and set(raw_module) == _MODULE_KEYS,
            f"{module_label} is invalid",
        )
        assert isinstance(raw_module, dict)
        source = _optional_ref(
            raw_module.get("source"),
            artifacts=artifacts,
            label=f"{module_label} source",
            installed=mode == "cuda",
        )
        host = _artifact_ref(
            raw_module.get("host"),
            artifacts=artifacts,
            label=f"{module_label} host",
            installed=mode == "cuda",
        )
        cubin = _optional_ref(
            raw_module.get("cubin"),
            artifacts=artifacts,
            label=f"{module_label} cubin",
            installed=mode == "cubin",
        )
        recipe = _optional_ref(
            raw_module.get("recipe"),
            artifacts=artifacts,
            label=f"{module_label} recipe",
            installed=mode == "cuda",
        )
        shared = _artifact_ref(
            raw_module.get("shared_library"),
            artifacts=artifacts,
            label=f"{module_label} shared library",
            installed=mode == "cubin",
        )
        build_output = raw_module.get("build_output")
        if mode == "cuda":
            _require(
                isinstance(host, _InstalledArtifact),
                f"{module_label} host shim is unavailable",
            )
            assert isinstance(host, _InstalledArtifact)
            _require(
                host.path.suffix in (".cc", ".cpp"),
                f"{module_label} host shim has the wrong suffix",
            )
            _require(
                isinstance(source, _InstalledArtifact),
                f"{module_label} CUDA source is unavailable",
            )
            _require(
                isinstance(recipe, _InstalledArtifact),
                f"{module_label} build recipe is unavailable",
            )
            _require(
                cubin is not None and not isinstance(cubin, _InstalledArtifact),
                f"{module_label} expected cubin evidence is invalid",
            )
            _require(
                isinstance(build_output, dict)
                and set(build_output) == _BUILD_OUTPUT_KEYS,
                f"{module_label} build output is invalid",
            )
            assert isinstance(build_output, dict)
            assert isinstance(source, _InstalledArtifact)
            assert isinstance(recipe, _InstalledArtifact)
            _require(source.path.suffix == ".cu", f"{module_label} source must be CUDA")
            _require(
                recipe.path.suffix == ".py", f"{module_label} recipe must be Python"
            )
            _require(recipe.executable, f"{module_label} recipe must be executable")
            _safe_id(build_output.get("id"), f"{module_label} build output id")
            _sha256_value(
                build_output.get("sha256"), f"{module_label} build output sha256"
            )
            _require(
                build_output.get("sha256") == cubin["sha256"],
                f"{module_label} build and cubin identities differ",
            )
            output_path = build_output.get("path")
            _require(
                isinstance(output_path, str),
                f"{module_label} build output path is invalid",
            )
            assert isinstance(output_path, str)
            output_relative = PurePosixPath(output_path)
            _require(
                output_path not in ("", ".", "..")
                and "\\" not in output_path
                and not output_relative.is_absolute()
                and output_relative.as_posix() == output_path
                and ".." not in output_relative.parts,
                f"{module_label} build output path is invalid",
            )
            output_size = build_output.get("size_bytes")
            _require(
                isinstance(output_size, int)
                and not isinstance(output_size, bool)
                and output_size >= 0,
                f"{module_label} build output size is invalid",
            )
            installed_cubin = None
        else:
            _require(
                source is None and recipe is None and build_output is None,
                f"{module_label} cubin mode has CUDA build fields",
            )
            _require(
                isinstance(cubin, _InstalledArtifact),
                f"{module_label} exact cubin is unavailable",
            )
            assert isinstance(cubin, _InstalledArtifact)
            _require(
                cubin.path.suffix == ".cubin",
                f"{module_label} cubin has the wrong suffix",
            )
            _require(
                isinstance(host, dict),
                f"{module_label} cubin mode host source must be evidence only",
            )
            _require(
                isinstance(shared, _InstalledArtifact),
                f"{module_label} exact host shared library is unavailable",
            )
            assert isinstance(shared, _InstalledArtifact)
            _require(
                shared.path.suffix == ".so",
                f"{module_label} host shared library has the wrong suffix",
            )
            installed_cubin = cubin
        modules.append(
            PromotionModuleSpec(
                build_output=build_output if isinstance(build_output, dict) else None,
                cubin=installed_cubin,
                entry_point=_identifier(
                    raw_module.get("entry_point"), f"{module_label} entry point"
                ),
                host=host,
                module_id=_safe_id(raw_module.get("id"), f"{module_label} id"),
                mode=mode,
                module_ident=_identifier(
                    raw_module.get("module_ident"), f"{module_label} module ident"
                ),
                recipe=recipe if isinstance(recipe, _InstalledArtifact) else None,
                shared_library=shared,
                source=source if isinstance(source, _InstalledArtifact) else None,
            )
        )
    module_ids = [module.module_id for module in modules]
    _require(len(module_ids) == len(set(module_ids)), f"{label} module ids repeat")
    installed_closure = {
        dispatcher_artifact.artifact_id,
        *(
            module.host.artifact_id
            for module in modules
            if isinstance(module.host, _InstalledArtifact)
        ),
        *(module.source.artifact_id for module in modules if module.source is not None),
        *(module.cubin.artifact_id for module in modules if module.cubin is not None),
        *(module.recipe.artifact_id for module in modules if module.recipe is not None),
        *(
            module.shared_library.artifact_id
            for module in modules
            if isinstance(module.shared_library, _InstalledArtifact)
        ),
    }
    _require(
        set(artifacts) == installed_closure,
        f"{label} installed artifact closure contains unused or missing files",
    )

    raw_seeds = inventory.get("seeds")
    raw_routes = inventory.get("routes")
    _require(
        isinstance(raw_seeds, list) and bool(raw_seeds),
        f"{label} seeds must be non-empty",
    )
    _require(
        isinstance(raw_routes, list) and bool(raw_routes),
        f"{label} routes must be non-empty",
    )
    seeds: list[dict[str, object]] = []
    for seed_index, seed in enumerate(raw_seeds):
        _require(
            isinstance(seed, dict) and set(seed) == _SEED_KEYS,
            f"{label} seed {seed_index} is invalid",
        )
        assert isinstance(seed, dict)
        seed_id = _safe_id(seed.get("id"), f"{label} seed id")
        seeds.append({"id": seed_id})
    seed_by_id = {str(seed["id"]): seed for seed in seeds}
    _require(len(seed_by_id) == len(seeds), f"{label} seed ids repeat")
    routes: list[dict[str, object]] = []
    referenced_seed_ids: set[str] = set()
    for route_index, route in enumerate(raw_routes):
        _require(
            isinstance(route, dict) and set(route) == _ROUTE_KEYS,
            f"{label} route {route_index} is invalid",
        )
        assert isinstance(route, dict)
        route_id = _safe_id(route.get("id"), f"{label} route id")
        seed_id = _safe_id(route.get("seed_id"), f"{label} route seed")
        route_modules = route.get("module_ids")
        _require(seed_id in seed_by_id, f"{label} route references an unknown seed")
        referenced_seed_ids.add(seed_id)
        _require(
            isinstance(route_modules, list) and bool(route_modules),
            f"{label} route modules are invalid",
        )
        _require(
            all(item in module_ids for item in route_modules),
            f"{label} route references unknown modules",
        )
        _require(
            isinstance(route.get("selector"), dict),
            f"{label} route selector must be an object",
        )
        routes.append(
            {
                "id": route_id,
                "seed_id": seed_id,
                "module_ids": list(route_modules),
                "selector": route["selector"],
            }
        )
    route_ids = [str(route["id"]) for route in routes]
    _require(len(route_ids) == len(set(route_ids)), f"{label} route ids repeat")
    _require(
        referenced_seed_ids == set(seed_by_id),
        f"{label} seed denominator differs from route references",
    )
    _require(value.get("route_count") == len(routes), f"{label} route count is invalid")
    _require(
        value.get("route_denominator_sha256")
        == hashlib.sha256(_canonical(routes)).hexdigest(),
        f"{label} route denominator is invalid",
    )
    dispatcher_seed = {
        "contract": RUNTIME_CONTRACT,
        "dispatcher": dispatcher,
        "routes": routes,
        "seeds": seeds,
    }
    _require(
        inventory.get("dispatcher_seed_identity") == _content_identity(dispatcher_seed),
        f"{label} dispatcher/seed identity is invalid",
    )
    dispatcher_run_entrypoint = _identifier(
        dispatcher.get("run_entrypoint"), f"{label} dispatcher run entrypoint"
    )
    dispatcher_select_entrypoint = _identifier(
        dispatcher.get("select_entrypoint"), f"{label} dispatcher select entrypoint"
    )
    _require(
        dispatcher_run_entrypoint != dispatcher_select_entrypoint,
        f"{label} dispatcher run/select entrypoints must differ",
    )
    return PromotionTargetSpec(
        artifact_root=artifact_root,
        dispatcher=dispatcher_artifact,
        dispatcher_run_entrypoint=dispatcher_run_entrypoint,
        dispatcher_select_entrypoint=dispatcher_select_entrypoint,
        identity=str(value["runtime_inventory_identity"]).removeprefix("sha256:")[:16],
        mode=mode,
        modules=tuple(modules),
        routes=tuple(routes),
        seeds=tuple(seeds),
        target=target,
    )


@functools.cache
def get_module_specs() -> tuple[PromotionTargetSpec, ...]:
    """Return the complete verified multi-target promotion, or empty if pending."""

    path = _manifest_path()
    _require(not path.is_symlink(), "manifest must not be a symlink")
    try:
        raw = path.read_bytes()
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except PromotionManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionManifestError(f"could not read {path}: {exc}") from exc
    _require(isinstance(document, dict), "root must be an object")
    assert isinstance(document, dict)
    if document.get("kind") == MANIFEST_KIND:
        _require(
            set(document) == _PENDING_ROOT_KEYS, "pending manifest envelope is invalid"
        )
        _require(
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("status") == "pending"
            and document.get("contract") == RUNTIME_CONTRACT
            and document.get("mode") is None
            and document.get("module_ident") is None
            and document.get("entry_point") is None
            and document.get("arguments") is None
            and document.get("entries") == [],
            "pending manifest is invalid",
        )
        return ()
    _require(set(document) == _ROOT_KEYS, f"root keys must be {sorted(_ROOT_KEYS)}")
    _require(
        document.get("kind") == PACK_KIND
        and document.get("schema_version") == SCHEMA_VERSION,
        "pack kind/schema is invalid",
    )
    _require(
        document.get("name") == "flashinfer-kda-indexed-prefill",
        "pack name does not match this runtime",
    )
    denominators = document.get("contract_denominators")
    _require(
        isinstance(denominators, dict)
        and set(denominators) == {"correctness", "performance"},
        "pack contract denominators are invalid",
    )
    assert isinstance(denominators, dict)
    for name, digest in denominators.items():
        _sha256_value(digest, f"pack {name} denominator")
    mode = document.get("mode")
    _require(mode in ("cuda", "cubin"), "pack mode must be explicit")
    assert mode in ("cuda", "cubin")
    entries = document.get("entries")
    _require(isinstance(entries, list), "pack entries must be an array")
    specs = tuple(
        _parse_target(value, mode=mode, index=index)
        for index, value in enumerate(entries)
    )
    _require(
        tuple(spec.target for spec in specs) == TARGETS,
        f"entries must be ordered exactly as {list(TARGETS)}",
    )
    return specs


def selected_mode() -> PromotionMode | None:
    """Return the installed representation without compiling or loading it."""

    specs = get_module_specs()
    return None if not specs else specs[0].mode


def _get_spec(compute_capability: tuple[int, int]) -> PromotionTargetSpec:
    target = _COMPUTE_CAPABILITY_TO_TARGET.get(tuple(compute_capability))
    if target is None:
        raise PromotionManifestError(
            f"unsupported compute capability {tuple(compute_capability)}"
        )
    for spec in get_module_specs():
        if spec.target == target:
            return spec
    raise PromotionManifestError(
        f"no complete FP32 indexed KDA promotion is installed for {target}"
    )


def is_available(*, compute_capability: tuple[int, int]) -> bool:
    """Return true only after the exact modules and dispatcher bind successfully."""

    try:
        spec = _get_spec(compute_capability)
        load(compute_capability=compute_capability, mode=spec.mode)
    except (
        ImportError,
        OSError,
        PromotionManifestError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def _module_name(target: PromotionTargetSpec, module: PromotionModuleSpec) -> str:
    return f"{module.module_ident}_{target.target}_{target.mode}_{target.identity}"


def _reverified_file_bytes(
    path: Path,
    *,
    sha256: str,
    size_bytes: int,
    label: str,
) -> bytes:
    """Read one regular file once and verify the exact bytes being consumed."""

    try:
        before = path.stat(follow_symlinks=False)
        _require(
            stat.S_ISREG(before.st_mode),
            f"{label} is missing or a symlink",
        )
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except PromotionManifestError:
        raise
    except OSError as exc:
        raise PromotionManifestError(
            f"invalid FP32 indexed KDA promotion manifest: could not read {label}: {exc}"
        ) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        all(getattr(before, field) == getattr(after, field) for field in stable_fields),
        f"{label} changed while being read",
    )
    _require(
        len(payload) == size_bytes and hashlib.sha256(payload).hexdigest() == sha256,
        f"{label} bytes drifted before use",
    )
    return payload


def _reverified_artifact_bytes(artifact: _InstalledArtifact, *, label: str) -> bytes:
    """Read one sealed artifact once and verify the exact bytes being consumed."""

    return _reverified_file_bytes(
        artifact.path,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        label=label,
    )


def _load_dispatcher_namespace(spec: PromotionTargetSpec) -> ModuleType:
    """Compile and execute the rehashed dispatcher bytes without path import."""

    payload = _reverified_artifact_bytes(spec.dispatcher, label="dispatcher source")
    try:
        source = payload.decode("utf-8")
        code = compile(
            source,
            str(spec.dispatcher.path),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise PromotionManifestError(
            "invalid FP32 indexed KDA promotion manifest: dispatcher source "
            f"does not compile as UTF-8 Python: {exc}"
        ) from exc
    namespace = ModuleType(
        f"_flashinfer_generated_dispatcher_{spec.target}_{spec.mode}_{spec.identity}"
    )
    namespace.__file__ = str(spec.dispatcher.path)
    namespace.__package__ = ""
    try:
        exec(code, namespace.__dict__)
    except Exception as exc:
        raise PromotionManifestError(
            "invalid FP32 indexed KDA promotion manifest: dispatcher import "
            f"failed with {type(exc).__name__}: {exc}"
        ) from exc
    return namespace


def _require_binder_signature(binder: object) -> None:
    try:
        signature = inspect.signature(binder)
    except (TypeError, ValueError) as exc:
        raise PromotionManifestError(
            "invalid FP32 indexed KDA promotion manifest: dispatcher binder "
            "signature is not inspectable"
        ) from exc
    parameters = tuple(signature.parameters.values())
    _require(
        len(parameters) == 1
        and parameters[0].name == "modules"
        and parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[0].default is inspect.Parameter.empty,
        "dispatcher binder signature must be exactly (modules)",
    )


def _require_exact_keyword_callable(
    value: object,
    *,
    argument_names: tuple[str, ...],
    label: str,
) -> Callable[..., object]:
    _require(callable(value), f"{label} must be callable")
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as exc:
        raise PromotionManifestError(
            f"invalid FP32 indexed KDA promotion manifest: {label} signature "
            "is not inspectable"
        ) from exc
    parameters = tuple(signature.parameters.values())
    _require(
        tuple(parameter.name for parameter in parameters) == argument_names
        and all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
            for parameter in parameters
        ),
        f"{label} must have exact required keyword-only signature {argument_names!r}",
    )
    assert callable(value)
    return value


def _bind_dispatcher(
    spec: PromotionTargetSpec,
    modules: Mapping[str, Callable[..., object]],
) -> _BoundDispatcher:
    expected_module_ids = tuple(module.module_id for module in spec.modules)
    _require(
        tuple(modules) == expected_module_ids,
        "loaded module map order differs from the sealed module inventory",
    )
    _require(
        all(callable(entry) for entry in modules.values()),
        "loaded module map contains a non-callable entry",
    )
    namespace = _load_dispatcher_namespace(spec)
    _require(
        getattr(namespace, DISPATCHER_ABI_ATTRIBUTE, None) == DISPATCHER_ABI,
        f"dispatcher {DISPATCHER_ABI_ATTRIBUTE} differs from {DISPATCHER_ABI!r}",
    )
    declared_module_ids = getattr(namespace, DISPATCHER_MODULE_IDS_ATTRIBUTE, None)
    _require(
        type(declared_module_ids) is tuple
        and declared_module_ids == expected_module_ids,
        f"dispatcher {DISPATCHER_MODULE_IDS_ATTRIBUTE} must be the exact ordered module tuple",
    )
    binder = getattr(namespace, DISPATCHER_BIND_ENTRYPOINT, None)
    _require(callable(binder), f"dispatcher must export {DISPATCHER_BIND_ENTRYPOINT}()")
    _require_binder_signature(binder)
    try:
        bound = binder(modules)
    except Exception as exc:
        raise PromotionManifestError(
            "invalid FP32 indexed KDA promotion manifest: dispatcher binder "
            f"failed with {type(exc).__name__}: {exc}"
        ) from exc
    expected_entries = {
        spec.dispatcher_run_entrypoint,
        spec.dispatcher_select_entrypoint,
    }
    _require(
        type(bound) is dict and set(bound) == expected_entries,
        "dispatcher binder result must be a plain dict containing exactly the "
        "declared run/select entrypoints",
    )
    assert isinstance(bound, dict)
    return _BoundDispatcher(
        prepare=_require_exact_keyword_callable(
            bound[spec.dispatcher_run_entrypoint],
            argument_names=DISPATCHER_RUN_ARGUMENTS,
            label="bound dispatcher prepare entrypoint",
        ),
        select=_require_exact_keyword_callable(
            bound[spec.dispatcher_select_entrypoint],
            argument_names=DISPATCHER_SELECT_ARGUMENTS,
            label="bound dispatcher select entrypoint",
        ),
    )


def _cuda_include_dir() -> Path:
    from .cpp_ext import get_cuda_path

    include_dir = Path(get_cuda_path()) / "include"
    if not include_dir.is_dir():
        raise RuntimeError(f"CUDA include directory does not exist: {include_dir}")
    return include_dir


def _load_host_module(
    target: PromotionTargetSpec,
    module: PromotionModuleSpec,
    cubin: bytes,
):
    import tvm_ffi

    if target.mode == "cubin":
        if not isinstance(module.cubin, _InstalledArtifact):
            raise PromotionManifestError(
                f"module {module.module_id!r} has no exact cubin"
            )
        expected_cubin_sha256 = module.cubin.sha256
        expected_cubin_size = module.cubin.size_bytes
    else:
        if module.build_output is None:
            raise PromotionManifestError(
                f"module {module.module_id!r} has no CUDA build output"
            )
        expected_cubin_sha256 = str(module.build_output["sha256"])
        expected_cubin_size = int(module.build_output["size_bytes"])
    _require(
        len(cubin) == expected_cubin_size
        and hashlib.sha256(cubin).hexdigest() == expected_cubin_sha256,
        f"module {module.module_id!r} cubin bytes drifted before embedding or loading",
    )

    if target.mode == "cubin":
        if not isinstance(module.shared_library, _InstalledArtifact):
            raise PromotionManifestError(
                f"module {module.module_id!r} has no exact host shared library"
            )
        shared_label = f"module {module.module_id!r} host shared library"
        before = _reverified_artifact_bytes(
            module.shared_library,
            label=shared_label,
        )
        loaded = tvm_ffi.load_module(str(module.shared_library.path))
        after = _reverified_artifact_bytes(
            module.shared_library,
            label=shared_label,
        )
        _require(before == after, f"{shared_label} changed while being loaded")
        entry = getattr(loaded, module.entry_point, None)
        if not callable(entry):
            raise RuntimeError(
                f"loaded promotion module {module.module_id!r} does not export "
                f"callable {module.entry_point!r}"
            )
        return entry

    from tvm_ffi import cpp

    if not isinstance(module.host, _InstalledArtifact):
        raise PromotionManifestError(
            f"module {module.module_id!r} has no exact CUDA host shim"
        )
    host_payload = _reverified_artifact_bytes(
        module.host,
        label=f"module {module.module_id!r} CUDA host source",
    )
    try:
        host_source = host_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromotionManifestError(
            "invalid FP32 indexed KDA promotion manifest: module "
            f"{module.module_id!r} CUDA host source is not UTF-8"
        ) from exc

    module_name = _module_name(target, module)
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    loaded = cpp.load_inline(
        module_name,
        cpp_sources=host_source,
        embed_cubin={module.module_ident: cubin},
        extra_include_paths=[
            str(_cuda_include_dir()),
            str(_get_csrc_dir()),
            str(_get_include_dir()),
        ],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )
    entry = getattr(loaded, module.entry_point, None)
    if not callable(entry):
        raise RuntimeError(
            f"loaded promotion module {module.module_id!r} does not export "
            f"callable {module.entry_point!r}"
        )
    return entry


def _build_cuda_cubins(spec: PromotionTargetSpec) -> dict[str, bytes]:
    recipes = {
        module.recipe.path for module in spec.modules if module.recipe is not None
    }
    if len(recipes) != 1:
        raise PromotionManifestError(
            "CUDA target must bind exactly one public build recipe"
        )
    recipe_artifact = next(
        module.recipe for module in spec.modules if module.recipe is not None
    )
    recipe_payload = _reverified_artifact_bytes(
        recipe_artifact,
        label="CUDA build recipe",
    )
    try:
        recipe_source = recipe_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromotionManifestError(
            "invalid FP32 indexed KDA promotion manifest: CUDA build recipe "
            "is not UTF-8"
        ) from exc
    build_root = jit_env.FLASHINFER_JIT_DIR / (
        f"kda_fp32_indexed_{spec.target}_cuda_{spec.identity}"
    )
    output_root = build_root / "cubin"
    report_path = build_root / "build-report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-",
        "--source-root",
        str(spec.artifact_root),
        "--output-root",
        str(output_root),
        "--report",
        str(report_path),
        "--include-dir",
        str(_cuda_include_dir()),
        "--include-dir",
        str(_get_include_dir()),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        input=recipe_source,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"exact CUDA build recipe failed: {detail[-2000:]}")
    try:
        report = json.loads(
            report_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"exact CUDA build report is unreadable: {exc}") from exc
    _require(
        isinstance(report, dict)
        and set(report)
        == {
            "kind",
            "outputs",
            "passed",
            "schema_version",
            "toolchain",
            "toolchain_identity",
        }
        and report.get("kind") == "flashinfer.generated_program_cuda_build_report"
        and report.get("schema_version") == 1
        and report.get("passed") is True,
        "exact CUDA build report kind/schema/result is invalid",
    )
    outputs = report.get("outputs")
    _require(isinstance(outputs, list), "exact CUDA build outputs must be an array")
    expected_outputs = [module.build_output for module in spec.modules]
    _require(
        all(
            isinstance(output, dict) and set(output) == _BUILD_OUTPUT_KEYS
            for output in outputs
        ),
        "exact CUDA build output envelope is invalid",
    )
    _require(
        all(
            isinstance(output, dict) and set(output) == _BUILD_OUTPUT_KEYS
            for output in expected_outputs
        ),
        "CUDA manifest build output envelope is invalid",
    )
    output_by_id: dict[str, dict[str, object]] = {}
    for output_index, output in enumerate(outputs):
        assert isinstance(output, dict)
        output_id = _safe_id(
            output.get("id"), f"exact CUDA build output {output_index} id"
        )
        _require(
            output_id not in output_by_id,
            "exact CUDA build output ids repeat",
        )
        output_by_id[output_id] = output
    expected_by_id: dict[str, dict[str, object]] = {}
    for expected_index, expected in enumerate(expected_outputs):
        assert isinstance(expected, dict)
        output_id = _safe_id(
            expected.get("id"), f"CUDA manifest build output {expected_index} id"
        )
        _require(
            output_id not in expected_by_id,
            "CUDA manifest build output ids repeat",
        )
        expected_by_id[output_id] = expected
    _require(
        output_by_id == expected_by_id,
        "exact CUDA build output closure differs from the manifest",
    )
    cubins: dict[str, bytes] = {}
    for module in spec.modules:
        expected = module.build_output
        if expected is None:
            raise PromotionManifestError(
                f"module {module.module_id!r} has no CUDA build output"
            )
        output_id = str(expected["id"])
        _safe_id(output_id, f"module {module.module_id!r} build output id")
        relative = PurePosixPath(str(expected["path"]))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"module {module.module_id!r} build output path is invalid",
        )
        path = output_root.joinpath(*relative.parts)
        _require(
            path.is_file() and not path.is_symlink(),
            f"module {module.module_id!r} build output is missing",
        )
        payload = _reverified_file_bytes(
            path,
            sha256=str(expected["sha256"]),
            size_bytes=int(expected["size_bytes"]),
            label=f"module {module.module_id!r} rebuilt cubin",
        )
        cubins[module.module_id] = payload
    return cubins


@functools.cache
def load(*, compute_capability: tuple[int, int], mode: PromotionMode):
    """Load exact modules and bind the sealed fixed dispatcher ABI."""

    if mode not in ("cuda", "cubin"):
        raise ValueError("mode must be 'cuda' or 'cubin'")
    spec = _get_spec(compute_capability)
    if spec.mode != mode:
        raise PromotionManifestError(
            f"installed promotion mode is {spec.mode!r}, not requested mode {mode!r}"
        )
    cubins = (
        _build_cuda_cubins(spec)
        if mode == "cuda"
        else {
            module.module_id: _reverified_artifact_bytes(
                module.cubin,
                label=f"module {module.module_id!r} cubin",
            )
            for module in spec.modules
            if module.cubin is not None
        }
    )
    if set(cubins) != {module.module_id for module in spec.modules}:
        raise PromotionManifestError(
            "runtime cubin closure differs from ordered modules"
        )
    mutable_modules = {
        module.module_id: _load_host_module(spec, module, cubins[module.module_id])
        for module in spec.modules
    }
    modules: Mapping[str, Callable[..., object]] = MappingProxyType(mutable_modules)
    dispatcher = _bind_dispatcher(spec, modules)
    return _LoadedProgram(
        dispatcher=dispatcher,
        modules=modules,
        spec=spec,
    )


def _compute_capability_from_q(q: object) -> tuple[int, int]:
    try:
        from ..utils import get_compute_capability

        return tuple(get_compute_capability(q.device))
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "compute_capability is required when q does not expose a CUDA device"
        ) from exc


def prepare(
    *,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    A_log: Any,
    dt_bias: Any,
    scale: Any,
    initial_state: Any,
    output_final_state: Any,
    lower_bound: Any,
    cu_seqlens: Any,
    output: Any,
    seq_order: Any,
    prefill_workspace: Any,
    state_indices: Any,
    state_checkpoints: Any,
    checkpoint_cu_starts: Any,
    checkpoint_every_n_tokens: Any,
    compute_capability: tuple[int, int] | None = None,
) -> Prepared:
    """Bind the sealed route and all launch arguments outside the hot path."""

    if compute_capability is None:
        compute_capability = _compute_capability_from_q(q)
    spec = _get_spec(compute_capability)
    loaded = load(
        compute_capability=compute_capability,
        mode=spec.mode,
    )
    prepared = loaded.dispatcher.prepare(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        output=output,
        seq_order=seq_order,
        prefill_workspace=prefill_workspace,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )
    return Prepared(prepared)


def run(
    *,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    A_log: Any,
    dt_bias: Any,
    scale: Any,
    initial_state: Any,
    output_final_state: Any,
    lower_bound: Any,
    cu_seqlens: Any,
    output: Any,
    seq_order: Any,
    prefill_workspace: Any,
    state_indices: Any,
    state_checkpoints: Any,
    checkpoint_cu_starts: Any,
    checkpoint_every_n_tokens: Any,
    compute_capability: tuple[int, int] | None = None,
) -> object:
    """Prepare, launch once, and always release the bound program closure."""

    prepared = prepare(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        output=output,
        seq_order=seq_order,
        prefill_workspace=prefill_workspace,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        compute_capability=compute_capability,
    )
    try:
        return prepared.launch()
    finally:
        prepared.close()


def _clear_caches_for_testing() -> None:
    """Clear manifest and loaded-module caches for isolated CPU tests."""

    get_module_specs.cache_clear()
    load.cache_clear()


__all__ = [
    "DISPATCHER_ABI",
    "DISPATCHER_ABI_ATTRIBUTE",
    "DISPATCHER_BIND_ENTRYPOINT",
    "DISPATCHER_MODULE_IDS_ATTRIBUTE",
    "DISPATCHER_RUN_ARGUMENTS",
    "DISPATCHER_SELECT_ARGUMENTS",
    "MANIFEST_FILENAME",
    "MANIFEST_KIND",
    "Prepared",
    "PromotionManifestError",
    "RUNTIME_CONTRACT",
    "get_module_specs",
    "is_available",
    "load",
    "prepare",
    "run",
    "selected_mode",
]
