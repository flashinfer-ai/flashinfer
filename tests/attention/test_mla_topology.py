"""Architecture contracts for the private MLA package layout."""

import ast
import importlib
from pathlib import Path

from flashinfer.mla import _core as mla_core
from flashinfer.mla._batch_mla import _functional as batch_mla_functional
from flashinfer.mla._batch_mla import _wrapper as batch_mla_wrapper


_REQUIRED_MODULES = {
    "__init__.py",
    "_core.py",
    "_sparse_mla_sm120.py",
    "_batch_mla/__init__.py",
    "_batch_mla/_functional.py",
    "_batch_mla/_planning.py",
    "_batch_mla/_wrapper.py",
    "_batch_mla/_backends/__init__.py",
    "_batch_mla/_backends/_fa_common.py",
    "_batch_mla/_backends/_layout.py",
    "_batch_mla/_backends/cute_dsl_backend.py",
    "_batch_mla/_backends/cutlass_backend.py",
    "_batch_mla/_backends/fa2_backend.py",
    "_batch_mla/_backends/fa3_backend.py",
    "_batch_mla/_backends/trtllm_gen_backend.py",
    "_batch_mla/_backends/xqa_backend.py",
}

_LEGACY_MODULES = {
    "batch_mla.py",
    "_batch_mla.py",
    "_batch_mla/_core.py",
    "_batch_mla/_metadata.py",
    "_batch_mla/_backends.py",
    "_planning/__init__.py",
    "_planning/batch_mla.py",
    "_backends/__init__.py",
    "_backends/batch_mla/__init__.py",
    "_backends/batch_mla/cute_dsl_backend.py",
    "_backends/batch_mla/cutlass_backend.py",
    "_backends/batch_mla/fa2_backend.py",
    "_backends/batch_mla/fa3_backend.py",
    "_backends/batch_mla/trtllm_gen_backend.py",
    "_backends/batch_mla/xqa_backend.py",
}

_ROOT_CORE_FUNCTIONAL_ALIASES = (
    "CuteDslMlaDecodeRunner",
    "TrtllmGenMlaDecodeRunner",
    "_run_mla_decode_cute_dsl",
    "_run_mla_decode_trtllm_gen",
    "_run_mla_decode_trtllm_gen_or_cute_dsl_impl",
    "_run_mla_decode_xqa",
    "get_batch_mla_module",
    "get_mla_module",
    "get_trtllm_gen_fmha_module",
    "xqa_batch_decode_with_kv_cache_mla",
)

_OWNED_SYMBOLS = {
    "flashinfer.mla._batch_mla._planning": (
        "_MLAPlanArguments",
        "_MLAWrapperPlanResult",
    ),
    "flashinfer.mla._batch_mla._backends._fa_common": (
        "_BatchMLAGeneratedFaMechanics",
    ),
    "flashinfer.mla._batch_mla._backends._layout": ("_concat_adjacent_views_or_cat",),
    "flashinfer.mla._batch_mla._backends.fa2_backend": (
        "_BatchMLAPagedAttentionFa2Backend",
    ),
    "flashinfer.mla._batch_mla._backends.fa3_backend": (
        "_BatchMLAPagedAttentionFa3Backend",
    ),
    "flashinfer.mla._batch_mla._backends.cutlass_backend": (
        "_BatchMLAPagedAttentionCutlassBackend",
        "_validate_cutlass_metadata",
    ),
    "flashinfer.mla._batch_mla._backends.trtllm_gen_backend": (
        "_BatchMLAPagedAttentionTrtllmGenBackend",
        "TrtllmGenMlaDecodeRunner",
    ),
    "flashinfer.mla._batch_mla._backends.cute_dsl_backend": (
        "_BatchMLAPagedAttentionCuteDslBackend",
        "CuteDslMlaDecodeRunner",
        "_cute_dsl_max_supported_batch",
    ),
    "flashinfer.mla._batch_mla._backends.xqa_backend": (
        "_BatchMLAPagedAttentionXqaBackend",
    ),
}


def _import_targets(path: Path) -> set[str]:
    targets = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module is not None:
                module = prefix + node.module
                targets.add(module)
                targets.update(f"{module}.{alias.name}" for alias in node.names)
            else:
                targets.update(prefix + alias.name for alias in node.names)
    return targets


def test_import_targets_include_from_import_aliases(tmp_path):
    source = tmp_path / "imports.py"
    source.write_text(
        "from flashinfer.mla import _core\n"
        "from flashinfer.mla._batch_mla import _functional\n"
        "from flashinfer.mla._batch_mla import _wrapper\n"
        "from . import _backends\n"
    )

    assert _import_targets(source) >= {
        "flashinfer.mla._core",
        "flashinfer.mla._batch_mla._functional",
        "flashinfer.mla._batch_mla._wrapper",
        "._backends",
    }


def test_private_mla_topology_and_dependency_direction():
    root = Path(mla_core.__file__).parent
    batch = root / "_batch_mla"
    backends = batch / "_backends"

    actual_modules = {str(path.relative_to(root)) for path in root.rglob("*.py")}
    assert actual_modules >= _REQUIRED_MODULES
    assert _LEGACY_MODULES.isdisjoint(actual_modules)
    assert (batch / "__init__.py").read_bytes() == b""
    assert (backends / "__init__.py").read_bytes() == b""

    for module_name, symbol_names in _OWNED_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol_name in symbol_names:
            assert getattr(module, symbol_name).__module__ == module_name

    for path in batch.rglob("*.py"):
        targets = _import_targets(path)
        assert "flashinfer.mla._core" not in targets
        assert not any(
            target.startswith(".") and target.lstrip(".") == "_core"
            for target in targets
        )

    for path in backends.glob("*_backend.py"):
        targets = _import_targets(path)
        assert not any(target.lstrip(".") == "_core" for target in targets)
        assert "flashinfer.mla._batch_mla._functional" not in targets
        assert "flashinfer.mla._batch_mla._wrapper" not in targets

    for path in (backends / "_fa_common.py", backends / "_layout.py"):
        assert not any(target.endswith("_backend") for target in _import_targets(path))

    assert not any(
        "_backends" in target for target in _import_targets(batch / "_planning.py")
    )


def test_root_core_reexports_batch_public_symbols():
    for name in _ROOT_CORE_FUNCTIONAL_ALIASES:
        assert getattr(mla_core, name) is getattr(batch_mla_functional, name)

    assert (
        mla_core.BatchMLAPagedAttentionWrapper
        is batch_mla_wrapper.BatchMLAPagedAttentionWrapper
    )
    assert (
        batch_mla_wrapper.BatchMLAPagedAttentionWrapper.__module__
        == batch_mla_wrapper.__name__
    )
    assert (
        mla_core.trtllm_batch_decode_with_kv_cache_mla.__module__ == mla_core.__name__
    )
    assert hasattr(batch_mla_functional, "_compute_mla_decode_buckets")
    assert hasattr(batch_mla_functional, "_build_mla_decode_tuning_config")
    assert not hasattr(mla_core, "_compute_mla_decode_buckets")
    assert not hasattr(mla_core, "_build_mla_decode_tuning_config")
