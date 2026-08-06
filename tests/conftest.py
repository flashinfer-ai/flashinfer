import json
import os
import types
from pathlib import Path
from typing import Any, Dict, Set

import pytest
import torch
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version


def _patch_cutlass_dsl_operand_major_mode():
    try:
        import cutlass.cute as cute
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
    except ImportError:
        return
    if not hasattr(cute.nvgpu, "OperandMajorMode"):
        cute.nvgpu.OperandMajorMode = OperandMajorMode


_patch_cutlass_dsl_operand_major_mode()

import flashinfer
from flashinfer.jit import MissingJITCacheError

# Global tracking for JIT cache coverage
# Store tuples of (test_name, module_name, spec_info)
_MISSING_JIT_CACHE_MODULES: Set[tuple] = set()

# File path for aggregating JIT cache info across multiple pytest runs
_JIT_CACHE_REPORT_FILE = os.environ.get("FLASHINFER_JIT_CACHE_REPORT_FILE", None)

TORCH_COMPILE_FNS = [
    flashinfer.activation.silu_and_mul,
    flashinfer.activation.gelu_and_mul,
    flashinfer.activation.gelu_tanh_and_mul,
    flashinfer.cascade.merge_state,
    flashinfer.cascade.merge_state_in_place,
    flashinfer.cascade.merge_states,
    flashinfer.cascade.MultiLevelCascadeAttentionWrapper.run,
    flashinfer.cascade.BatchDecodeWithSharedPrefixPagedKVCacheWrapper.forward,
    flashinfer.cascade.BatchPrefillWithSharedPrefixPagedKVCacheWrapper.forward,
    flashinfer.decode.single_decode_with_kv_cache,
    flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run,
    flashinfer.gemm.bmm_fp8,
    flashinfer.gemm.SegmentGEMMWrapper.run,
    flashinfer.norm.rmsnorm,
    flashinfer.norm.fused_add_rmsnorm,
    flashinfer.norm.gemma_rmsnorm,
    flashinfer.norm.gemma_fused_add_rmsnorm,
    flashinfer.page.append_paged_kv_cache,
    flashinfer.prefill.single_prefill_with_kv_cache,
    flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run,
    flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run,
    flashinfer.quantization.packbits,
    flashinfer.rope.apply_rope,
    flashinfer.rope.apply_rope_inplace,
    flashinfer.rope.apply_rope_pos_ids,
    flashinfer.rope.apply_rope_pos_ids_inplace,
    flashinfer.rope.apply_llama31_rope,
    flashinfer.rope.apply_llama31_rope_inplace,
    flashinfer.rope.apply_llama31_rope_pos_ids,
    flashinfer.rope.apply_llama31_rope_pos_ids_inplace,
    flashinfer.sampling.sampling_from_probs,
    flashinfer.sampling.sampling_from_logits,
    flashinfer.sampling.top_p_sampling_from_probs,
    flashinfer.sampling.top_k_sampling_from_probs,
    flashinfer.sampling.min_p_sampling_from_probs,
    flashinfer.sampling.top_k_top_p_sampling_from_probs,
    flashinfer.sampling.top_p_renorm_probs,
    flashinfer.sampling.top_k_renorm_probs,
    flashinfer.sampling.top_k_mask_logits,
    flashinfer.sampling.chain_speculative_sampling,
]

_TORCH_COMPILE_CACHE: Dict[str, Any] = dict()


def _set_torch_compile_options():
    import torch._dynamo.config

    torch._dynamo.config.cache_size_limit = 128


def _monkeypatch_add_torch_compile(func):
    """
    Replace the given function with its torch.compile version.
    """

    from torch._library.custom_ops import CustomOpDef

    if type(func) is types.FunctionType:
        fn = func
    elif isinstance(func, CustomOpDef):
        fn = func._init_fn
    else:
        raise ValueError(f"Unsupported fn type {type(func)}")

    fullname = fn.__module__ + "." + fn.__qualname__
    components = fullname.split(".")
    assert components[0] == "flashinfer"
    module = flashinfer
    for component in components[1:-1]:
        module = getattr(module, component)
    if not hasattr(module, components[-1]):
        raise ValueError(f"Failed to monkeypatch: {fullname}")

    def wrapper(*args, **kwargs):
        compiled = _TORCH_COMPILE_CACHE.get(fullname)
        if compiled is None:
            # Warmup -- JIT compile / import the kernels.
            #
            # From user side, users also need to warmup the model beforehand,
            # as suggested by PyTorch Cuda Graph docs (not sure if it's also
            # recommended for torch.compile as well.)
            #
            # For the convenience of FlashInfer testing, we do the warmup here,
            # on the first run of the function. The caveat is that the first
            # call will run twice: once to warmup, and another through the
            # compiled version.
            func(*args, **kwargs)

            # Compile
            compiled = torch.compile(
                func,
                fullgraph=True,
                backend="inductor",
                mode="max-autotune-no-cudagraphs",
            )
            _TORCH_COMPILE_CACHE[fn.__name__] = compiled

        return compiled(*args, **kwargs)

    setattr(module, fn.__name__, wrapper)
    print("Applied torch.compile to", fullname)


def pytest_addoption(parser):
    """moe_ep-specific CLI flags."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="moe_ep backend to exercise (nccl_ep / nixl_ep / both); "
        "consumed by tests/moe_ep/test_moe_ep_layer_multirank.py",
    )


def pytest_configure(config):
    if os.environ.get("FLASHINFER_TEST_TORCH_COMPILE", "0") == "1":
        if torch_version < TorchVersion("2.4"):
            pytest.skip("torch.compile requires torch >= 2.4")
        _set_torch_compile_options()
        for fn in TORCH_COMPILE_FNS:
            _monkeypatch_add_torch_compile(fn)
    # moe_ep markers (Part B of the EP API design integration).
    config.addinivalue_line(
        "markers", "nvep: requires a moe_ep-enabled install (default)"
    )
    config.addinivalue_line("markers", "gpu_2: requires >=2 GPUs")
    config.addinivalue_line("markers", "gpu_4: requires >=4 GPUs")
    config.addinivalue_line("markers", "gpu_8: requires >=8 GPUs")
    config.addinivalue_line("markers", "arch_blackwell: requires sm_100 or sm_103")
    config.addinivalue_line(
        "markers",
        "long_running: front-load this test file at the start of the parallel CI queue",
    )
    config.addinivalue_line(
        "markers", "solo: run this whole test file alone (memory-heavy)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip moe_ep tests on hosts that lack the requisite env / GPUs / arch."""
    nvep_built = False
    try:
        from importlib import import_module

        nvep_built = bool(import_module("flashinfer.moe_ep").available_backends())
    except ImportError:
        pass

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cc = (
        torch.cuda.get_device_capability(0)
        if torch.cuda.is_available() and ngpu > 0
        else (0, 0)
    )

    for item in items:
        if "nvep" in item.keywords and not nvep_built:
            item.add_marker(
                pytest.mark.skip(
                    reason="no moe_ep backend built (EP builds by default; "
                    "check install log for skipped-backend warnings)"
                )
            )
        for mk, req in (("gpu_2", 2), ("gpu_4", 4), ("gpu_8", 8)):
            if mk in item.keywords and ngpu < req:
                item.add_marker(pytest.mark.skip(reason=f"needs >= {req} GPUs"))
            elif mk in item.keywords and "WORLD_SIZE" not in os.environ:
                # Multi-rank tests must be launched via torchrun (see
                # tests/moe_ep/run_tests.sh); under plain pytest auto-discovery
                # (e.g. CI unit-test sweeps) they would hang on dist init.
                item.add_marker(
                    pytest.mark.skip(
                        reason="requires torchrun launch (WORLD_SIZE unset)"
                    )
                )
        if "arch_blackwell" in item.keywords and cc < (10, 0):
            item.add_marker(pytest.mark.skip(reason="needs sm_100+"))


def is_cuda_oom_error_str(e: str) -> bool:
    return "CUDA" in e and "out of memory" in e


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    # skip OOM error and missing JIT cache errors
    try:
        yield
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or is_cuda_oom_error_str(str(e)):
            pytest.skip("Skipping due to OOM")
        elif isinstance(e, MissingJITCacheError):
            # Record the test that was skipped due to missing JIT cache
            test_name = item.nodeid
            spec = e.spec
            module_name = spec.name if spec else "unknown"

            # Create a dict with module info for reporting
            spec_info = None
            if spec:
                spec_info = {
                    "name": spec.name,
                    "sources": [str(s) for s in spec.sources],
                    "needs_device_linking": spec.needs_device_linking,
                    "aot_path": str(spec.aot_path),
                }

            _MISSING_JIT_CACHE_MODULES.add((test_name, module_name, str(spec_info)))
            pytest.skip(f"Skipping due to missing JIT cache for module: {module_name}")
        else:
            raise


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate JIT cache coverage report at the end of test session"""
    if not _MISSING_JIT_CACHE_MODULES:
        return  # No missing modules

    # If report file is specified, write to file for later aggregation
    # Otherwise, print summary directly
    if _JIT_CACHE_REPORT_FILE:
        from filelock import FileLock

        # Convert set to list for JSON serialization
        data = [
            {"test_name": test_name, "module_name": module_name, "spec_info": spec_info}
            for test_name, module_name, spec_info in _MISSING_JIT_CACHE_MODULES
        ]

        # Use file locking to handle concurrent writes from multiple pytest processes
        Path(_JIT_CACHE_REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)
        lock_file = _JIT_CACHE_REPORT_FILE + ".lock"
        with FileLock(lock_file), open(_JIT_CACHE_REPORT_FILE, "a") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        return

    # Single pytest run - print summary directly
    terminalreporter.section("flashinfer-jit-cache Package Coverage Report")
    terminalreporter.write_line("")
    terminalreporter.write_line(
        "This report shows the coverage of the flashinfer-jit-cache package."
    )
    terminalreporter.write_line(
        "Tests are skipped when required modules are not found in the installed JIT cache."
    )
    terminalreporter.write_line("")
    terminalreporter.write_line(
        f"⚠️  {len(_MISSING_JIT_CACHE_MODULES)} test(s) skipped due to missing JIT cache modules:"
    )
    terminalreporter.write_line("")

    # Group by module name
    module_to_tests = {}
    for test_name, module_name, spec_info in _MISSING_JIT_CACHE_MODULES:
        if module_name not in module_to_tests:
            module_to_tests[module_name] = {"tests": [], "spec_info": spec_info}
        module_to_tests[module_name]["tests"].append(test_name)

    for module_name in sorted(module_to_tests.keys()):
        info = module_to_tests[module_name]
        terminalreporter.write_line(f"Module: {module_name}")
        terminalreporter.write_line(f"  Spec: {info['spec_info']}")
        terminalreporter.write_line(f"  Affected tests ({len(info['tests'])}):")
        for test in sorted(info["tests"]):
            terminalreporter.write_line(f"    - {test}")
        terminalreporter.write_line("")

    terminalreporter.write_line(
        "These tests require JIT compilation but FLASHINFER_DISABLE_JIT=1 was set."
    )
    terminalreporter.write_line(
        "To improve coverage, add the missing modules to the flashinfer-jit-cache build configuration."
    )
