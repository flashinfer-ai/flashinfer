import os
import types

import pytest
import torch
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

import flashinfer

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
    flashinfer.sampling.top_p_sampling_from_probs,
    flashinfer.sampling.top_k_sampling_from_probs,
    flashinfer.sampling.min_p_sampling_from_probs,
    flashinfer.sampling.top_k_top_p_sampling_from_probs,
    flashinfer.sampling.top_p_renorm_probs,
    flashinfer.sampling.top_k_renorm_probs,
    flashinfer.sampling.top_k_mask_logits,
    flashinfer.sampling.chain_speculative_sampling,
]

_TORCH_COMPILE_CACHE = dict()


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


def pytest_configure(config):
    if os.environ.get("FLASHINFER_TEST_TORCH_COMPILE", "0") == "1":
        if torch_version < TorchVersion("2.4"):
            pytest.skip("torch.compile requires torch >= 2.4")
        _set_torch_compile_options()
        for fn in TORCH_COMPILE_FNS:
            _monkeypatch_add_torch_compile(fn)
