"""The custom-JIT ragged prefill path must forward kv_cache_sf to ragged_run."""

import pytest
import torch

from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

_NUM_FIXED_RUN_ARGS = 14  # workspace buffers ... enable_pdl, see ragged run()


class _CapturingModule:
    def __init__(self):
        self.calls = []

    def ragged_run(self, *args):
        self.calls.append(args)


def _custom_jit_wrapper(additional_tensor_names, total_tokens):
    wrapper = BatchPrefillWithRaggedKVCacheWrapper.__new__(
        BatchPrefillWithRaggedKVCacheWrapper
    )
    module = _CapturingModule()
    attrs = dict(
        _backend="fa2",
        _batch_invariant=None,
        _jit_module=object(),
        _jit_additional_tensor_names=list(additional_tensor_names),
        _cached_module=module,
        _cached_q_data_type=torch.float16,
        _cached_kv_data_type=torch.float8_e4m3fn,
        _cached_o_data_type=torch.float16,
        _qo_indptr_last=total_tokens,
        _window_left=-1,
        _logits_soft_cap=None,
        _sm_scale=None,
        _rope_scale=None,
        _rope_theta=None,
        _causal=False,
        _custom_mask_buf=None,
        _mask_indptr_buf=None,
        _variant_owns_mask=False,
        _cudnn_plan=None,
        _kv_layout="NHD",
        _float_workspace_buffer=torch.empty(1),
        _int_workspace_buffer=torch.empty(1),
        _plan_info=[0],
        _qo_indptr_buf=torch.tensor([0, total_tokens], dtype=torch.int32),
        _kv_indptr_buf=torch.tensor([0, total_tokens], dtype=torch.int32),
        _use_token_head_sf=False,
        device=torch.device("cpu"),
    )
    for name, value in attrs.items():
        setattr(wrapper, name, value)
    return wrapper, module


def _qkv(total_tokens=4, num_heads=2, head_dim=64):
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16)
    k = torch.zeros(total_tokens, num_heads, head_dim, dtype=torch.float8_e4m3fn)
    v = torch.zeros(total_tokens, num_heads, head_dim, dtype=torch.float8_e4m3fn)
    return q, k, v


@pytest.mark.cpu_only
@pytest.mark.parametrize("packed", [False, True])
def test_custom_jit_ragged_run_forwards_kv_cache_sf(packed):
    names = ["maybe_k_cache_sf", "maybe_v_cache_sf"]
    q, k, v = _qkv()
    wrapper, module = _custom_jit_wrapper(names, q.shape[0])
    k_sf = torch.ones(q.shape[0], k.shape[1])
    v_sf = torch.full((q.shape[0], k.shape[1]), 2.0)
    kv_cache_sf = torch.stack([k_sf, v_sf], dim=1) if packed else (k_sf, v_sf)

    wrapper.run(q, k, v, kv_cache_sf=kv_cache_sf, enable_pdl=False)

    (run_args,) = module.calls
    forwarded_k_sf, forwarded_v_sf = run_args[_NUM_FIXED_RUN_ARGS:]
    assert forwarded_k_sf is not None and forwarded_v_sf is not None
    assert torch.equal(forwarded_k_sf, k_sf)
    assert torch.equal(forwarded_v_sf, v_sf)
    if not packed:
        assert forwarded_k_sf is k_sf and forwarded_v_sf is v_sf


@pytest.mark.cpu_only
def test_custom_jit_ragged_run_keeps_positional_args_after_kv_cache_sf():
    # User-declared tensors still come from *args, in declaration order, around
    # the scale tensors filled from kv_cache_sf.
    names = ["maybe_k_cache_sf", "user_bias", "maybe_v_cache_sf", "user_gate"]
    q, k, v = _qkv()
    wrapper, module = _custom_jit_wrapper(names, q.shape[0])
    k_sf = torch.ones(q.shape[0], k.shape[1])
    v_sf = torch.ones(q.shape[0], k.shape[1])
    bias, gate, scalar = torch.zeros(3), torch.ones(3), 0.5

    wrapper.run(q, k, v, bias, gate, scalar, kv_cache_sf=(k_sf, v_sf), enable_pdl=False)

    (run_args,) = module.calls
    forwarded = run_args[_NUM_FIXED_RUN_ARGS:]
    expected = (k_sf, bias, v_sf, gate, scalar)
    assert len(forwarded) == len(expected)
    assert all(a is b for a, b in zip(forwarded, expected, strict=True))


@pytest.mark.cpu_only
def test_custom_jit_ragged_run_without_kv_cache_sf_passes_none():
    names = ["maybe_k_cache_sf", "maybe_v_cache_sf"]
    q, k, v = _qkv()
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    wrapper, module = _custom_jit_wrapper(names, q.shape[0])
    wrapper._cached_kv_data_type = torch.float16

    wrapper.run(q, k, v, enable_pdl=False)

    (run_args,) = module.calls
    assert run_args[_NUM_FIXED_RUN_ARGS:] == (None, None)


@pytest.mark.cpu_only
def test_custom_jit_ragged_run_positional_scales_without_kv_cache_sf():
    # Callers that pass the scales positionally (the only way before kv_cache_sf
    # was forwarded here) keep working when kv_cache_sf is omitted.
    names = ["maybe_k_cache_sf", "user_bias", "maybe_v_cache_sf"]
    q, k, v = _qkv()
    wrapper, module = _custom_jit_wrapper(names, q.shape[0])
    k_sf = torch.ones(q.shape[0], k.shape[1])
    v_sf = torch.ones(q.shape[0], k.shape[1])
    bias, scalar = torch.zeros(3), 0.5

    wrapper.run(q, k, v, k_sf, bias, v_sf, scalar, enable_pdl=False)

    (run_args,) = module.calls
    forwarded = run_args[_NUM_FIXED_RUN_ARGS:]
    expected = (k_sf, bias, v_sf, scalar)
    assert len(forwarded) == len(expected)
    assert all(a is b for a, b in zip(forwarded, expected, strict=True))


@pytest.mark.cpu_only
@pytest.mark.parametrize("packed", [False, True])
def test_custom_jit_ragged_run_positional_scales_with_kv_cache_sf(packed):
    # Passing the scales both positionally and through kv_cache_sf does not
    # shift the positional args that follow them.
    names = ["maybe_k_cache_sf", "user_bias", "maybe_v_cache_sf"]
    q, k, v = _qkv()
    wrapper, module = _custom_jit_wrapper(names, q.shape[0])
    k_sf = torch.ones(q.shape[0], k.shape[1])
    v_sf = torch.full((q.shape[0], k.shape[1]), 2.0)
    if packed:
        kv_cache_sf = torch.stack([k_sf, v_sf], dim=1)
        k_sf, v_sf = kv_cache_sf.unbind(dim=1)
    else:
        kv_cache_sf = (k_sf, v_sf)
    bias, scalar = torch.zeros(3), 0.5

    wrapper.run(
        q, k, v, k_sf, bias, v_sf, scalar, kv_cache_sf=kv_cache_sf, enable_pdl=False
    )

    (run_args,) = module.calls
    forwarded = run_args[_NUM_FIXED_RUN_ARGS:]
    assert len(forwarded) == 4
    assert torch.equal(forwarded[0], k_sf) and torch.equal(forwarded[2], v_sf)
    assert forwarded[1] is bias and forwarded[3] == scalar
