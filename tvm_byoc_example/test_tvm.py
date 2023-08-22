import tvm
import numpy as np
from tvm.script import ir as I, relax as R, tir as T
from scipy.special import softmax
from flashinfer_byoc import FlashInferIRModuleGen


def _gen_ref(seq_len, num_heads, head_dim, dtype, rope=False):
    def f_rope(x):
        d = head_dim // 2
        permuted = np.concatenate((-x[..., d:], x[..., :d]), axis=-1)
        idx = np.arange(d).astype(dtype)
        inv_freq = (seq_len - 1) * np.power(1e-4, idx / d)
        inv_freq = np.concatenate((inv_freq, inv_freq), axis=-1)
        return np.cos(inv_freq) * x + np.sin(inv_freq) * permuted

    q_np = np.random.randn(num_heads, head_dim).astype(dtype)
    k_np = np.random.randn(seq_len, num_heads, head_dim).astype(dtype)
    v_np = np.random.randn(seq_len, num_heads, head_dim).astype(dtype)
    tmp_np = np.zeros((2 * 1024 * 1024,), "float32")

    if rope:
        q_np = f_rope(q_np)
        k_np = f_rope(k_np)

    q_np_T = np.reshape(q_np, [num_heads, 1, head_dim])
    k_np_T = np.transpose(k_np, [1, 2, 0])
    v_np_T = np.transpose(v_np, [1, 0, 2])
    p_np = q_np_T @ k_np_T / np.sqrt(head_dim)
    s_np = softmax(p_np, axis=-1)
    o_np = np.reshape(s_np @ v_np_T, [num_heads, head_dim])

    return q_np, k_np, v_np, tmp_np, o_np


def test_decode(seq_len, num_heads, head_dim, dtype):
    mod = FlashInferIRModuleGen(dtype, dtype)
    ex = tvm.relax.build(mod, "cuda")

    dev = tvm.device("cuda", 0)
    vm = tvm.relax.VirtualMachine(ex, dev)
    f = vm["decode"]
    q, k, v, tmp, ref = _gen_ref(seq_len, num_heads, head_dim, dtype)

    inputs = [tvm.nd.array(t, dev) for t in [q, k, v, tmp]]
    o = f(*inputs).numpy()
    nans = np.count_nonzero(np.isnan(o))
    assert nans == 0, f"nans = {nans}"

    np.testing.assert_allclose(o, ref, rtol=1e-3, atol=1e-3, verbose=True)


def test_fused_rope_decode(seq_len, num_heads, head_dim, dtype):
    mod = FlashInferIRModuleGen(dtype, dtype)
    ex = tvm.relax.build(mod, "cuda")

    dev = tvm.device("cuda", 0)
    vm = tvm.relax.VirtualMachine(ex, dev)
    f = vm["fused_rope_decode"]
    q, k, v, ref = _gen_ref(seq_len, num_heads, head_dim, dtype, rope=True)

    inputs = [tvm.nd.array(t, dev) for t in [q, k, v]]
    k_new, o = [x.numpy() for x in f(*inputs)]
    nans = np.count_nonzero(np.isnan(o))
    assert nans == 0, f"nans = {nans}"

    np.testing.assert_allclose(o, ref, rtol=1e-3, atol=1e-3, verbose=True)


def test_fused_updated_rope_decode(seq_len, num_heads, head_dim, dtype):
    mod = FlashInferIRModuleGen(dtype, dtype)
    ex = tvm.relax.build(mod, "cuda")

    dev = tvm.device("cuda", 0)
    vm = tvm.relax.VirtualMachine(ex, dev)
    f = vm["fused_updated_rope_decode"]
    q, k, v, ref = _gen_ref(seq_len, num_heads, head_dim, dtype)

    inputs = [tvm.nd.array(t, dev) for t in [q, k, v]]
    k_new, o = [x.numpy() for x in f(*inputs)]
    nans = np.count_nonzero(np.isnan(o))
    assert nans == 0, f"nans = {nans}"

    np.testing.assert_allclose(o, ref, rtol=1e-3, atol=1e-3, verbose=True)


if __name__ == "__main__":
    test_decode(16, 32, 128, "float32")
    test_decode(16, 32, 128, "float16")
    # test_fused_rope_decode(16384, 32, 128, "float32")
    # test_fused_rope_decode(16, 32, 128, "float16")
    # test_fused_updated_rope_decode(16, 32, 128, "float32")
    # test_fused_updated_rope_decode(16, 32, 128, "float16")
