import tvm
import numpy as np
from tvm.script import ir as I, relax as R, tir as T
from scipy.special import softmax
from flashinfer_byoc import FlashInferIRModuleGen


def _gen_ref(seq_len, num_heads, head_dim, dtype, rope=False):
    def f_rope(x):
        l = x.shape[0]
        d = head_dim // 2
        permuted = np.concatenate((-x[..., d:], x[..., :d]), axis=-1).astype("float32")
        idx = np.arange(0, head_dim, 2).astype("float32")
        inv_freq = np.power(1e-4, idx / float(head_dim), dtype=np.float32)
        t = np.arange(seq_len - l, seq_len, dtype=np.float32)
        freqs = np.einsum("i,j->ij", t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        emb = np.expand_dims(emb, 1)
        return np.cos(emb) * x + np.sin(emb) * permuted

    q_np = np.random.randn(1, num_heads, head_dim).astype(dtype)
    k_np = np.random.randn(seq_len, num_heads, head_dim)
    v_np = np.random.randn(seq_len, num_heads, head_dim)
    tmp_np = np.zeros((8 * 1024 * 1024,), "float32")

    q_before_rope = q_np
    k_before_rope = k_np

    if rope:
        q_np = f_rope(q_np)
        k_np = f_rope(k_np)

    q_np_T = np.reshape(q_np, [num_heads, 1, head_dim])
    k_np_T = np.transpose(k_np, [1, 2, 0])
    v_np_T = np.transpose(v_np, [1, 0, 2])
    p_np = q_np_T @ k_np_T / np.sqrt(head_dim)
    s_np = softmax(p_np, axis=-1)
    o_np = np.reshape(s_np @ v_np_T, [num_heads, head_dim])

    return (
        q_before_rope.squeeze(0).astype(dtype),
        k_before_rope.astype(dtype),
        v_np.astype(dtype),
        tmp_np,
        o_np.astype(dtype),
    )


def test_decode(seq_len, num_heads, head_dim, dtype, rope=False):
    mod = FlashInferIRModuleGen(dtype, dtype, int(rope))
    ex = tvm.relax.build(mod, "cuda")

    dev = tvm.device("cuda", 0)
    vm = tvm.relax.VirtualMachine(ex, dev)
    f = vm["decode"]
    q, k, v, tmp, ref = _gen_ref(seq_len, num_heads, head_dim, dtype, rope=rope)

    inputs = [tvm.nd.array(t, dev) for t in [q, k, v, tmp]]
    o = f(*inputs).numpy()
    tmp = inputs[-1].numpy()
    nans = np.count_nonzero(np.isnan(o))
    assert nans == 0, f"nans = {nans}"
    np.testing.assert_allclose(o, ref, rtol=1e-3, atol=1e-3, verbose=True)


if __name__ == "__main__":
    test_decode(16, 32, 128, "float32")
    test_decode(16, 32, 128, "float16")
    test_decode(99, 32, 128, "float16")
    test_decode(16384, 32, 128, "float32", rope=True)
    test_decode(99, 32, 128, "float16", rope=True)
