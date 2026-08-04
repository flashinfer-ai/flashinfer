"""
Regression test for the ``rotary_dim <= head_dim`` guard in the RoPE launchers.

When ``rotary_dim > head_dim`` the RoPE kernel reads past the current head's data
(``x + rotary_dim/2``) into the next head/token, producing silently wrong attention
output (no crash, no NaN).  The Python wrappers now raise ``ValueError`` before any
kernel/JIT work; this test exercises that guard for every public RoPE entry point.

The bad-path assertions run without a GPU because the guard fires pre-kernel.  The
happy path (``rotary_dim == head_dim``) drives the kernel and therefore needs CUDA.
"""

import pytest

torch = pytest.importorskip("torch")

import flashinfer  # noqa: E402

HEAD_DIM = 64


def _rope_inputs(device, dtype=torch.float16):
    """Minimal q/k/indptr/offsets/pos_ids tensors for head_dim=HEAD_DIM."""
    nnz, nq, nk = 2, 2, 2
    q = torch.zeros((nnz, nq, HEAD_DIM), dtype=dtype, device=device)
    k = torch.zeros((nnz, nk, HEAD_DIM), dtype=dtype, device=device)
    indptr = torch.tensor([0, nnz], dtype=torch.int32, device=device)
    offsets = torch.zeros((nnz,), dtype=torch.int32, device=device)
    pos_ids = torch.arange(nnz, dtype=torch.int32, device=device)
    return q, k, indptr, offsets, pos_ids


def _call_apply_rope_inplace(rotary_dim, device):
    q, k, indptr, offsets, _ = _rope_inputs(device)
    flashinfer.apply_rope_inplace(q, k, indptr, offsets, rotary_dim=rotary_dim)


def _call_apply_rope_pos_ids_inplace(rotary_dim, device):
    q, k, _, _, pos_ids = _rope_inputs(device)
    flashinfer.apply_rope_pos_ids_inplace(q, k, pos_ids, rotary_dim=rotary_dim)


def _call_apply_llama31_rope_inplace(rotary_dim, device):
    q, k, indptr, offsets, _ = _rope_inputs(device)
    flashinfer.apply_llama31_rope_inplace(q, k, indptr, offsets, rotary_dim=rotary_dim)


def _call_apply_llama31_rope_pos_ids_inplace(rotary_dim, device):
    q, k, _, _, pos_ids = _rope_inputs(device)
    flashinfer.apply_llama31_rope_pos_ids_inplace(q, k, pos_ids, rotary_dim=rotary_dim)


WRAPPERS = [
    ("apply_rope_inplace", _call_apply_rope_inplace),
    ("apply_rope_pos_ids_inplace", _call_apply_rope_pos_ids_inplace),
    ("apply_llama31_rope_inplace", _call_apply_llama31_rope_inplace),
    ("apply_llama31_rope_pos_ids_inplace", _call_apply_llama31_rope_pos_ids_inplace),
]
WRAPPER_IDS = [name for name, _ in WRAPPERS]


@pytest.mark.parametrize("rotary_dim", [128, 192])
@pytest.mark.parametrize("name,caller", WRAPPERS, ids=WRAPPER_IDS)
def test_rotary_dim_exceeds_head_dim_raises(name, caller, rotary_dim):
    # The guard fires before any kernel/JIT work, so this runs without a GPU.
    with pytest.raises(ValueError, match=r"head_dim .* must be >= rotary_dim"):
        caller(rotary_dim, device="cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU for kernel"
)
@pytest.mark.parametrize("name,caller", WRAPPERS, ids=WRAPPER_IDS)
def test_rotary_dim_equal_head_dim_does_not_raise_bound(name, caller):
    # rotary_dim == head_dim: the bound guard must NOT fire. The kernel itself
    # needs a CUDA tensor to run, hence the skip above.
    try:
        caller(HEAD_DIM, device="cuda")
    except ValueError as e:
        pytest.fail(f"bound guard fired for {name} with rotary_dim==head_dim: {e}")
