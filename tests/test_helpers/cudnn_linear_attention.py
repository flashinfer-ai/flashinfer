"""Shared gate, oracles and metrics for the cuDNN linear-attention tests.

GDN, GDN-2 and KDA run on the same cuDNN FROST engines, so they share one
availability gate and one family of fp32 token-serial oracles. Both live here
rather than being retyped in ``tests/gdn``, ``tests/gdn2`` and ``tests/kda``.

The oracles hold the state V-major as ``[num_seqs, HO, V, K]``, which is the
layout FlashInfer and cuDNN both use at this boundary. With ``K == V == 128``
every shape check is blind to a transposed state, so the tests pin the
orientation by fit instead -- see ``assert_state_orientation``.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability

try:
    import cudnn

    CUDNN_AVAILABLE = True
    CUDNN_FRONTEND_VERSION = tuple(
        int(part) for part in cudnn.__version__.split(".")[:2]
    )
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False
    CUDNN_FRONTEND_VERSION = (0, 0)

HEAD_DIM = 128

MIN_FRONTEND_VERSION = (1, 28)

SUPPORTED_COMPUTE_CAPABILITIES = frozenset(
    {(10, 0), (10, 1), (10, 2), (10, 3), (10, 7)}
)


def cudnn_linear_attention_unavailable_reason() -> Optional[str]:
    """Why this host cannot run the cuDNN linear-attention engines, or ``None``.

    Only the things a test can know without launching: the package, its
    version, and the device. Everything finer -- head dims, dtypes, head-count
    relations -- is the engine's call, so a test that wants to know whether a
    specific call is served should make the call and read the exception.
    """
    if not torch.cuda.is_available():
        return "CUDA is required"
    if not CUDNN_AVAILABLE:
        return "the cudnn python frontend is not installed"
    if CUDNN_FRONTEND_VERSION < MIN_FRONTEND_VERSION:
        want = ".".join(str(part) for part in MIN_FRONTEND_VERSION)
        return f"cudnn-frontend >= {want} is required, found {cudnn.__version__}"
    capability = get_compute_capability(torch.device("cuda"))
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        return (
            "the cuDNN linear-attention engines are SM100-family only, found "
            f"sm{capability[0]}{capability[1]}"
        )
    return None


_UNAVAILABLE_REASON = cudnn_linear_attention_unavailable_reason()

requires_cudnn_linear_attention = pytest.mark.skipif(
    _UNAVAILABLE_REASON is not None,
    reason=f"cuDNN linear attention unavailable: {_UNAVAILABLE_REASON}",
)


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Frobenius relative error.

    A recurrence accumulates over the whole sequence, so a single late token
    can be elementwise far from the oracle while the run as a whole is right.
    The norm ratio is the metric that survives that; elementwise
    ``assert_close`` is kept for the same-kernel comparisons where it holds.
    """
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-6)
    ).item()


def assert_rel_close(
    prefix: str, actual: torch.Tensor, expected: torch.Tensor, bound: float
):
    assert torch.isfinite(actual).all(), f"{prefix}: non-finite values in actual"
    assert torch.isfinite(expected).all(), f"{prefix}: non-finite values in expected"
    error = rel_err(actual, expected)
    assert error < bound, f"{prefix}: relative error {error:.3e} exceeds {bound:.3e}"


def assert_state_orientation(final_state: torch.Tensor, reference_state: torch.Tensor):
    """Pin the state's V-major orientation by fit.

    ``K == V == 128`` makes every shape and stride check blind to a transposed
    state, so the only evidence is that the transpose fits an order of
    magnitude worse.
    """
    upright = rel_err(final_state, reference_state)
    flipped = rel_err(final_state.transpose(-1, -2).contiguous(), reference_state)
    assert flipped > 10 * upright, (
        f"state orientation is not pinned: upright {upright:.3e} vs "
        f"transposed {flipped:.3e}"
    )


def head_index_maps(num_q_heads: int, num_k_heads: int, num_v_heads: int, device):
    """Per-output-head index maps into q, k and v.

    The output lives at ``HO = max(HQ, HV)`` heads and every input is expanded
    to it by integer division, which is how both cuDNN and FlashInfer group
    GQA and GVA.
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    heads = torch.arange(num_o_heads, device=device)
    return (
        heads // (num_o_heads // num_q_heads),
        heads // (num_o_heads // num_k_heads),
        heads // (num_o_heads // num_v_heads),
    )


def serial_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: float,
    l2norm: bool = False,
):
    """Token-at-a-time fp32 gated delta rule, state V-major as ``[V, K]``.

    One body covers GDN and KDA: ``alpha`` is per-head for GDN (``[T, HO]``)
    and per key channel for KDA (``[T, HO, K]``), and both broadcast against
    the state's trailing K axis once reshaped to ``[HO, 1, -1]``.

    ``alpha`` is linear-space decay in both cases -- the caller converts, the
    same way the wrappers do.
    """
    q_index, k_index, v_index = head_index_maps(
        q.shape[1], k.shape[1], v.shape[1], q.device
    )
    num_o_heads = max(q.shape[1], v.shape[1])
    head_dim, v_dim = q.shape[2], v.shape[2]
    offsets = [int(x) for x in cu_seqlens.tolist()]

    query = q.float()
    key = k.float()
    if l2norm:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
    value = v.float()

    out = torch.empty(
        q.shape[0], num_o_heads, v_dim, dtype=torch.float32, device=q.device
    )
    if initial_state is None:
        state = torch.zeros(
            len(offsets) - 1,
            num_o_heads,
            v_dim,
            head_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.float().clone()

    for sequence in range(len(offsets) - 1):
        running = state[sequence]
        for token in range(offsets[sequence], offsets[sequence + 1]):
            key_t = key[token, k_index]
            if alpha is not None:
                running = running * alpha[token].float().reshape(num_o_heads, 1, -1)
            residual = value[token, v_index] - torch.einsum(
                "hvk,hk->hv", running, key_t
            )
            if beta is not None:
                residual = beta[token].float()[:, None] * residual
            running = running + residual[:, :, None] * key_t[:, None, :]
            out[token] = scale * torch.einsum(
                "hvk,hk->hv", running, query[token, q_index]
            )
        state[sequence] = running
    return out, state


def serial_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    num_householder: int,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: float,
    l2norm: bool = False,
):
    """Token-at-a-time fp32 gated delta product, state V-major as ``[V, K]``.

    GDP is GDN on the expanded sub-token timeline: the decay acts on
    sub-token 0 and the readout on sub-token ``n - 1``, so this expands q and
    ``alpha`` onto k/v/beta's timeline and defers to
    :func:`serial_delta_rule`.
    """
    n = int(num_householder)
    total = q.shape[0]
    q_expanded = q.new_zeros(total * n, *q.shape[1:])
    q_expanded[n - 1 :: n] = q
    alpha_expanded = None
    if alpha is not None:
        alpha_expanded = alpha.new_ones(total * n, *alpha.shape[1:])
        alpha_expanded[0::n] = alpha
    out, state = serial_delta_rule(
        q_expanded,
        k,
        v,
        cu_seqlens * n,
        alpha=alpha_expanded,
        beta=beta,
        initial_state=initial_state,
        scale=scale,
        l2norm=l2norm,
    )
    return out[n - 1 :: n], state


def serial_delta_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: float,
    l2norm: bool = False,
):
    """Token-at-a-time fp32 GDN-2 recurrence, state V-major as ``[V, K]``.

    ``S = diag(alpha) S``; ``v_new = w * v - S (beta * k)``;
    ``S += v_new (x) k``; ``o = scale * S q``. All three gates are per channel:
    ``alpha`` and ``beta`` on K, ``w`` on V.

    Setting all three to channel constants with ``w == beta`` recovers
    :func:`serial_delta_rule`, which the GDN-2 tests exploit as a cross-family
    oracle.
    """
    q_index, k_index, v_index = head_index_maps(
        q.shape[1], k.shape[1], v.shape[1], q.device
    )
    num_o_heads = max(q.shape[1], v.shape[1])
    head_dim, v_dim = q.shape[2], v.shape[2]
    offsets = [int(x) for x in cu_seqlens.tolist()]

    query = q.float()
    key = k.float()
    if l2norm:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
    value = v.float()

    out = torch.empty(
        q.shape[0], num_o_heads, v_dim, dtype=torch.float32, device=q.device
    )
    if initial_state is None:
        state = torch.zeros(
            len(offsets) - 1,
            num_o_heads,
            v_dim,
            head_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.float().clone()

    for sequence in range(len(offsets) - 1):
        running = state[sequence]
        for token in range(offsets[sequence], offsets[sequence + 1]):
            key_t = key[token, k_index]
            if alpha is not None:
                running = running * alpha[token].float()[:, None, :]
            erased = key_t if beta is None else beta[token].float() * key_t
            written = value[token, v_index]
            if w is not None:
                written = w[token].float() * written
            v_new = written - torch.einsum("hvk,hk->hv", running, erased)
            running = running + v_new[:, :, None] * key_t[:, None, :]
            out[token] = scale * torch.einsum(
                "hvk,hk->hv", running, query[token, q_index]
            )
        state[sequence] = running
    return out, state


def kda_safe_gate(
    g: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float
) -> torch.Tensor:
    """The in-kernel safe gate, in linear space.

    ``alpha = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))``, the
    transform cuDNN applies when ``safe_gate`` is on. Returned in linear space
    so it drops straight into :func:`serial_delta_rule`'s ``alpha``.
    """
    return (
        lower_bound
        * torch.sigmoid(A_log.float().exp()[:, None] * (g.float() + dt_bias.float()))
    ).exp()


def packed_offsets(seq_lens, device, dtype=torch.int32) -> torch.Tensor:
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=dtype, device=device)


def widened_view(x: torch.Tensor, axis: int, extra: int = 4) -> torch.Tensor:
    """A non-contiguous view of ``x`` with the same values.

    Allocates a wider buffer along ``axis``, copies ``x`` into the leading
    slice and returns that slice. The innermost dim stays stride-1, which is
    the one layout rule the engines enforce at execute time; every other stride
    is now larger than a contiguous tensor's, so a wrapper that dropped strides
    and assumed compactness reads the wrong elements.
    """
    shape = list(x.shape)
    shape[axis] += extra
    wide = torch.empty(shape, dtype=x.dtype, device=x.device)
    index = [slice(None)] * x.dim()
    index[axis] = slice(0, x.shape[axis])
    wide[tuple(index)] = x
    view = wide[tuple(index)]
    assert not view.is_contiguous()
    return view
