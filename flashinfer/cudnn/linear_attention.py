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

import math
from enum import Enum
from typing import Optional, Union

import torch

from ..api_logging import flashinfer_api
from ..utils import _get_cache_buf

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False

_MIN_FRONTEND_VERSION = (1, 28)

# Global cudnn handle. need to make it per device in future
_cudnn_handle = None


def _check_cudnn_frontend(feature: str) -> None:
    """Fail fast on a frontend that has no linear-attention graph node."""
    if not CUDNN_AVAILABLE:
        raise RuntimeError(
            f"cuDNN {feature} requires the cudnn Python frontend. Install with: "
            "pip install -U 'nvidia-cudnn-frontend[cutedsl]'"
        )
    try:
        version = tuple(int(part) for part in cudnn.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        return
    if version < _MIN_FRONTEND_VERSION:
        want = ".".join(str(part) for part in _MIN_FRONTEND_VERSION)
        raise RuntimeError(
            f"cuDNN {feature} requires cudnn-frontend >= {want}, found "
            f"{cudnn.__version__}. Upgrade with: "
            "pip install -U 'nvidia-cudnn-frontend[cutedsl]'"
        )


def _create_cudnn_handle(stream: torch.cuda.Stream):
    global _cudnn_handle

    if _cudnn_handle is None:
        _cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(_cudnn_handle, stream.cuda_stream)
    return _cudnn_handle


# Tensor ids
class UIDs(Enum):
    RESERVED_INVALID_UID = 0

    Q_UID = 1  # Query tensor
    K_UID = 2  # Key tensor
    V_UID = 3  # Value tensor

    G_UID = 10  # Forget gate, log space
    BETA_UID = 11  # Update / erase gate
    W_UID = 12  # GDN-2 write gate

    CU_SEQLENS_UID = 100  # Packed sequence boundaries

    A_LOG_UID = 150  # Safe-gate log decay rate
    DT_BIAS_UID = 151  # Safe-gate bias

    INITIAL_STATE_UID = 200  # Incoming recurrent state

    O_UID = 1000  # Output tensor
    FINAL_STATE_UID = 1001  # Outgoing recurrent state


def _la_graph_key_fn(
    family: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    o: torch.Tensor,
    *,
    w: Optional[torch.Tensor] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    num_householder: Optional[int] = None,
    scale: float,
    use_qk_l2norm: bool,
    use_beta_sigmoid: bool,
    safe_gate: bool,
    gate_lower_bound: Optional[float],
    batch_invariant: bool,
):
    def layout(t):
        return None if t is None else (t.shape, t.stride(), t.dtype)

    return (
        family,
        layout(q),
        layout(k),
        layout(v),
        layout(g),
        layout(beta),
        layout(w),
        layout(cu_seqlens),
        layout(o),
        layout(initial_state),
        layout(final_state),
        layout(a_log),
        layout(dt_bias),
        num_householder,
        scale,
        use_qk_l2norm,
        use_beta_sigmoid,
        safe_gate,
        gate_lower_bound,
        batch_invariant,
    )


if CUDNN_AVAILABLE:

    @cudnn.jit(heur_modes=[cudnn.heur_mode.A])
    @cudnn.graph_cache(key_fn=_la_graph_key_fn)
    def _build_la_graph(
        family: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        o: torch.Tensor,
        *,
        w: Optional[torch.Tensor] = None,
        a_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        final_state: Optional[torch.Tensor] = None,
        num_householder: Optional[int] = None,
        scale: float,
        use_qk_l2norm: bool,
        use_beta_sigmoid: bool,
        safe_gate: bool,
        gate_lower_bound: Optional[float],
        batch_invariant: bool,
    ):
        handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))

        if not cudnn.datatypes.is_torch_available():
            raise RuntimeError("torch is not available")

        def dtype_of(t):
            return cudnn.datatypes._torch_to_cudnn_data_type(t.dtype)

        with cudnn.graph(handle) as (graph, _):

            def declare(t, name, uid):
                if t is None:
                    return None
                return graph.tensor(
                    name=name,
                    dim=list(t.shape),
                    stride=list(t.stride()),
                    data_type=dtype_of(t),
                ).set_uid(uid.value)

            cudnn_q = declare(q, "q", UIDs.Q_UID)
            cudnn_k = declare(k, "k", UIDs.K_UID)
            cudnn_v = declare(v, "v", UIDs.V_UID)
            cudnn_g = declare(g, "g", UIDs.G_UID)
            cudnn_beta = declare(beta, "beta", UIDs.BETA_UID)
            cudnn_w = declare(w, "w", UIDs.W_UID)
            cudnn_cu_seqlens = declare(cu_seqlens, "cu_seqlens", UIDs.CU_SEQLENS_UID)
            cudnn_a_log = declare(a_log, "a_log", UIDs.A_LOG_UID)
            cudnn_dt_bias = declare(dt_bias, "dt_bias", UIDs.DT_BIAS_UID)
            cudnn_initial_state = declare(
                initial_state, "initial_state", UIDs.INITIAL_STATE_UID
            )

            ports = dict(
                q=cudnn_q,
                k=cudnn_k,
                v=cudnn_v,
                g=cudnn_g,
                beta=cudnn_beta,
                cu_seqlens=cudnn_cu_seqlens,
                initial_state=cudnn_initial_state,
                a_log=cudnn_a_log,
                dt_bias=cudnn_dt_bias,
            )
            attrs = dict(
                scale=scale,
                output_final_state=final_state is not None,
                use_qk_l2norm=use_qk_l2norm,
                use_beta_sigmoid=use_beta_sigmoid,
                safe_gate=safe_gate,
                batch_invariant=batch_invariant,
                name=family,
            )
            if family == "gdn2":
                ports["w"] = cudnn_w
            if family in ("kda", "gdn2"):
                attrs["gate_lower_bound"] = gate_lower_bound
            if family == "gdp":
                attrs["num_householder"] = num_householder

            O, fs, _checkpoints = getattr(graph, family)(**ports, **attrs)

            O.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
                list(o.shape)
            ).set_stride(list(o.stride())).set_data_type(dtype_of(o))
            if fs is not None:
                fs.set_uid(UIDs.FINAL_STATE_UID.value).set_output(True).set_dim(
                    list(final_state.shape)
                ).set_stride(list(final_state.stride())).set_data_type(
                    dtype_of(final_state)
                )

            tensors = [cudnn_q, cudnn_k, cudnn_v, O]
            if fs is not None:
                tensors.append(fs)
            return graph, tensors


def _run_la_graph(
    family: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    o: torch.Tensor,
    *,
    w: Optional[torch.Tensor] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    num_householder: Optional[int] = None,
    scale: float,
    use_qk_l2norm: bool,
    use_beta_sigmoid: bool,
    safe_gate: bool,
    gate_lower_bound: Optional[float],
    batch_invariant: bool,
) -> None:
    graph, _ = _build_la_graph(
        family,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        o,
        w=w,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        final_state=final_state,
        num_householder=num_householder,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        use_beta_sigmoid=use_beta_sigmoid,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        batch_invariant=batch_invariant,
    )

    var_map = {
        UIDs.Q_UID.value: q,
        UIDs.K_UID.value: k,
        UIDs.V_UID.value: v,
        UIDs.G_UID.value: g,
        UIDs.BETA_UID.value: beta,
        UIDs.CU_SEQLENS_UID.value: cu_seqlens,
        UIDs.O_UID.value: o,
    }
    if w is not None:
        var_map[UIDs.W_UID.value] = w
    if a_log is not None:
        var_map[UIDs.A_LOG_UID.value] = a_log
    if dt_bias is not None:
        var_map[UIDs.DT_BIAS_UID.value] = dt_bias
    if initial_state is not None:
        var_map[UIDs.INITIAL_STATE_UID.value] = initial_state
    if final_state is not None:
        var_map[UIDs.FINAL_STATE_UID.value] = final_state

    workspace_buffer = _get_cache_buf(
        "cudnn_linear_attention", max(graph.get_workspace_size(), 1), q.device
    )
    handle = _create_cudnn_handle(torch.cuda.current_stream(q.device))
    graph.execute(var_map, workspace=workspace_buffer, handle=handle)


def _state_out(
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    num_seqs: int,
    num_heads: int,
    head_dim: int,
    v_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Pick the buffer cuDNN writes the final state into."""
    if output_state is not None:
        return output_state
    dtype = torch.float32 if initial_state is None else initial_state.dtype
    return torch.empty(num_seqs, num_heads, v_dim, head_dim, dtype=dtype, device=device)


@flashinfer_api
def cudnn_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule prefill on cuDNN's fused SM100 engine.

    Argument meanings match :func:`flashinfer.chunk_gated_delta_rule`.

    Requires cudnn-frontend 1.28+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself: it declines a graph it cannot serve (the
    per-engine reason lands in the frontend's log).

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[total_seq_len, num_q_heads / num_k_heads / num_v_heads, 128]``,
        packed. Strides are passed through to cuDNN, so only the innermost dim
        has to be contiguous.
    g : torch.Tensor, optional
        Per-head forget gate in linear space (``alpha = exp(log_g)``), shape
        ``[total_seq_len, num_sab_heads]``. cuDNN's gate is natural-log, so
        this is converted; the conversion keeps ``g``'s own dtype, which cuDNN
        reads at float32, bfloat16 or float16. All-ones when ``None``.
    beta : torch.Tensor, optional
        Per-head update gate ``[total_seq_len, num_sab_heads]``, post-sigmoid,
        in float32 or ``q.dtype``. All-ones when ``None``.
    scale : float, optional
        Query scale; ``1 / sqrt(head_dim)`` when ``None`` or ``0.0``, matching
        the native GDN path.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, num_sab_heads, 128, 128]``, V-major, float32 or
        bfloat16. cuDNN uses the same layout, so these pass through
        untransposed and ``output_state`` is written in place by the kernel.
        ``output_state`` must not alias ``initial_state``: the engines split
        one sequence across CTAs, so the chunk-0 CTA reading the incoming
        state would race the last-chunk CTA writing the outgoing one. Like the
        state-slot uniqueness ``state_indices`` relies on, this is a caller
        precondition rather than a launch-time check.
    output_final_state : bool
        Return the outgoing recurrent state alongside the output; when unset
        no state is written at all.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32 or int64. Required.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel.
    output : torch.Tensor, optional
        Pre-allocated ``[total_seq_len, num_o_heads, 128]``, written in place
        by the kernel.
    batch_invariant : bool
        Disable the split-K partition so the reduction order, and hence the
        result, does not depend on how sequences are batched. Costs the
        parallelism split-K exists to create on few long sequences and saves
        its fixed scheduling cost on many short ones.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``output``, or ``(output, final_state)`` when ``output_final_state``.
    """
    _check_cudnn_frontend("chunk_gated_delta_rule")
    if cu_seqlens is None:
        raise ValueError("cudnn_chunk_gated_delta_rule: cu_seqlens is required")

    total, num_q_heads, head_dim = q.shape
    v_dim = v.shape[2]
    num_sab_heads = max(num_q_heads, v.shape[1])

    g_log = (
        torch.zeros(total, num_sab_heads, dtype=q.dtype, device=q.device)
        if g is None
        else torch.log(g)
    )
    if beta is None:
        beta = torch.ones(total, num_sab_heads, dtype=torch.float32, device=q.device)

    if output is None:
        output = torch.empty(
            total, num_sab_heads, v_dim, dtype=q.dtype, device=q.device
        )
    num_seqs = cu_seqlens.shape[0] - 1
    final_state = (
        _state_out(
            initial_state,
            output_state,
            num_seqs,
            num_sab_heads,
            head_dim,
            v_dim,
            q.device,
        )
        if output_final_state
        else None
    )

    _run_la_graph(
        "gdn",
        q,
        k,
        v,
        g_log,
        beta,
        cu_seqlens,
        output,
        initial_state=initial_state,
        final_state=final_state,
        scale=float(scale) if scale else 1.0 / math.sqrt(head_dim),
        use_qk_l2norm=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid=False,
        safe_gate=False,
        gate_lower_bound=None,
        batch_invariant=bool(batch_invariant),
    )
    if not output_final_state:
        return output
    return output, final_state


@flashinfer_api
def cudnn_chunk_gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    num_householder: int = 1,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated DeltaProduct prefill on cuDNN's fused SM100 engine.

    GDP applies ``n = num_householder`` beta-gated Householder updates per
    token with one per-head scalar decay per token: the GDN recurrence on an
    expanded sub-token timeline, with the decay acting before the token's
    updates and the readout following the last one. ``num_householder == 1``
    is exactly :func:`cudnn_chunk_gated_delta_rule`.

    Requires cudnn-frontend 1.28+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself: it declines a graph it cannot serve (the
    per-engine reason lands in the frontend's log).

    Parameters
    ----------
    q : torch.Tensor
        ``[total_seq_len, num_q_heads, head_size]``, packed at real-token
        rows. Strides are passed through to cuDNN, so only the innermost dim
        has to be contiguous.
    k, v : torch.Tensor
        ``[total_seq_len * num_householder, num_k_heads / num_v_heads,
        head_size]``, packed on the expanded sub-token timeline: the ``n``
        Householder updates of token ``t`` occupy rows ``t*n .. t*n + n - 1``.
        ``num_k_heads`` must equal ``num_q_heads`` or ``num_v_heads``.
    g : torch.Tensor, optional
        Per-head forget gate in linear space (``alpha = exp(log_g)``), shape
        ``[total_seq_len, num_sab_heads]`` at real-token rows. cuDNN's gate is
        natural-log, so this is converted; the conversion keeps ``g``'s own
        dtype. All-ones when ``None``.
    beta : torch.Tensor, optional
        Per-head, per-Householder update gate
        ``[total_seq_len * num_householder, num_sab_heads]``, post-sigmoid,
        in float32 or ``q.dtype``. All-ones when ``None``.
    num_householder : int
        Householder updates per token (``n >= 1``).
    scale : float, optional
        Query scale; ``1 / sqrt(head_size)`` when ``None`` or ``0.0``,
        matching the native GDN path.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, num_sab_heads, head_size, head_size]``, V-major,
        float32 or bfloat16. ``output_state`` must not alias
        ``initial_state``; see :func:`cudnn_chunk_gated_delta_rule`.
    output_final_state : bool
        Return the outgoing recurrent state alongside the output; when unset
        no state is written at all.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32 or int64, over the real tokens. Required.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel.
    output : torch.Tensor, optional
        Pre-allocated ``[total_seq_len, num_o_heads, head_size]`` at
        real-token rows, written in place by the kernel.
    batch_invariant : bool
        Disable the split-K partition; see
        :func:`cudnn_chunk_gated_delta_rule`.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``output``, or ``(output, final_state)`` when ``output_final_state``.
    """
    _check_cudnn_frontend("chunk_gated_delta_product")
    if cu_seqlens is None:
        raise ValueError("cudnn_chunk_gated_delta_product: cu_seqlens is required")

    total, num_q_heads, head_dim = q.shape
    v_dim = v.shape[2]
    num_sab_heads = max(num_q_heads, v.shape[1])
    n = int(num_householder)

    g_log = (
        torch.zeros(total, num_sab_heads, dtype=q.dtype, device=q.device)
        if g is None
        else torch.log(g)
    )
    if beta is None:
        beta = torch.ones(
            total * n, num_sab_heads, dtype=torch.float32, device=q.device
        )

    if output is None:
        output = torch.empty(
            total, num_sab_heads, v_dim, dtype=q.dtype, device=q.device
        )
    num_seqs = cu_seqlens.shape[0] - 1
    final_state = (
        _state_out(
            initial_state,
            output_state,
            num_seqs,
            num_sab_heads,
            head_dim,
            v_dim,
            q.device,
        )
        if output_final_state
        else None
    )

    _run_la_graph(
        "gdp",
        q,
        k,
        v,
        g_log,
        beta,
        cu_seqlens,
        output,
        initial_state=initial_state,
        final_state=final_state,
        num_householder=n,
        scale=float(scale) if scale else 1.0 / math.sqrt(head_dim),
        use_qk_l2norm=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid=False,
        safe_gate=False,
        gate_lower_bound=None,
        batch_invariant=bool(batch_invariant),
    )
    if not output_final_state:
        return output
    return output, final_state


@flashinfer_api
def cudnn_chunk_gated_delta_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule 2 prefill on cuDNN's fused SM100 engine.

    Argument meanings match :func:`flashinfer.chunk_gated_delta_rule2`.

    GDN-2 generalizes GDN's per-head scalar gates to channel-wise ones: the
    forget gate ``g`` and erase gate ``beta`` are per key channel and the write
    gate ``w`` is per value channel.

    .. math::

        S_t &= \mathrm{diag}(g_t) S_{t-1} \\
        v^{new}_t &= w_t \odot v_t - (\beta_t \odot k_t)^\top S_t \\
        S_t &\mathrel{+}= k_t \otimes v^{new}_t \\
        o_t &= \mathrm{scale} \cdot q_t^\top S_t

    Requires cudnn-frontend 1.28+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[total_seq_len, num_q_heads / num_k_heads / num_v_heads, 128]``,
        packed.
    g : torch.Tensor, optional
        Channel-wise forget gate in linear space, shape
        ``[total_seq_len, num_sab_heads, 128]``. Converted to cuDNN's log-space
        gate at ``g``'s own dtype, which cuDNN reads at float32, bfloat16 or
        float16. All-ones when ``None``.
    beta : torch.Tensor, optional
        Channel-wise erase gate ``[total_seq_len, num_sab_heads, 128]``,
        converted to ``q.dtype``. All-ones when ``None``.
    w : torch.Tensor, optional
        Channel-wise write gate ``[total_seq_len, num_sab_heads, 128]``,
        converted to ``q.dtype``. All-ones when ``None``.
    scale : float, optional
        Query scale; ``1 / sqrt(head_dim)`` when ``None`` or ``0.0``, matching
        the native GDN path.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, num_sab_heads, 128, 128]``, V-major, float32 or
        bfloat16. ``output_state`` must not alias ``initial_state``; see
        :func:`cudnn_chunk_gated_delta_rule`.
    output_final_state : bool
        Return the outgoing recurrent state alongside the output; when unset
        no state is written at all.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32 or int64. Required.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel.
    output : torch.Tensor, optional
        Pre-allocated ``[total_seq_len, num_o_heads, 128]``.
    batch_invariant : bool
        Disable the split-K partition; see
        :func:`cudnn_chunk_gated_delta_rule`.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``output``, or ``(output, final_state)`` when ``output_final_state``.
    """
    _check_cudnn_frontend("chunk_gated_delta_rule2")
    if cu_seqlens is None:
        raise ValueError("cudnn_chunk_gated_delta_rule2: cu_seqlens is required")

    total, num_q_heads, head_dim = q.shape
    v_dim = v.shape[2]
    num_sab_heads = max(num_q_heads, v.shape[1])

    g_log = (
        torch.zeros(total, num_sab_heads, head_dim, dtype=q.dtype, device=q.device)
        if g is None
        else torch.log(g)
    )
    if beta is None:
        beta = torch.ones(
            total, num_sab_heads, head_dim, dtype=q.dtype, device=q.device
        )
    elif beta.dtype != q.dtype:
        beta = beta.to(q.dtype)
    if w is None:
        w = torch.ones(total, num_sab_heads, v_dim, dtype=q.dtype, device=q.device)
    elif w.dtype != q.dtype:
        w = w.to(q.dtype)

    if output is None:
        output = torch.empty(
            total, num_sab_heads, v_dim, dtype=q.dtype, device=q.device
        )
    num_seqs = cu_seqlens.shape[0] - 1
    final_state = (
        _state_out(
            initial_state,
            output_state,
            num_seqs,
            num_sab_heads,
            head_dim,
            v_dim,
            q.device,
        )
        if output_final_state
        else None
    )

    _run_la_graph(
        "gdn2",
        q,
        k,
        v,
        g_log,
        beta,
        cu_seqlens,
        output,
        w=w,
        initial_state=initial_state,
        final_state=final_state,
        scale=float(scale) if scale else 1.0 / math.sqrt(head_dim),
        use_qk_l2norm=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid=False,
        safe_gate=False,
        gate_lower_bound=None,
        batch_invariant=bool(batch_invariant),
    )
    if not output_final_state:
        return output
    return output, final_state


@flashinfer_api
def cudnn_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Kimi Delta Attention prefill on cuDNN's fused SM100 engine.

    Argument meanings match :func:`flashinfer.recurrent_kda`, restricted to the
    ordinary multi-token prefill subset: no speculative decode, no state pool,
    no ``initial_state_source``.

    Requires cudnn-frontend 1.28+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[1, total_tokens, H, 128]`` or ``[total_tokens, H, 128]``, bfloat16
        or float16.
    g : torch.Tensor
        Channel-wise gate ``[..., total_tokens, HV, 128]``. Log-space unless
        ``use_gate_in_kernel``, in which case it is the raw pre-activation and
        cuDNN applies the safe-gate transform from ``A_log`` / ``dt_bias`` /
        ``lower_bound``. float32, bfloat16 or float16; cuDNN takes all three
        and only the gate's memory format follows the choice, so this is
        forwarded with no copy. In float16 the kernel's chunk-cumulative decay
        inverse bounds how strong the decay may be (roughly ``alpha >= 0.9``
        per token per channel before it overflows); bfloat16 carries an
        fp32-like exponent and has no such bound.
    beta : torch.Tensor
        ``[..., total_tokens, HV]``. Post-sigmoid in float32 or ``q.dtype``,
        or ``q.dtype`` logits when ``beta_is_logit``.
    A_log, dt_bias : torch.Tensor, optional
        Safe-gate parameters, required together when ``use_gate_in_kernel``.
    scale : float, optional
        Query scale; ``1 / sqrt(head_dim)`` when ``None``.
    output_final_state : bool
        Return the final state alongside the output. This gates only the
        return value; see ``initial_state`` for when a state is written.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel.
    use_gate_in_kernel : bool
        Read ``g`` as the raw pre-activation and apply the safe-gate transform
        from ``A_log`` / ``dt_bias`` / ``lower_bound`` in the kernel.
    beta_is_logit : bool
        Read ``beta`` as logits and apply the sigmoid in the kernel.
    lower_bound : float, optional
        Safe-gate lower bound, forwarded as cuDNN's ``gate_lower_bound``.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32 or int64. Required.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, HV, 128, 128]``, V-major, float32 or bfloat16.
        Following the Cake and CuTe DSL prefill backends, ``initial_state`` is
        advanced to the final state whenever one is given and no separate
        ``output_state`` is supplied, independently of
        ``output_final_state`` -- which gates only what is returned.
        ``output_state`` must not alias ``initial_state``; see
        :func:`cudnn_chunk_gated_delta_rule`.
    output : torch.Tensor, optional
        Pre-allocated output, written in place by the kernel.
    batch_invariant : bool
        Disable the split-K partition; see
        :func:`cudnn_chunk_gated_delta_rule`.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        ``(output, final_state)``, with ``final_state`` ``None`` when
        ``output_final_state=False``.
    """
    _check_cudnn_frontend("recurrent_kda")
    if cu_seqlens is None:
        raise ValueError("cudnn_recurrent_kda: cu_seqlens is required")
    if use_gate_in_kernel and (A_log is None or dt_bias is None):
        raise ValueError(
            "cudnn_recurrent_kda: use_gate_in_kernel requires both A_log and dt_bias"
        )

    out_shape = tuple(v.shape)
    q = q.squeeze(0) if q.dim() == 4 else q
    k = k.squeeze(0) if k.dim() == 4 else k
    v = v.squeeze(0) if v.dim() == 4 else v
    g = g.squeeze(0) if g.dim() == 4 else g
    beta = beta.squeeze(0) if beta.dim() == 3 else beta
    head_dim = q.shape[-1]
    v_dim = v.shape[2]
    if v.shape[1] < q.shape[1]:
        raise NotImplementedError(
            f"cudnn_recurrent_kda: cuDNN carries the KDA state at max(H, HV) heads, "
            f"FlashInfer at HV; got H={q.shape[1]} > HV={v.shape[1]}"
        )
    num_heads = v.shape[1]
    if beta_is_logit and beta.dtype != q.dtype:
        beta = beta.to(q.dtype)

    if output is None:
        output = torch.empty(
            q.shape[0], num_heads, v_dim, dtype=q.dtype, device=q.device
        )
    o = output.squeeze(0) if output.dim() == 4 else output
    num_seqs = cu_seqlens.shape[0] - 1
    final_state = (
        _state_out(
            initial_state, output_state, num_seqs, num_heads, head_dim, v_dim, q.device
        )
        if (output_final_state or output_state is not None or initial_state is not None)
        else None
    )

    _run_la_graph(
        "kda",
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        o,
        a_log=A_log.float() if use_gate_in_kernel else None,
        dt_bias=(dt_bias.float().reshape(-1, head_dim) if use_gate_in_kernel else None),
        initial_state=initial_state,
        final_state=final_state,
        scale=float(scale) if scale is not None else 1.0 / math.sqrt(head_dim),
        use_qk_l2norm=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid=bool(beta_is_logit),
        safe_gate=bool(use_gate_in_kernel),
        gate_lower_bound=float(lower_bound) if lower_bound is not None else None,
        batch_invariant=bool(batch_invariant),
    )

    if output_state is None and initial_state is not None:
        initial_state.copy_(final_state)
        final_state = initial_state
    return output.reshape(out_shape), final_state if output_final_state else None
