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

import inspect
import math
from collections import OrderedDict
from enum import Enum
from functools import cache
from importlib.metadata import version as package_version
from threading import Lock
from typing import Optional, Union

import torch

from ..api_logging import flashinfer_api
from ..utils import _get_cache_buf, get_device_index

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False

_MIN_FRONTEND_VERSION = (1, 29)


def _linear_gate(g: Optional[torch.Tensor], ones_shape, dtype, device):
    """The GDN / GDP forget gate for the graph as linear alpha
    (``gate_domain="linear"``); all-ones when ``None``."""
    if g is None:
        g = torch.ones(*ones_shape, dtype=dtype, device=device)
    return g


# One cuDNN handle per device: a handle binds the device that is current when
# it is created and only takes streams of that device.
_cudnn_handles: dict = {}


# Auto may recover from marked build failures; explicit APIs retain the exact
# original exception type, object and traceback. The marker is never set by
# execute, where retrying could advance a recurrent state twice.
_LINEAR_ATTENTION_BUILD_ERRORS = (NotImplementedError, TypeError) + (
    (cudnn.cudnnGraphNotSupportedError,) if CUDNN_AVAILABLE else ()
)

# Process-local support decisions, never tensors, pointers or exception objects.
# Keep successful auto calls free of descriptor-key work until a build declines.
_LA_AUTO_DECLINE_LIMIT = 128
_la_auto_declines: OrderedDict = OrderedDict()
_la_auto_decline_scopes: dict = {}
_la_auto_decline_lock = Lock()


@cache
def _la_auto_runtime_versions(frontend, frontend_version):
    # Loaded backend/DSL libraries are fixed for this process. Include frontend
    # identity/version so replacing the frontend cannot inherit old declines.
    return frontend.backend_version(), package_version("nvidia-cutlass-dsl")


def _la_auto_decline_scope(call, q):
    return call, q.device, q.shape, q.dtype


def _la_auto_decline_key(scope, args, kwargs):
    def descriptor(value):
        if isinstance(value, torch.Tensor):
            return (
                value.device,
                value.shape,
                value.stride(),
                value.dtype,
                value.requires_grad,
                value.is_inference(),
            )
        return value

    return (
        scope,
        _build_la_graph,
        cudnn,
        cudnn.__version__,
        _la_auto_runtime_versions(cudnn, cudnn.__version__),
        torch.is_inference_mode_enabled(),
        tuple(descriptor(value) for value in args),
        tuple((name, descriptor(value)) for name, value in sorted(kwargs.items())),
    )


def _try_cudnn_auto(call, *args, **kwargs):
    """Return None only for a deterministic, pre-execution support decline.

    Callers must validate native eligibility and storage aliases before entry.
    Explicit cuDNN calls bypass this cache and retain their original diagnostics.
    """
    key = None
    scope = None
    known_scopes = _la_auto_decline_scopes.get(call)
    if known_scopes:
        scope = _la_auto_decline_scope(call, args[0])
    if known_scopes and scope in known_scopes:
        key = _la_auto_decline_key(scope, args, kwargs)
        with _la_auto_decline_lock:
            if key in _la_auto_declines:
                _la_auto_declines.move_to_end(key)
                return None
    try:
        return call(*args, **kwargs)
    except _LINEAR_ATTENTION_BUILD_ERRORS as exc:
        if not getattr(exc, "_fi_la_build_unsupported", False):
            raise
        # A build can fail only because it was attempted during capture. Do
        # not turn that contextual failure into a lasting eager support claim.
        q = args[0]
        if q.is_cuda:
            with torch.cuda.device(q.device):
                if torch.cuda.is_current_stream_capturing():
                    return None
        if scope is None:
            scope = _la_auto_decline_scope(call, q)
        if key is None:
            key = _la_auto_decline_key(scope, args, kwargs)
        with _la_auto_decline_lock:
            _la_auto_declines[key] = None
            _la_auto_decline_scopes.setdefault(call, set()).add(scope)
            _la_auto_declines.move_to_end(key)
            if len(_la_auto_declines) > _LA_AUTO_DECLINE_LIMIT:
                old_key, _ = _la_auto_declines.popitem(last=False)
                # Scan only on cold eviction; warm calls for other families
                # or shapes never construct full descriptor keys.
                if not any(k[0] == old_key[0] for k in _la_auto_declines):
                    old_call = old_key[0][0]
                    old_scopes = _la_auto_decline_scopes[old_call]
                    old_scopes.discard(old_key[0])
                    if not old_scopes:
                        del _la_auto_decline_scopes[old_call]
        return None


def _linear_attention_auto_available() -> bool:
    """Shared optional-runtime floor; callers cache their lazy probe."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import InvalidVersion, Version

    if not CUDNN_AVAILABLE:
        return False
    try:
        return Version(cudnn.__version__) >= Version("1.31") and (
            Version(version("nvidia-cutlass-dsl")) >= Version("4.7")
        )
    except (AttributeError, InvalidVersion, PackageNotFoundError):
        return False


def _mamba2_auto_available() -> bool:
    """The Mamba-2 node floor: frontend 1.32+ with CuTe DSL 4.7+."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import InvalidVersion, Version

    if not CUDNN_AVAILABLE:
        return False
    try:
        return Version(cudnn.__version__) >= Version("1.32") and (
            Version(version("nvidia-cutlass-dsl")) >= Version("4.7")
        )
    except (AttributeError, InvalidVersion, PackageNotFoundError):
        return False


def _check_cudnn_frontend(feature: str, minimum=_MIN_FRONTEND_VERSION) -> None:
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
    if version < minimum:
        want = ".".join(str(part) for part in minimum)
        raise RuntimeError(
            f"cuDNN {feature} requires cudnn-frontend >= {want}, found "
            f"{cudnn.__version__}. Upgrade with: "
            "pip install -U 'nvidia-cudnn-frontend[cutedsl]'"
        )


def _create_cudnn_handle(stream: torch.cuda.Stream):
    handle = _cudnn_handles.get(stream.device_index)
    if handle is None:
        with torch.cuda.device(stream.device_index):
            handle = cudnn.create_handle()
        _cudnn_handles[stream.device_index] = handle
    cudnn.set_stream(handle, stream.cuda_stream)
    return handle


# Tensor ids
class UIDs(Enum):
    RESERVED_INVALID_UID = 0

    Q_UID = 1  # Query tensor
    K_UID = 2  # Key tensor
    V_UID = 3  # Value tensor

    G_UID = 10  # Forget gate
    BETA_UID = 11  # Update / erase gate
    W_UID = 12  # GDN-2 write gate

    CU_SEQLENS_UID = 100  # Packed sequence boundaries

    A_LOG_UID = 150  # Safe-gate log decay rate
    DT_BIAS_UID = 151  # Safe-gate bias
    A_UID = 152  # Mamba-2 per-head decay rate
    D_UID = 153  # Mamba-2 skip connection
    Z_UID = 154  # Mamba-2 output gate

    INITIAL_STATE_UID = 200  # Incoming recurrent state
    STATE_INDICES_UID = 201  # Sequence-to-pool row mapping

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
    gate_domain: str = "log",
    overwrite_initial_state: bool = False,
    state_indices: Optional[torch.Tensor] = None,
    a: Optional[torch.Tensor] = None,
    d: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
):
    def layout(t):
        return None if t is None else (t.shape, t.stride(), t.dtype)

    return (
        family,
        get_device_index(q.device),
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
        gate_domain,
        overwrite_initial_state,
        layout(state_indices),
        layout(a),
        layout(d),
        layout(z),
        dt_softplus,
        dt_min,
        dt_max,
    )


if CUDNN_AVAILABLE:

    @cudnn.jit(heur_modes=[cudnn.heur_mode.A])
    def _create_la_graph(
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
        gate_domain: str = "log",
        overwrite_initial_state: bool = False,
        state_indices: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_softplus: bool = True,
        dt_min: float = 0.0,
        dt_max: Optional[float] = None,
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
            cudnn_state_indices = declare(
                state_indices, "state_indices", UIDs.STATE_INDICES_UID
            )
            cudnn_a = declare(a, "a", UIDs.A_UID)
            cudnn_d = declare(d, "d", UIDs.D_UID)
            cudnn_z = declare(z, "z", UIDs.Z_UID)

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
            if cudnn_state_indices is not None:
                ports["state_indices"] = cudnn_state_indices
            if family in ("kda", "gdn2"):
                attrs["gate_lower_bound"] = gate_lower_bound
            if family == "gdp":
                attrs["num_householder"] = num_householder
            attrs["gate_domain"] = gate_domain
            if family == "mamba2":
                # Mamba-2 in the delta-rule vocabulary: q = C, k = B, v = x,
                # g = the raw dt and a = A. No erase gate, no scale.
                ports = dict(
                    q=cudnn_q,
                    k=cudnn_k,
                    v=cudnn_v,
                    g=cudnn_g,
                    a=cudnn_a,
                    cu_seqlens=cudnn_cu_seqlens,
                    initial_state=cudnn_initial_state,
                    dt_bias=cudnn_dt_bias,
                    d=cudnn_d,
                    z=cudnn_z,
                )
                if cudnn_state_indices is not None:
                    ports["state_indices"] = cudnn_state_indices
                attrs = dict(
                    output_final_state=final_state is not None,
                    dt_softplus=dt_softplus,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    batch_invariant=batch_invariant,
                    name=family,
                )
            if overwrite_initial_state:
                attrs["overwrite_initial_state"] = True

            if family == "mamba2":
                O, fs, _checkpoints, _pre_gate = graph.mamba2(**ports, **attrs)
            else:
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
            # Only the binding order is cached. Operands, workspace and stream
            # are observed afresh on every execute, including graph capture.
            graph._fi_la_uids = tuple(
                uid.value
                for uid, tensor in (
                    (UIDs.Q_UID, q),
                    (UIDs.K_UID, k),
                    (UIDs.V_UID, v),
                    (UIDs.G_UID, g),
                    (UIDs.BETA_UID, beta),
                    (UIDs.CU_SEQLENS_UID, cu_seqlens),
                    (UIDs.O_UID, o),
                    (UIDs.W_UID, w),
                    (UIDs.A_LOG_UID, a_log),
                    (UIDs.DT_BIAS_UID, dt_bias),
                    (UIDs.A_UID, a),
                    (UIDs.D_UID, d),
                    (UIDs.Z_UID, z),
                    (UIDs.INITIAL_STATE_UID, initial_state),
                    (UIDs.FINAL_STATE_UID, final_state),
                    (UIDs.STATE_INDICES_UID, state_indices),
                )
                if tensor is not None
            )
            return graph, tensors

    @cudnn.graph_cache(key_fn=_la_graph_key_fn)
    def _build_la_graph(*args, overwrite_initial_state=False, **kwargs):
        # Cache the completed build, including a compatibility fallback. A
        # failed overwrite graph must neither enter the cache nor be retried
        # on every warm call. This is an FE capability, not a backend version.
        try:
            version = tuple(int(part) for part in cudnn.__version__.split(".")[:2])
        except (AttributeError, ValueError):
            version = ()
        overwrite = overwrite_initial_state and (
            version >= (1, 30) or kwargs.get("state_indices") is not None
        )
        try:
            result = _create_la_graph(
                *args, overwrite_initial_state=overwrite, **kwargs
            )
        except (cudnn.cudnnGraphNotSupportedError, NotImplementedError):
            if not overwrite or kwargs.get("state_indices") is not None:
                raise
            result = _create_la_graph(*args, **kwargs)
            overwrite = False
        graph = result[0]
        graph._fi_la_overwrite = overwrite
        graph._fi_la_workspace_size = max(graph.get_workspace_size(), 1)
        try:
            graph._fi_la_ordered = (
                "tensor_uids" in inspect.signature(graph.execute).parameters
            )
        except (TypeError, ValueError):
            graph._fi_la_ordered = False
        return result


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
    gate_domain: str = "log",
    overwrite_initial_state: bool = False,
    state_indices: Optional[torch.Tensor] = None,
    a: Optional[torch.Tensor] = None,
    d: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
) -> Optional[torch.Tensor]:
    try:
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
            gate_domain=gate_domain,
            overwrite_initial_state=overwrite_initial_state,
            state_indices=state_indices,
            a=a,
            d=d,
            z=z,
            dt_softplus=dt_softplus,
            dt_min=dt_min,
            dt_max=dt_max,
        )
    except (cudnn.cudnnGraphNotSupportedError, NotImplementedError) as exc:
        exc.__dict__["_fi_la_build_unsupported"] = True
        raise

    if overwrite_initial_state and not graph._fi_la_overwrite:
        # The overwrite candidate has a compact state descriptor. Older FE or
        # a plan that cannot overwrite retains the separate scratch + copy.
        assert final_state is not None
        final_state = torch.empty_like(final_state)

    # Mamba-2 has no erase gate: beta is None and skipped, as in the graph.
    buffers: tuple[torch.Tensor, ...] = tuple(
        t for t in (q, k, v, g, beta, cu_seqlens, o) if t is not None
    )
    if w is not None:
        buffers += (w,)
    if a_log is not None:
        buffers += (a_log,)
    if dt_bias is not None:
        buffers += (dt_bias,)
    for extra in (a, d, z):
        if extra is not None:
            buffers += (extra,)
    if initial_state is not None:
        buffers += (initial_state,)
    if final_state is not None:
        buffers += (final_state,)
    if state_indices is not None:
        buffers += (state_indices,)

    workspace_size = graph._fi_la_workspace_size
    stream = torch.cuda.current_stream(q.device)
    if q.device.index == torch.cuda.current_device():
        capturing = torch.cuda.is_current_stream_capturing()
    else:
        # Query capture status on the operand's device.
        with torch.cuda.device(q.device):
            capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        # Separate captures need private scratch when replayed concurrently.
        workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=q.device
        )
    else:
        workspace_buffer = _get_cache_buf(
            f"cudnn_linear_attention_{stream.cuda_stream}", workspace_size, q.device
        )
    handle = _create_cudnn_handle(stream)
    if graph._fi_la_ordered:
        graph.execute(
            buffers,
            workspace=workspace_buffer,
            handle=handle,
            tensor_uids=graph._fi_la_uids,
        )
    else:
        graph.execute(
            dict(zip(graph._fi_la_uids, buffers, strict=True)),
            workspace=workspace_buffer,
            handle=handle,
        )
    return final_state


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


def _validate_la_state_pool(
    state_indices,
    initial_state,
    output_state,
    output_final_state,
    num_seqs,
    num_heads,
    head_dim,
    v_dim,
    device,
):
    """Validate pool metadata without reading device-side slot ids."""
    if state_indices.dtype != torch.int32:
        raise ValueError("cuDNN linear-attention state_indices must have dtype int32")
    if state_indices.shape != (num_seqs,) or not state_indices.is_contiguous():
        raise ValueError(
            "cuDNN linear-attention state_indices must be contiguous [num_seqs]"
        )
    if state_indices.device != device:
        raise ValueError("cuDNN linear-attention state_indices must be on q's device")
    if initial_state is None:
        raise ValueError("state_indices requires an initial_state pool")
    if output_final_state and output_state is None:
        raise ValueError("state_indices requires an explicit output_state pool")
    for name, pool in (
        ("initial_state", initial_state),
        ("output_state", output_state if output_final_state else None),
    ):
        if pool is None:
            continue
        if pool.ndim != 4 or pool.shape[1:] != (num_heads, v_dim, head_dim):
            raise ValueError(
                f"{name} must have shape [N_pool, {num_heads}, {v_dim}, {head_dim}]"
            )
        if pool.shape[0] <= 0 or pool.device != device:
            raise ValueError(f"{name} must be a nonempty pool on q's device")
        if pool.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError(
                f"cuDNN linear attention {name} must have dtype float32, bfloat16 or float16"
            )
        if (
            pool.stride()[1:] != (v_dim * head_dim, head_dim, 1)
            or pool.stride(0) < num_heads * v_dim * head_dim
            or pool.stride(0) * pool.element_size() % 16
            or pool.data_ptr() % 16
        ):
            raise ValueError(
                f"{name} must have dense [H, V, K] rows with a 16-byte aligned slot stride"
            )
    if not output_final_state:
        return False
    if (
        output_state.shape != initial_state.shape
        or output_state.dtype != initial_state.dtype
    ):
        raise ValueError(
            "cuDNN linear-attention state pools must have the same shape and dtype"
        )
    same_view = (
        output_state.data_ptr() == initial_state.data_ptr()
        and output_state.stride() == initial_state.stride()
    )

    def byte_span(pool):
        begin = pool.data_ptr()
        elements = (pool.shape[0] - 1) * pool.stride(0) + num_heads * v_dim * head_dim
        return begin, begin + elements * pool.element_size()

    input_begin, input_end = byte_span(initial_state)
    output_begin, output_end = byte_span(output_state)
    # DLPack wrappers can share addresses while having different Storage objects.
    if not same_view and input_begin < output_end and output_begin < input_end:
        raise ValueError(
            "cuDNN linear-attention state pools must be identical views or use disjoint memory"
        )
    if same_view and torch.is_grad_enabled() and initial_state.requires_grad:
        raise RuntimeError(
            "cuDNN linear attention cannot overwrite a state pool that requires gradients"
        )
    if (
        same_view
        and torch.is_inference(initial_state)
        and not torch.is_inference_mode_enabled()
    ):
        raise RuntimeError(
            "cuDNN linear attention cannot overwrite an inference state pool outside inference_mode"
        )
    return same_view


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
    state_indices: Optional[torch.Tensor] = None,
    *,
    gate_domain: str = "linear",
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule prefill on cuDNN's fused SM100 engine.

    Argument meanings match :func:`flashinfer.chunk_gated_delta_rule`.

    Requires cudnn-frontend 1.31.0+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself: it declines a graph it cannot serve (the
    per-engine reason lands in the frontend's log).

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[total_seq_len, num_q_heads / num_k_heads, K]`` and
        ``[total_seq_len, num_v_heads, V]`` with K and V each 64 or 128,
        packed. Strides are passed through to cuDNN, so only the innermost dim
        has to be contiguous.
    g : torch.Tensor, optional
        Per-head forget gate in linear space (``alpha = exp(log_g)``), shape
        ``[total_seq_len, num_sab_heads]``, passed to cuDNN as is
        (``gate_domain="linear"``) at ``g``'s own dtype, which cuDNN reads at
        float32, bfloat16 or float16. All-ones when ``None``.
    beta : torch.Tensor, optional
        Per-head update gate ``[total_seq_len, num_sab_heads]``, post-sigmoid,
        in float32 or ``q.dtype``. All-ones when ``None``.
    scale : float, optional
        Query scale; ``1 / sqrt(head_dim)`` when ``None`` or ``0.0``, matching
        the native GDN path.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, num_sab_heads, V, K]``, V-major, float32,
        bfloat16 or float16. cuDNN uses the same layout, so these pass through
        untransposed and ``output_state`` is written in place by the kernel.
        Without ``state_indices``, ``output_state`` must not alias ``initial_state``: the engines split
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
        Normalize Q/K with additive epsilon ``1e-6`` and round to the input
        dtype inside the frontend, matching the public GDN API.
    output : torch.Tensor, optional
        Pre-allocated ``[total_seq_len, num_o_heads, V]``, written in place
        by the kernel.
    batch_invariant : bool
        Disable the split-K partition so the reduction order, and hence the
        result, does not depend on how sequences are batched. Costs the
        parallelism split-K exists to create on few long sequences and saves
        its fixed scheduling cost on many short ones.
    state_indices : torch.Tensor, optional
        Contiguous int32 ``[num_seqs]`` pool slots on q's device, requiring
        cuDNN frontend 1.31+. Both states are pools ``[N_pool, H, V, K]``
        with dense inner rows and a 16-byte aligned slot stride. Slot ids
        must be unique and in range; these device values are a caller
        precondition and are not read back. Returning state requires an
        explicit output pool of the same shape/dtype, either the identical
        input view (a safe in-place plan) or disjoint memory. Unselected
        rows remain untouched. No pool is written when output_final_state=False.
    gate_domain : {"linear", "log"}
        Precomputed decay domain, matching the public GDN API.
    use_gate_in_kernel : bool
        Fuse ``-exp(A_log) * softplus(g + dt_bias)`` from raw g logits.
        Requires per-head A_log and dt_bias; gate_domain then does not apply.
    A_log, dt_bias : torch.Tensor, optional
        Per-head ``[num_sab_heads]`` raw-gate parameters on q's device.
    beta_is_logit : bool
        Fuse sigmoid with rounding to beta's input dtype before FP32 use.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``output``, or ``(output, final_state)`` when ``output_final_state``.
    """
    _check_cudnn_frontend("chunk_gated_delta_rule")
    if cu_seqlens is None:
        raise ValueError("cudnn_chunk_gated_delta_rule: cu_seqlens is required")

    if (
        gate_domain != "linear"
        or use_gate_in_kernel
        or beta_is_logit
        or A_log is not None
        or dt_bias is not None
    ):
        from ..gdn_kernels.gates import validate_gate_inputs

        validate_gate_inputs(
            q,
            v,
            g,
            beta,
            gate_domain,
            use_gate_in_kernel,
            A_log,
            dt_bias,
            beta_is_logit,
        )

    total, num_q_heads, head_dim = q.shape
    v_dim = v.shape[2]
    num_sab_heads = max(num_q_heads, v.shape[1])
    num_seqs = cu_seqlens.shape[0] - 1
    overwrite = False
    if state_indices is not None:
        _check_cudnn_frontend("GDN state pools", (1, 31))
        overwrite = _validate_la_state_pool(
            state_indices,
            initial_state,
            output_state,
            output_final_state,
            num_seqs,
            num_sab_heads,
            head_dim,
            v_dim,
            q.device,
        )
    g_in = (
        torch.zeros(total, num_sab_heads, dtype=torch.float32, device=q.device)
        if g is None and gate_domain == "log"
        else _linear_gate(g, (total, num_sab_heads), q.dtype, q.device)
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
        g_in,
        beta,
        cu_seqlens,
        output,
        initial_state=initial_state,
        final_state=final_state,
        scale=float(scale) if scale else 1.0 / math.sqrt(head_dim),
        use_qk_l2norm=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid=bool(beta_is_logit),
        safe_gate=bool(use_gate_in_kernel),
        a_log=A_log,
        dt_bias=dt_bias,
        gate_lower_bound=None,
        batch_invariant=bool(batch_invariant),
        gate_domain="log" if use_gate_in_kernel else gate_domain,
        state_indices=state_indices,
        overwrite_initial_state=overwrite,
    )
    if overwrite:
        torch.autograd.graph.increment_version(initial_state)
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
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated DeltaProduct prefill on cuDNN's fused SM100 engine.

    GDP applies ``n = num_householder`` beta-gated Householder updates per
    token with one per-head scalar decay per token: the GDN recurrence on an
    expanded sub-token timeline, with the decay acting before the token's
    updates and the readout following the last one. ``num_householder == 1``
    is exactly :func:`cudnn_chunk_gated_delta_rule`.

    Requires cudnn-frontend 1.31.0+ with the ``cutedsl`` extra. Everything else
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
        ``[total_seq_len, num_sab_heads]`` at real-token rows, passed to cuDNN
        as is (``gate_domain="linear"``) at ``g``'s own dtype. All-ones when
        ``None``.
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
        float32, bfloat16 or float16. Without ``state_indices``, ``output_state`` must
        not alias ``initial_state``; see :func:`cudnn_chunk_gated_delta_rule`.
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

    state_indices : torch.Tensor, optional
        Contiguous int32 ``[num_seqs]`` pool slots, requiring cuDNN frontend
        1.31+. State pools use ``[N_pool, H, V, K]`` with dense inner rows
        and an aligned slot stride. Returning state requires an explicit
        output pool, either the identical input view or disjoint memory.
        Slots must be unique and in range; values are not read back.
        Unselected rows stay unchanged; no pool is written when
        ``output_final_state=False``. See :func:`cudnn_chunk_gated_delta_rule`.

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
    num_seqs = cu_seqlens.shape[0] - 1
    overwrite = False
    if state_indices is not None:
        _check_cudnn_frontend("GDP state pools", (1, 31))
        overwrite = _validate_la_state_pool(
            state_indices,
            initial_state,
            output_state,
            output_final_state,
            num_seqs,
            num_sab_heads,
            head_dim,
            v_dim,
            q.device,
        )
    n = int(num_householder)

    g_in = _linear_gate(g, (total, num_sab_heads), q.dtype, q.device)
    if beta is None:
        beta = torch.ones(
            total * n, num_sab_heads, dtype=torch.float32, device=q.device
        )

    if output is None:
        output = torch.empty(
            total, num_sab_heads, v_dim, dtype=q.dtype, device=q.device
        )
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
        g_in,
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
        gate_domain="linear",
        state_indices=state_indices,
        overwrite_initial_state=overwrite,
    )
    if overwrite:
        torch.autograd.graph.increment_version(initial_state)
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
    state_indices: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule 2 prefill on cuDNN's fused SM100 engine.

    Argument meanings match :func:`flashinfer.chunk_gated_delta_rule2`.

    GDN-2 generalizes GDN's per-head scalar gates to channel-wise ones: the
    forget gate ``g`` and erase gate ``beta`` are per key channel and the write
    gate ``w`` is per value channel.

    .. math::

        S_t &= \mathrm{diag}(e^{g_t}) S_{t-1} \\
        v^{new}_t &= w_t \odot v_t - (\beta_t \odot k_t)^\top S_t \\
        S_t &\mathrel{+}= k_t \otimes v^{new}_t \\
        o_t &= \mathrm{scale} \cdot q_t^\top S_t

    Requires cudnn-frontend 1.31.0+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[total_seq_len, num_q_heads / num_k_heads, K]`` and
        ``[total_seq_len, num_v_heads, V]`` with K and V each 64 or 128,
        packed.
    g : torch.Tensor, optional
        Channel-wise forget gate as the natural-log decay, shape
        ``[total_seq_len, num_sab_heads, K]``, passed to cuDNN as is
        (``gate_domain="log"``) at ``g``'s own dtype, which cuDNN reads at
        float32, bfloat16 or float16. All-zeros when ``None``.
    beta : torch.Tensor, optional
        Channel-wise erase gate ``[total_seq_len, num_sab_heads, K]``,
        converted to ``q.dtype``. All-ones when ``None``.
    w : torch.Tensor, optional
        Channel-wise write gate ``[total_seq_len, num_sab_heads, V]``,
        converted to ``q.dtype``. All-ones when ``None``.
    scale : float, optional
        Query scale; ``1 / sqrt(head_dim)`` when ``None`` or ``0.0``, matching
        the native GDN path.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, num_sab_heads, V, K]``, V-major, float32,
        bfloat16 or float16. Without ``state_indices``, ``output_state`` must not alias
        ``initial_state``; see
        :func:`cudnn_chunk_gated_delta_rule`.
    output_final_state : bool
        Return the outgoing recurrent state alongside the output; when unset
        no state is written at all.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32 or int64. Required.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel.
    output : torch.Tensor, optional
        Pre-allocated ``[total_seq_len, num_o_heads, V]``.
    batch_invariant : bool
        Disable the split-K partition; see
        :func:`cudnn_chunk_gated_delta_rule`.

    state_indices : torch.Tensor, optional
        Contiguous int32 ``[num_seqs]`` pool slots, requiring cuDNN frontend
        1.31+. State pools use ``[N_pool, H, V, K]`` with dense inner rows
        and an aligned slot stride. Returning state requires an explicit
        output pool, either the identical input view or disjoint memory.
        Slots must be unique and in range; values are not read back.
        Unselected rows stay unchanged; no pool is written when
        ``output_final_state=False``. See :func:`cudnn_chunk_gated_delta_rule`.

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
    num_seqs = cu_seqlens.shape[0] - 1
    overwrite = False
    if state_indices is not None:
        _check_cudnn_frontend("GDN2 state pools", (1, 31))
        overwrite = _validate_la_state_pool(
            state_indices,
            initial_state,
            output_state,
            output_final_state,
            num_seqs,
            num_sab_heads,
            head_dim,
            v_dim,
            q.device,
        )

    if g is None:
        g = torch.zeros(total, num_sab_heads, head_dim, dtype=q.dtype, device=q.device)
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
        g,
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
        gate_domain="log",
        state_indices=state_indices,
        overwrite_initial_state=overwrite,
    )
    if overwrite:
        torch.autograd.graph.increment_version(initial_state)
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
    state_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Kimi Delta Attention prefill on cuDNN's fused SM100 engine.

    Argument meanings match :func:`flashinfer.recurrent_kda`, restricted to the
    ordinary multi-token prefill subset: no speculative decode,
    no ``initial_state_source``, no state checkpoints.

    Requires cudnn-frontend 1.31.0+ with the ``cutedsl`` extra. Everything else
    the engine decides for itself.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[1, total_tokens, H, K]`` or ``[total_tokens, H, K]`` for q and k,
        ``[..., HV, V]`` for v, K and V each 64 or 128, bfloat16 or float16.
    g : torch.Tensor
        Channel-wise gate ``[..., total_tokens, HV, K]``. Log-space unless
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
        Required with ``use_gate_in_kernel``: cuDNN's in-kernel gate is
        ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` and it has no
        unbounded softplus form, so ``None`` raises ``NotImplementedError``
        instead of running the bounded gate. A softplus model passes its
        log-space gate as ``g`` with ``use_gate_in_kernel=False``.
    cu_seqlens : torch.Tensor
        ``[num_seqs + 1]`` int32 or int64. Required.
    initial_state, output_state : torch.Tensor, optional
        State ``[num_seqs, HV, V, K]``, V-major, float32, bfloat16 or float16.
        Following the Cake and CuTe DSL prefill backends, ``initial_state`` is
        advanced to the final state whenever one is given and no separate
        ``output_state`` is supplied, independently of
        ``output_final_state`` -- which gates only what is returned.
        Without ``state_indices``, ``output_state`` must not alias
        ``initial_state``; see :func:`cudnn_chunk_gated_delta_rule`.
    output : torch.Tensor, optional
        Pre-allocated output, written in place by the kernel.
    batch_invariant : bool
        Disable the split-K partition; see
        :func:`cudnn_chunk_gated_delta_rule`.
    state_indices : torch.Tensor, optional
        Contiguous int32 ``[num_seqs]`` slots in recurrent state pools,
        requiring cuDNN frontend 1.31+ and ordinary packed prefill.
        Pools have shape ``[N_pool, HV, V, K]`` with dense inner rows and
        a 16-byte aligned slot stride, in float32, bfloat16 or float16. Slot ids
        must be unique and in range; their values are not read back.
        Selected input rows advance in place unless a separate
        ``output_state`` pool is supplied, even when ``output_final_state``
        is false. An explicit output pool must have the same shape/dtype
        and either be the identical view or use disjoint memory. Untouched
        slots retain their values. Returning state returns the whole pool.

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
    if use_gate_in_kernel and lower_bound is None:
        raise NotImplementedError(
            "cudnn_recurrent_kda: cuDNN's in-kernel KDA gate is "
            "lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); the unbounded "
            "softplus gate (lower_bound=None) is not served. Pass a negative "
            "lower_bound, or a precomputed log-space g with use_gate_in_kernel=False."
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
    if state_indices is not None:
        _check_cudnn_frontend("KDA state pools", (1, 31))
        if num_seqs <= 0 or q.shape[0] <= num_seqs:
            raise NotImplementedError(
                "cuDNN KDA state_indices requires ordinary packed multi-token prefill"
            )
        final_state = initial_state if output_state is None else output_state
        # KDA advances its supplied state independently of whether it is returned.
        overwrite_initial_state = _validate_la_state_pool(
            state_indices,
            initial_state,
            final_state,
            True,
            num_seqs,
            num_heads,
            head_dim,
            v_dim,
            q.device,
        )
    else:
        # The compatibility scratch is compact; keep strided states on the
        # existing copy-back path so their declared and actual layouts match.
        overwrite_initial_state = (
            initial_state is not None
            and output_state is None
            and initial_state.is_contiguous()
            and not initial_state.requires_grad
            and (torch.is_inference_mode_enabled() or not initial_state.is_inference())
        )
        final_state = (
            initial_state
            if overwrite_initial_state
            else (
                _state_out(
                    initial_state,
                    output_state,
                    num_seqs,
                    num_heads,
                    head_dim,
                    v_dim,
                    q.device,
                )
                if (
                    output_final_state
                    or output_state is not None
                    or initial_state is not None
                )
                else None
            )
        )

    final_state = _run_la_graph(
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
        overwrite_initial_state=overwrite_initial_state,
        state_indices=state_indices,
    )

    if state_indices is not None:
        if overwrite_initial_state:
            torch.autograd.graph.increment_version(initial_state)
    elif output_state is None and initial_state is not None:
        if final_state is initial_state:
            # FE writes through a raw pointer. Preserve the mutation tracking
            # previously supplied by copy_ (a no-op for inference tensors).
            torch.autograd.graph.increment_version(initial_state)
        else:
            initial_state.copy_(final_state)
            final_state = initial_state
    return output.reshape(out_shape), final_state if output_final_state else None


# One ``[0, S, 2S, ...]`` boundary tensor per dense Mamba-2 call shape and device.
_ssd_dense_offsets: dict = {}


def _ssd_dense_cu_seqlens(
    batch: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    key = (get_device_index(device), batch, seqlen)
    offsets = _ssd_dense_offsets.get(key)
    if offsets is None:
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=device) * seqlen
        _ssd_dense_offsets[key] = offsets
    return offsets


@flashinfer_api
def cudnn_ssd_combined_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
    num_seqs: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    final_states: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Mamba-2 SSD combined forward on cuDNN's fused SM100 engine.

    Argument meanings match :meth:`flashinfer.mamba.SSDCombined.run`. Mamba-2
    is the gated delta rule without the erase term: one per-token scalar
    ``dt`` is both the decay exponent and the write strength,

    .. math::

        dt_t &= \mathrm{clamp}(\mathrm{softplus}(dt_t + dt\_bias), dt_{min}, dt_{max}) \\
        S_t &= e^{A\, dt_t} S_{t-1} + dt_t\, B_t \otimes x_t \\
        y_t &= C_t^\top S_t + D x_t \quad (\cdot\; z_t\, \sigma(z_t))

    with the softplus applied only under ``dt_softplus``. In cuDNN's
    delta-rule vocabulary ``C`` is ``q``, ``B`` is ``k``, ``x`` is ``v`` and the
    raw ``dt`` is the gate.

    Requires cudnn-frontend 1.32.0+ with the ``cutedsl`` extra. The engine
    serves ``headdim == 64`` and ``dstate == 128`` on the SM100 family and
    declines everything else it cannot run.

    Parameters
    ----------
    x : torch.Tensor
        ``[batch, seqlen, nheads, headdim]``, bfloat16 or float16. Packed
        varlen calls carry ``batch == 1``.
    dt : torch.Tensor
        ``[batch, seqlen, nheads]`` step sizes, raw (pre-softplus) when
        ``dt_softplus`` is set; float32, bfloat16 or float16.
    A : torch.Tensor
        ``[nheads]`` negative decay rates, float32.
    B, C : torch.Tensor
        ``[batch, seqlen, ngroups, dstate]`` at ``x``'s dtype; ``nheads`` must be
        a multiple of ``ngroups``.
    D : torch.Tensor, optional
        Skip connection ``[nheads]`` or ``[nheads, headdim]`` (``[nheads, 1]`` is
        read as per head).
    z : torch.Tensor, optional
        Output gate like ``x``; ``y *= z sigmoid(z)`` is fused.
    dt_bias : torch.Tensor, optional
        ``[nheads]``, added to ``dt`` before the softplus.
    dt_softplus : bool
        Apply the softplus to ``dt + dt_bias``.
    dt_limit : tuple[float, float]
        Inclusive clamp of the processed step size; ``inf`` is unbounded.
    initial_states : torch.Tensor, optional
        ``[num_seqs, nheads, headdim, dstate]`` (``[batch, ...]`` for a dense
        call), float32, bfloat16 or float16. A zero state when ``None``.
    seq_idx, chunk_indices, chunk_offsets : torch.Tensor, optional
        The CuTe backend's packed-varlen form. ``seq_idx`` ``[1, seqlen]``
        holds non-decreasing in-range sequence IDs; the boundaries are derived
        from it on the device (an out-of-range ID is clamped, which keeps the
        call memory-safe and leaves those tokens' outputs undefined).
        ``chunk_indices`` / ``chunk_offsets`` are accepted for call
        compatibility and not read: the engine derives its chunk tables from
        ``cu_seqlens``.
    out : torch.Tensor, optional
        Caller-owned contiguous token-major ``[batch, seqlen, nheads, headdim]``.
    return_final_states : bool
        Return the per-sequence final state; when unset no state is written.
    num_seqs : int, optional
        Sequence count for a ``seq_idx`` call without ``initial_states``.
    cu_seqlens : torch.Tensor, optional
        ``[num_seqs + 1]`` int32 or int64 token offsets, the packed-varlen form
        the engine takes directly.
    final_states : torch.Tensor, optional
        Caller-owned ``[num_seqs, nheads, headdim, dstate]`` final-state buffer
        at the state dtype.
    batch_invariant : bool
        Disable the split-K partition; see
        :func:`cudnn_chunk_gated_delta_rule`.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        ``(out, final_states)``; ``final_states`` is ``None`` unless
        ``return_final_states``.
    """
    _check_cudnn_frontend("ssd_combined_fwd", (1, 32))
    if x.dim() != 4 or B.dim() != 4 or C.dim() != 4 or dt.dim() != 3:
        raise ValueError(
            "cudnn_ssd_combined_fwd: x, B and C are [batch, seqlen, heads, dim] "
            "and dt is [batch, seqlen, heads]"
        )
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(
            f"cudnn_ssd_combined_fwd: x, B, C and z must be bfloat16 or float16, got {x.dtype}"
        )
    batch, seqlen, num_heads, head_dim = x.shape
    num_groups, state_dim = B.shape[2], B.shape[3]
    total = batch * seqlen
    if (cu_seqlens is not None or seq_idx is not None) and batch != 1:
        raise ValueError("cudnn_ssd_combined_fwd: packed-varlen calls carry batch == 1")
    if cu_seqlens is not None:
        if cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.dim() != 1:
            raise ValueError(
                "cudnn_ssd_combined_fwd: cu_seqlens must be a 1-D int32 or int64 tensor"
            )
        count = cu_seqlens.shape[0] - 1
        if num_seqs is not None and num_seqs != count:
            raise ValueError(
                f"cudnn_ssd_combined_fwd: num_seqs ({num_seqs}) disagrees with cu_seqlens ({count})"
            )
    elif seq_idx is not None:
        if tuple(seq_idx.shape) != (batch, seqlen):
            raise ValueError(
                f"cudnn_ssd_combined_fwd: seq_idx must be [batch, seqlen], got {tuple(seq_idx.shape)}"
            )
        count = (
            num_seqs
            if num_seqs is not None
            else initial_states.shape[0]
            if initial_states is not None
            else None
        )
        if count is None:
            raise ValueError(
                "cudnn_ssd_combined_fwd: a seq_idx call needs initial_states or "
                "num_seqs for the sequence count"
            )
        # Per-sequence token counts, then their prefix sum: the boundaries of
        # non-decreasing IDs, derived on the device without a host sync. An
        # out-of-range ID is clamped so the call stays memory-safe; those
        # tokens' outputs are then undefined, as on the Cake backend.
        counts = torch.zeros(count, dtype=torch.int32, device=x.device)
        counts.scatter_add_(
            0,
            seq_idx.reshape(-1).long().clamp_(0, count - 1),
            torch.ones(total, dtype=torch.int32, device=x.device),
        )
        cu_seqlens = torch.zeros(count + 1, dtype=torch.int32, device=x.device)
        cu_seqlens[1:] = counts.cumsum(0)
    else:
        count = batch
        cu_seqlens = _ssd_dense_cu_seqlens(batch, seqlen, x.device)
    del chunk_indices, chunk_offsets
    if initial_states is not None and initial_states.shape[0] != count:
        raise ValueError(
            f"cudnn_ssd_combined_fwd: initial_states holds {initial_states.shape[0]} "
            f"rows for {count} sequences"
        )
    state_dtype = (
        initial_states.dtype
        if initial_states is not None
        else final_states.dtype
        if final_states is not None
        else torch.bfloat16
    )
    if state_dtype not in (torch.bfloat16, torch.float32, torch.float16):
        raise NotImplementedError(
            f"cudnn_ssd_combined_fwd: the state is carried at bfloat16, float32 or float16, got {state_dtype}"
        )
    if D is not None and D.dim() == 2 and D.shape[1] == 1:
        D = D[:, 0]

    x_thd = x.reshape(total, num_heads, head_dim)
    b_thd = B.reshape(total, num_groups, state_dim)
    c_thd = C.reshape(total, num_groups, state_dim)
    dt_thd = dt.reshape(total, num_heads)
    z_thd = None if z is None else z.reshape(total, num_heads, head_dim)
    if out is None:
        out = torch.empty(
            batch, seqlen, num_heads, head_dim, dtype=x.dtype, device=x.device
        )
    elif tuple(out.shape) != tuple(x.shape) or out.dtype != x.dtype:
        raise ValueError(
            "cudnn_ssd_combined_fwd: out must match x's shape and dtype, got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    elif not out.is_contiguous():
        raise ValueError("cudnn_ssd_combined_fwd: out must be contiguous")
    o_thd = out.view(total, num_heads, head_dim)

    final = None
    if return_final_states:
        shape = (count, num_heads, head_dim, state_dim)
        if final_states is None:
            final_states = torch.empty(shape, dtype=state_dtype, device=x.device)
        elif tuple(final_states.shape) != shape or final_states.dtype != state_dtype:
            raise ValueError(
                f"cudnn_ssd_combined_fwd: final_states must be {shape} at {state_dtype}, "
                f"got {tuple(final_states.shape)} {final_states.dtype}"
            )
        final = final_states
    dt_min = float(dt_limit[0])
    dt_max = None if math.isinf(float(dt_limit[1])) else float(dt_limit[1])

    final = _run_la_graph(
        "mamba2",
        c_thd,
        b_thd,
        x_thd,
        dt_thd,
        None,
        cu_seqlens,
        o_thd,
        a=A,
        dt_bias=dt_bias,
        d=D,
        z=z_thd,
        initial_state=initial_states,
        final_state=final,
        scale=1.0,
        use_qk_l2norm=False,
        use_beta_sigmoid=False,
        safe_gate=False,
        gate_lower_bound=None,
        batch_invariant=bool(batch_invariant),
        dt_softplus=bool(dt_softplus),
        dt_min=dt_min,
        dt_max=dt_max,
    )
    return out, final if return_final_states else None
