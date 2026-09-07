"""Static backend selection (proposal §5.3, level 1).

Tensor-free: callable at engine init, before the KV pool exists and before
any CUDA graph is captured. Evaluates every declared capability so the
returned ``Resolution`` carries a reason for each excluded backend — the
"explain" answer consumer-side tables cannot give.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from ._backends._capabilities import CAPABILITIES
from ._contracts import (
    Resolution,
    _expect_page_size,
    _expect_window_left,
    resolve_config_key,
)


def _probe_fa(backend: str) -> Optional[str]:
    if backend == "fa3":
        from ...utils import is_sm90a_supported

        if not is_sm90a_supported(torch.device("cuda")):
            return "fa3 requires SM90a (Hopper) and CUDA >= 12.3"
    return None


def _probe_cudnn() -> Optional[str]:
    from ...cudnn import prefill as cudnn_prefill

    if not cudnn_prefill.CUDNN_AVAILABLE:
        return "cudnn-frontend python package not importable"
    return None


def _probe_trtllm() -> Optional[str]:
    # Cubin availability is a real capability question (proposal: it should be
    # a library answer, not an engine-side HTTP probe).  The prototype defers
    # to first-run download; a production probe would consult the local cubin
    # cache / FLASHINFER_NO_DOWNLOAD.
    return None


# Environment probes: things the static capability table cannot know
# (installed packages, toolkit level). Run only for capability-admitted
# backends so explain() stays cheap.
PROBES: Dict[str, Callable[[], Optional[str]]] = {
    "fa2": lambda: _probe_fa("fa2"),
    "fa3": lambda: _probe_fa("fa3"),
    "cudnn": _probe_cudnn,
    "trtllm-gen": _probe_trtllm,
}

# Static heuristic placeholder (proposal §5.2: to be seeded from the benchmark
# suite; it only has to beat consumer tables that rot).  Order = preference.
HEURISTIC_ORDER: Dict[int, Tuple[str, ...]] = {
    10: ("trtllm-gen", "cudnn", "fa2"),
    9: ("fa3", "fa2", "cudnn"),
    8: ("fa2", "cudnn"),
    12: ("fa2", "cudnn"),
}


def resolve_paged_attention(
    *,
    device: Optional[torch.device] = None,
    cc_major: Optional[int] = None,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: Optional[int] = None,
    q_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype] = None,
    page_size: int,
    kv_layout: str = "HND",
    causal: bool = True,
    need_lse: bool = False,
    window_left: int = -1,
    kv_input_form: str = "block_tables",
    backend: str = "auto",
) -> Resolution:
    """Static backend resolution — no plan state, no tensors.

    Raises ``ValueError`` when nothing can run, with per-backend reasons.
    """
    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    if kv_dtype is None:
        kv_dtype = q_dtype
    if cc_major is None:
        dev = device if device is not None else torch.device("cuda")
        cc_major = torch.cuda.get_device_properties(dev).major
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) and num_kv_heads ({num_kv_heads}) "
            "must be positive integers"
        )
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for GQA/MQA"
        )
    if kv_input_form not in ("block_tables", "page_indices"):
        raise ValueError(
            f"kv_input_form must be 'block_tables' or 'page_indices', got "
            f"{kv_input_form!r}"
        )
    _expect_window_left(window_left)
    _expect_page_size(page_size, kv_input_form)

    order = HEURISTIC_ORDER.get(cc_major, ())
    if backend != "auto":
        if backend not in CAPABILITIES:
            raise ValueError(
                f"unknown backend {backend!r}; known: {sorted(CAPABILITIES)} or 'auto'"
            )
        evaluate: Tuple[str, ...] = (backend,)
    else:
        # Evaluate EVERY known backend so explain() is complete: candidates
        # are ordered by the heuristic; everything else carries its reason.
        evaluate = tuple(order) + tuple(n for n in CAPABILITIES if n not in order)

    candidates, excluded = [], {}
    for name in evaluate:
        cap = CAPABILITIES[name]
        reason = cap.rejection_reason(
            cc_major=cc_major,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=kv_input_form,
        )
        if reason is None:
            reason = PROBES[name]()
        if reason is None:
            candidates.append(name)
        else:
            excluded[name] = reason

    if not candidates:
        detail = "; ".join(f"{k}: {v}" for k, v in excluded.items())
        raise ValueError(f"no runnable backend for this configuration ({detail})")
    return Resolution(
        backends=tuple(candidates),
        excluded=excluded,
        kv_layout=kv_layout,
        config=resolve_config_key(
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo,
            q_dtype,
            kv_dtype,
            page_size,
            kv_layout,
            causal,
            need_lse,
            window_left,
            kv_input_form,
        ),
    )


__all__ = ["HEURISTIC_ORDER", "PROBES", "resolve_paged_attention"]
