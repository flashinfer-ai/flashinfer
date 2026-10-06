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

from typing import Optional

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the backend contract, the
# chunk plans, the workspace, launch binding, autograd wrappers and JIT
# registration live in flashinfer.experimental.cake_lm_head_loss.

_FEATURE = "chunked LM-head + loss training kernels (SM100/SM103)"


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div: Optional[float] = None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
    return_logp: bool = False,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
):
    r"""Differentiable chunked LM-head projection + loss with a memory-bounded
    backward, SM100 / SM103 only.

    Computes the projection ``z = X @ W^T`` over token chunks of ``chunk_size``
    rows, the per-row log-probability of the selected token
    ``logp_t = z[t, y_t] - logsumexp_v z[t, v]`` and one of two objectives,
    without ever materializing a vocabulary-sized tensor of more than
    ``chunk_size`` rows.  The forward already produces the FP32 gradient
    accumulators of the trainable inputs (three GEMMs per chunk, no logits
    recomputation); the backward applies the incoming scalar gradient and
    performs the single cast to the output dtypes.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, H]`` hidden states, row-major with any leading stride
        (``H`` a multiple of 256; a stride whose byte pitch is not a multiple
        of 16 is copied to a contiguous tensor first).
    W : torch.Tensor
        BF16 ``[V, H]`` output weight, contiguous (``V`` a multiple of 256).
    labels : torch.Tensor
        int64 ``[T]``; ``-100`` marks an ignored row (zero loss, zero gradient,
        ``logp = 0``).
    objective : str
        ``"ce"``: ``loss = -sum(logp[valid]) / loss_div``.
        ``"policy"``: ``loss = -sum(loss_weights * min(exp(logp - infer_logp), 2))``
        over valid rows; the per-row logit-gradient scale is
        ``-loss_weights_t * ratio_t`` when ``ratio_t <= 2`` and ``0`` above the
        clipping boundary.
    loss_div : Optional[float]
        Positive scalar divisor of the cross-entropy objective (caller-supplied,
        never replaced by a local token mean).  Required for ``"ce"``.
    infer_logp : Optional[torch.Tensor]
        FP32 ``[T]`` inference log-probabilities of the policy objective.
    loss_weights : Optional[torch.Tensor]
        FP32 ``[T]`` signed weights of the policy objective; they already
        include masking and normalization (ignored rows carry weight 0).
    chunk_size : int
        Token chunk ``C`` (default 4096).  Every logits / dlogits / probability
        buffer spans at most ``C`` tokens; a batch smaller than ``C`` is one
        chunk, and a tail ``T % C`` is neither dropped nor padded.  Changing
        ``C`` may change reduction rounding, never the loss definition.
    return_logp : bool
        Also return the detached FP32 ``logp [T]`` (a diagnostic output).
    grad_weight_dtype : torch.dtype
        dtype of the weight gradient: ``torch.bfloat16`` (default) or
        ``torch.float32``.  ``dW`` is accumulated in FP32 across chunks and
        cast once at the output boundary.  Through this autograd entry the
        value must equal ``W.dtype`` (the autograd engine casts every gradient
        to its leaf's dtype); an FP32 ``dW`` for a BF16 ``W`` is available from
        the explicit ``cake_backend.forward_loss`` / ``backward_loss`` pair.
    deterministic : bool
        Only ``True`` is available: fixed sequential chunk order, no atomics,
        bitwise reproducible ``loss``, ``logp``, ``dX`` and ``dW``.
    backend : str
        Only ``"cake"``.
    compact_rows : bool, optional
        Chunk over the valid rows only (rows whose label is not ``-100``):
        the valid-row index is formed once per call, each chunk's rows of
        ``X`` are gathered into one reusable ``[chunk_size, H]`` buffer and
        ``logp`` / ``dX`` are scattered back with exact zeros on the ignored
        rows -- the same computation over fewer rows (per-row ``logp`` is
        bitwise that of the uncompacted path; the ``loss`` / ``dW`` reductions
        run over different chunk boundaries and differ by FP32 rounding).
        ``None`` (default) reads ``FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS``
        (on unless set to ``0``).  Compaction costs two device
        synchronizations per call.
    fuse_dw_cast : bool, optional
        Run the last chunk's weight-gradient GEMM in the backward, where the
        upstream scalar gradient is known, with the scale and the output cast
        fused into its epilogue: the forward accumulates the chunks before it
        in FP32 (no accumulator at all for a one-chunk call), the backward
        writes ``dW = cast(g * (dW_acc + dz_c^T @ X_c))`` straight from the
        GEMM and the separate pass over the ``[V, H]`` accumulator disappears.
        The same FP32 operations in the same order: ``dW`` is bitwise the
        unfused result.  The last chunk's BF16 ``dlogits`` rows and a view of
        its rows of ``X`` are saved for the backward (an in-place write to
        ``X`` between the forward and the backward raises PyTorch's
        saved-tensor version error).  ``None`` (default) reads
        ``FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST`` (on unless set to ``0``).

    The fused dX finalize (``FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE``, on
    unless set to ``0``) adds the K-slice slabs of a sliced ``dX`` GEMM with one
    fixed-order kernel and, with ``compact_rows``, writes the ``[T, H]`` ``dX`` in
    one pass (``bf16(g * dX_acc)`` on the valid rows, exact zeros elsewhere)
    instead of casting, zero-filling and scattering the compact rows; the same
    FP32 operations in the same order, so ``dX`` is bitwise the same.

    The dW side stream (``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM``: ``auto``
    (default) for calls of three or more chunks, ``1`` for every multi-chunk
    call, ``0`` never) launches each chunk's weight-gradient accumulate GEMM on
    a per-device side stream, forked after the chunk's row gradients and joined
    before the chunk buffer is reused and before the call returns, so it overlaps
    the tail of the chunk's ``dX`` GEMM; the same kernels in the same order per
    kernel, so every output is bitwise the same.

    The hidden valid-row count (``FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT``,
    on unless set to ``0``) forms a compacted call's valid-row index on the
    device and queues chunk 0's row gather and logits GEMM (its device-count
    form, bounded by the count read from device memory) before the count
    reaches the host through a pinned cell and a CUDA event, instead of
    counting and indexing on the host before the first launch; the same
    kernels per row, so every output is bitwise the same.

    Returns
    -------
    loss : torch.Tensor
        FP32 scalar.
    logp : torch.Tensor
        Detached FP32 ``[T]``, only with ``return_logp=True``.

    Precision boundary: BF16 GEMM output promoted to FP32 for the max /
    log-sum-exp / loss arithmetic, BF16 ``dlogits``, FP32 ``dW`` accumulation
    with one cast at the output, BF16 ``dX`` (FP32 accumulation, one cast).
    Gradients are produced only for inputs that require them (a frozen ``X``
    or ``W`` skips its GEMM).  The saved accumulators are re-scaled, never
    mutated, so a retained graph may run the backward repeatedly.  ``T == 0``
    gives loss 0, an empty ``logp`` and zero gradients.

    Host side: inputs are validated and bound on the first call for an input
    binding (``data_ptr``, shape, stride and dtype of every input plus the
    options); later calls with the same binding launch from the remembered
    argument plans
    (``flashinfer.experimental.cake_lm_head_loss.cake_backend.BINDING_CACHE``;
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE=0`` disables it).  Peak
    temporary memory is reported separately from the weights, the outputs and
    the FP32 accumulators by ``cake_backend.memory_report``.
    """
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    from .experimental.cake_lm_head_loss import cake_backend

    return cake_backend.chunked_lm_head_loss(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        return_logp=return_logp,
        grad_weight_dtype=grad_weight_dtype,
        deterministic=deterministic,
        backend="cake",
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunked_lm_head_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 4096,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
) -> torch.Tensor:
    r"""Differentiable chunked selected-token log-probability, SM100 / SM103 only.

    Returns the FP32 ``logp [T]`` (``logp_t = z[t, y_t] - logsumexp_v z[t, v]``
    for valid rows, 0 for rows whose label is ``-100``) of the projection
    ``z = X @ W^T`` for arbitrary downstream losses; the per-row gradient
    ``dlogp`` arrives from autograd.  The forward saves only FP32 row
    statistics (the log-sum-exp and the selected logit); the backward
    recomputes each chunk's logits (four GEMMs per chunk) and produces BF16
    ``dX`` and ``dW`` through FP32 accumulators with one cast each (with
    ``fuse_dw_cast`` the last chunk's weight-gradient GEMM writes the BF16
    ``dW`` from its epilogue).  Arguments as in :func:`chunked_lm_head_loss`;
    ``dW`` is returned in BF16.
    """
    if backend != "cake":
        raise ValueError(
            "the chunked LM-head + loss kernels currently support backend='cake'"
        )
    from .experimental.cake_lm_head_loss import cake_backend

    return cake_backend.chunked_lm_head_logprob(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=deterministic,
        backend="cake",
        compact_rows=compact_rows,
        fuse_dw_cast=fuse_dw_cast,
    )
