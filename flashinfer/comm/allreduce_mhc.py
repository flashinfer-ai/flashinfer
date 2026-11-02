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

from typing import Any, Optional, Tuple

import torch

from ..api_logging import flashinfer_experimental_api

# Thin experimental entry points. The supported surface, the support query, the JIT module and
# its cache, the fused launch and the composable path live in
# flashinfer.experimental.allreduce_mhc_post_prenorm.

_FEATURE = "MNNVL all-reduce fused with the mHC post-map and pre-norm partials"


@flashinfer_experimental_api(feature=_FEATURE)
def allreduce_mhc_post_prenorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    workspace: Any,
    launch_with_pdl: bool = True,
    yp: Optional[torch.Tensor] = None,
    rp: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""All-reduce a sublayer output over an MNNVL workspace and apply the mHC post-map and the
    split-K pre-norm partials of the next mHC pre block.

    At an mHC (manifold-constrained hyper-connection) sublayer boundary the tensor-parallel
    partial ``input`` is all-reduced into ``x`` and consumed by::

        new_r[m, j]  = post_layer_mix[m, j] * x[m] + sum_k comb_res_mix[m, k, j] * residual[m, k]
        residual_out = new_r rounded to bf16
        yp[s, m, n]  = sum_{j, h in split s} fn[n, j * H + h] * new_r[m, j, h]
        rp[s, m]     = sum_{j, h in split s} new_r[m, j, h] ** 2

    with ``new_r`` in fp32 and ``yp`` / ``rp`` from the unrounded ``new_r`` over 8 contiguous
    hidden splits of ``H / 8``.

    On the supported surface (8 ranks, ``H = 4096``, 4 residual streams, bf16, 1 to 32 tokens,
    compute capability 10.x, :class:`~flashinfer.comm.MNNVLAllReduceFusionWorkspace` with
    one-shot capacity for the call) one kernel performs the one-shot Lamport all-reduce and the
    epilogue. The all-reduce is bit-identical to the stock one-shot kernel (fp32 sum in rank
    order, one bf16 rounding) and shares its workspace and buffers, so calls interleave with
    ``trtllm_mnnvl_allreduce`` freely. Elsewhere the stock MNNVL all-reduce and a PyTorch epilogue
    produce the same outputs. :func:`allreduce_mhc_post_prenorm_supported` tells which path a
    call takes. This API takes no part in automatic backend selection.

    Parameters
    ----------
    input : torch.Tensor
        This rank's un-reduced sublayer output, bf16 ``[num_tokens, H]``.
    residual : torch.Tensor
        mHC residual streams, bf16 ``[num_tokens, hc_mult, H]``.
    post_layer_mix : torch.Tensor
        fp32 ``[num_tokens, hc_mult]`` (or ``[num_tokens, hc_mult, 1]``).
    comb_res_mix : torch.Tensor
        fp32 ``[num_tokens, hc_mult, hc_mult]``; ``comb_res_mix[m, k, j]`` weighs stream ``k`` in
        new stream ``j``.
    fn : torch.Tensor
        fp32 pre-norm projection ``[mix, hc_mult * H]`` with ``mix = hc_mult * (2 + hc_mult)``.
    workspace : MNNVLAllReduceFusionWorkspace
        Workspace of the tensor-parallel group, shared with the stock MNNVL all-reduce.
    launch_with_pdl : bool
        Launch with programmatic dependent launch.
    yp, rp, residual_out : Optional[torch.Tensor]
        Preallocated outputs (``[8, num_tokens, mix]`` fp32, ``[8, num_tokens]`` fp32 and the
        shape and dtype of ``residual``).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(yp, rp, residual_out)``.

    Notes
    -----
    Every rank of the group must make the same call. Each rank chooses its path on its own (a rank
    that cannot load the fused module takes the composable path); inside the fused kernel's
    surface the composable path all-reduces with the stock one-shot kernel, the protocol of the
    fused kernel, so the call completes on every rank whatever each rank chose. Before capturing
    a CUDA graph that contains this call, call :func:`allreduce_mhc_post_prenorm_prepare` outside
    capture: a call made during capture never builds the module, and without it the composable
    path is captured instead.

    Any strides are accepted on both paths. The fused kernel reads and writes dense row-major
    tensors, so a strided input (e.g. ``post_layer_mix`` / ``comb_res_mix`` sliced from one
    packed mix tensor) is copied first and a strided preallocated output receives a copy of the
    result; contiguous tensors are used as they are. ``residual_out`` may be ``residual`` (an
    in-place update). Tensors outside the dtypes above take the composable path.

    With ``launch_with_pdl`` the fused kernel loads ``fn`` before it waits on the preceding
    kernel, as it would a weight: ``fn`` must not be written by the kernel launched right before
    this call. A non-contiguous ``fn`` is copied, and that call launches without programmatic
    dependent launch.
    """
    from ..experimental.allreduce_mhc_post_prenorm.backend import run

    return run(
        input,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        workspace,
        launch_with_pdl=launch_with_pdl,
        yp=yp,
        rp=rp,
        residual_out=residual_out,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def allreduce_mhc_post_prenorm_supported(
    workspace: Any,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> bool:
    r"""Whether :func:`allreduce_mhc_post_prenorm` runs the fused kernel for this problem.

    Checks the supported surface (8 ranks, ``hidden_size == 4096``, ``hc_mult == 4``, bf16, 1 to
    32 tokens, compute capability 10.x), the workspace type and its one-shot capacity, and that
    the fused module is loaded; outside CUDA-graph capture it builds the module on first use.
    The answer holds for the documented tensor dtypes and for any strides. Callers that keep
    their own unfused path take it whenever this returns False.

    Parameters
    ----------
    workspace : MNNVLAllReduceFusionWorkspace
        Workspace of the tensor-parallel group.
    num_tokens : int
        Tokens of the call.
    hidden_size : int
        Hidden size ``H``.
    hc_mult : int
        Number of mHC residual streams.
    dtype : torch.dtype
        Dtype of ``input`` and ``residual``.
    device : Optional[torch.device]
        Device of the call (default: the current CUDA device).

    Returns
    -------
    bool
        The same answer on every rank for the same arguments and process state.
    """
    from ..experimental.allreduce_mhc_post_prenorm.backend import is_supported

    return is_supported(workspace, num_tokens, hidden_size, hc_mult, dtype, device)


@flashinfer_experimental_api(feature=_FEATURE)
def allreduce_mhc_post_prenorm_prepare() -> bool:
    r"""Build and load the fused module outside CUDA-graph capture.

    Call once per process before capturing graphs that contain
    :func:`allreduce_mhc_post_prenorm`. A call made during capture never builds.

    Returns
    -------
    bool
        True when the fused module is loaded.
    """
    from ..experimental.allreduce_mhc_post_prenorm.backend import prepare

    return prepare()
