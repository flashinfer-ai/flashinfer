"""Experimental GC MoE (Green Context MoE) operator family.

One dispatch -> per-expert compute -> combine call over a symmetric workspace,
covering SM90, SM100/SM103 and SM120 GPUs. The implementation, the vendored
upstream sources and the architecture routing live under
``flashinfer.experimental.gc_moe``; this module only carries the public
signatures and hands off.
"""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api(feature="gc_moe")
def gc_moe(
    y,
    l1_weights,
    l2_weights,
    workspace,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32),
    activation="swiglu",
    activation_clamp=None,
    fast_math=True,
    shared_recipe=None,
    activation_alpha=None,
    num_tokens_bound=None,
    backend=None,
):
    """Run one quantized GC MoE forward pass into ``y``.

    ``y`` is BF16 ``[T, H]`` (contiguous, CUDA) and is written in place; stage
    the token payload, scales, routing indices and routing weights into
    ``workspace`` first (``workspace.x``, ``workspace.x_sf``,
    ``workspace.topk_idx``, ``workspace.topk_weights``). Routed weights are the
    packed pairs returned by :func:`transform_gc_moe_weights`; per-expert
    statistics accumulate into ``cumulative_local_expert_recv_stats``.

    The call launches on the current CUDA stream and supports capture by an
    outer ``torch.cuda.graph``: while capturing, the internal role graph is
    nested into the enclosing graph, so replay does not re-run the host path.
    Warm the exact call once eagerly before capturing. Calls sharing one
    workspace must be serialized by the caller, and every captured graph must
    be retired before the workspace is destroyed.

    ``backend`` (``"sm90"``, ``"sm100"``, ``"sm120"``) overrides the
    architecture selected from the device compute capability.
    ``activation_alpha`` is rejected: it only parameterizes the ``swiglu_oai``
    activation, which this operator does not implement. ``num_tokens_bound``
    (a capture hint) is accepted by the SM90 and SM120 paths only.
    """
    from .experimental.gc_moe.dispatch import run_gc_moe

    return run_gc_moe(
        y,
        l1_weights,
        l2_weights,
        workspace,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        shared_recipe=shared_recipe,
        activation_alpha=activation_alpha,
        num_tokens_bound=num_tokens_bound,
        backend=backend,
    )


@flashinfer_experimental_api(feature="gc_moe workspace")
def GCMoEWorkspace(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    num_shared_experts=0,
    *,
    base=None,
    backend=None,
    ptxas=None,
):
    """Allocate the symmetric GC MoE workspace owned by this process.

    ``group`` is the expert-parallel process group; every rank keeps one local
    expert shard (``num_experts % group.size() == 0``). ``num_max_tokens_per_rank``
    bounds the tokens staged per rank per call. The returned object exposes the
    upstream staging views (``x``, ``x_sf``, ``topk_idx``, ``topk_weights``,
    plus the shared-expert views), holds the routing workspace for the whole
    lifetime of any graph that captured it, and must be destroyed with
    ``destroy()`` or by leaving a ``with`` block.

    ``base`` reuses the allocation of another workspace with an identical
    layout (SM100 only). ``backend`` overrides the architecture selected from
    the device compute capability. ``ptxas`` overrides the PTX assembler used
    by the JIT compiler; it defaults to ``$GC_MOE_PTXAS``, then the CUDA
    toolkit found on the machine.
    """
    from .experimental.gc_moe.workspace import create_workspace

    return create_workspace(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        base=base,
        backend=backend,
        ptxas=ptxas,
    )


@flashinfer_experimental_api(feature="gc_moe weight transform")
def transform_gc_moe_weights(
    l1_weights,
    l2_weights,
    *,
    l1_scale_2=None,
    l2_scale_2=None,
    shared_recipe=None,
    activation="swiglu",
    backend=None,
):
    """Convert checkpoint weights to the packed pairs ``gc_moe`` expects.

    Inputs are the checkpoint-format quantized ``(payload, scales)`` pairs for
    the FC1 (``l1``) and FC2 (``l2``) weights; the accepted payload dtype and
    scale layout follow the selected architecture:

    - SM100: int8 MXFP4 payloads with packed int32 UE8M0 scales in the
      MN-major layout;
    - SM90: int8 MXFP4 payloads with FP32 (or UE8M0 byte) scales, plus optional
      per-expert secondary scales for checkpoints that carry one;
    - SM120: int8 MXFP4 payloads with ``[E, N, K/128]`` packed int32 UE8M0
      scales; FP8 shared weights take a K32 or K128 ``shared_recipe``.

    Run it outside graph capture. The returned tensors are allocated on the
    current stream and must stay alive and address-stable until every launch
    and captured graph that references them has retired.
    """
    from .experimental.gc_moe.weights import transform_weights

    return transform_weights(
        l1_weights,
        l2_weights,
        l1_scale_2=l1_scale_2,
        l2_scale_2=l2_scale_2,
        shared_recipe=shared_recipe,
        activation=activation,
        backend=backend,
    )
