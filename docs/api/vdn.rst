.. _apivdn:

flashinfer.vdn
==============

Exact BF16 window softmax for Video DeltaNet (VDN) on SM120.
This operator computes only VDN's softmax branch. The linear delta-rule
recurrence, learned gates, checkpoint loading, and video generation belong
to the caller.

Each video query attends to the union of its frame window, global tokens
outside the video interval, and optional first/last anchor frames. Global
queries attend to the full sequence. Anchor ``rows`` make first/last-frame
queries dense; anchor ``columns`` expose those frames to every query;
``both`` combines the two. Overlapping keys are counted once in one joint
softmax. Window bounds are inclusive and clipped to the video interval.

The implementation groups queries with identical KV sets, schedules longer
KV groups first, and uses FA2 paged prefill with one token per page. A shared
KV pool avoids gathering global and anchor K/V repeatedly. Triton fuses Q
permutation with a strided V copy, then restores output order after attention.
There is no FP8 conversion or skip-softmax approximation.

Usage
-----

.. code-block:: python

    import torch
    from flashinfer import VDNWindowAttentionWrapper

    frames, tokens_per_frame, prefix, heads = 37, 510, 3623, 7
    seq_len = prefix + frames * tokens_per_frame
    # Each five-frame chunk sees the preceding, current and following chunk.
    bounds = [((f // 5 - 1) * 5, (f // 5 + 2) * 5 - 1) for f in range(frames)]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = VDNWindowAttentionWrapper(workspace)
    wrapper.plan(seq_len, heads, prefix, frames, tokens_per_frame, bounds,
                 anchor_frames="both")
    q, k, v = [torch.randn(seq_len, heads, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    out = wrapper.run(q, k, v)

Reuse the wrapper across layers with the same geometry. The caller owns
the workspace and wrapper lifetime; no global plan cache retains them.
Use a separate wrapper and workspace for each concurrent stream. This
initial implementation supports eager inference on SM120 with BF16 Q/K/V
and output, equal Q/KV head counts, and head dimension 128. Autograd and
CUDA graph capture are unsupported.
The softmax scale must remain positive and finite when represented in FP32;
the default is ``1 / sqrt(128)``.

Validation
----------

Run ``pytest tests/attention/test_vdn_window.py --full`` for independent FP32
masked attention checks, tile boundaries, strided projections, anchor
deduplication, plan reuse, stream/device handling and input validation.
The full suite includes sampled FP32 references at 22,493 and 58,193 tokens
and a copy-kernel regression beyond the signed int32 element-offset boundary.
The latter requires 18 GiB of free device memory; multi-device checks require
two visible CUDA devices.

.. currentmodule:: flashinfer.vdn

.. autoclass:: VDNWindowAttentionWrapper
    :members: plan, run

    .. automethod:: __init__
