.. _apigdn_decode:

flashinfer.gdn_decode
=====================

Gated Delta-Rule decode-side kernels used in Mamba-2 / GDN-style sequence
models. These functions consume a pre-built KV / state cache and run the
recurrent gated delta-rule update for the current decode step.

.. currentmodule:: flashinfer.gdn_decode

.. autosummary::
    :toctree: ../generated

    gated_delta_rule_decode
    gated_delta_rule_decode_pretranspose
    gated_delta_rule_mtp
    gated_delta_rule_replayssm_commit

Rubin T=1 bf16-state decode (SM107)
-----------------------------------

On Rubin (compute capability 10.7), ``gated_delta_rule_decode_pretranspose``
routes T=1 calls with a bfloat16 state, K=V=128 and ``B * HV >= 512`` to a
Rubin-tuned kernel under ``backend="auto"``. It requests each 64-row state tile
at once into shared memory, prefetches a later tile into L2 and issues fewer
instructions; outputs and the updated state are bitwise equal to the generic
kernel. ``backend="rubin"`` forces it (any batch size, Rubin only) and
``backend="flashinfer"`` keeps the generic kernel.

uGPU localization: Rubin is built from two dies, each with its own SMs and
memory. Allocate the state pool with :func:`ugpu_state_pool` (layout
``[pool_size, HV, 2, V // 2, K]``, half ``t`` of every state in die ``t``'s
memory) and a T=1 call on Rubin runs half ``t`` on die ``t``'s SMs, so no
state byte crosses the die-to-die link, whatever the batch order or slot
indices. Results are bitwise equal to the one-launch kernel. This targets
CUDA-graph captured decode (make one eager call before capture, which creates
the per-die green contexts); eager calls with ``B * HV < 8192`` are host-bound
and run the two halves back to back on the current stream.

.. autosummary::
    :toctree: ../generated

    ugpu_state_pool

Locality-domain FP32 decode state (SM107a)
------------------------------------------

The K-major FP32 decode state of ``gated_delta_rule_decode`` can be homed on
the two dies of a Rubin GPU: :func:`localize_gdn_decode_state` copies a
``[B, HV, K, V]`` state into one allocation whose batch rows ``[0, rows0)`` are
in die 0's memory and the rest in die 1's (by default half the rows on each
die). Pass the returned tensor as ``state`` from then on. A T=1 call on it with
``backend="auto"`` or ``backend="cake_gdn"`` runs the locality-domain twin of
the Cake nontranspose kernel: one launch over both dies whose CTAs take the
batch rows of their own die first. The output and the updated state are bitwise
equal to the whole-GPU kernel's.

The twin covers contiguous T=1 calls with BF16 q/k/v/a/b and output, FP32
``A_log`` and ``dt_bias`` and ``use_qk_l2norm=True``, at H=16 / HV=32 from B=48
on (scale ``K ** -0.5`` or 1.0), on 212-SM devices, and needs a CUDA 13.5
toolkit and driver. Two fallbacks apply:

- A call the Cake kernels serve but the twin does not (another row, a view of
  part of the state, a copy of it), or one whose twin fails to build, load or
  launch, runs the whole-GPU Cake kernel, with the same bits.
- A call the Cake kernels do not serve (for example FP16 inputs, a BF16
  ``dt_bias``, non-contiguous inputs or ``use_qk_l2norm=False``) runs the
  FlashInfer kernel under ``backend="auto"`` and raises
  ``CakeGDNUnsupportedError`` under ``backend="cake_gdn"``.

A T=1 call with the contiguous BF16 and FP32 tensors listed above and
K = V = 128 raises ``ValueError``, under ``backend="auto"`` as under
``backend="cake_gdn"`` and for every row and GPU, if q or k does not start at
an 8-byte aligned address or the state does not start at a 16-byte aligned
one. ``backend="auto"`` does not fall back to the FlashInfer kernel here: that
kernel assumes 16-byte aligned q, k, v, output and state, also on the calls
that ``backend="auto"`` hands to it. On a call the Cake kernels run, the other
tensors need only the alignment of their element type.

The helper returns ``state`` itself, without a copy, under CUDA graph capture
and wherever it does not home the state (for example a row the twin does not
cover); otherwise it returns a new homed copy. So
``localize_gdn_decode_state(state, H, scale) is state`` means only that no copy
was made, not which kernel a call runs: a call on a state that was never homed
never runs the twin, but a state that was already homed stays homed, and a call
of a listed row on it can still run the twin. The homed copy is a new
allocation that the CUDA driver makes outside PyTorch's caching allocator
(``torch.cuda.memory_allocated()`` does not count it); the input state stays
allocated until the caller drops it. Make one eager call before a CUDA graph
capture, and keep the state alive as long as a graph that reads it.

.. autosummary::
    :toctree: ../generated

    localize_gdn_decode_state

ReplaySSM speculative verification (SM100/SM103, SM120)
-------------------------------------------------------

Use ``gated_delta_rule_mtp(cache_replayssm=True, disable_state_update=True)``
with FP32 128x128 checkpoints. For each layer, pass contiguous views into
all-layer BF16 raw K/V windows and FP32 log-decay/beta windows. The windows are
indexed by the same pool slots as the checkpoints. Verify produces outputs
without changing the checkpoint or storing a full state for every token.

Once the accepted lengths are known, call ``gated_delta_rule_replayssm_commit``
once across all layers. It replays only the accepted prefix, or the accepted
root-to-node path when ``accept_paths`` is given, and updates each live
checkpoint in place. Zero acceptance leaves the checkpoint unchanged.
The next verify overwrites the raw windows; no circular cache or deferred
state update is retained between rounds. Use identical Q/K normalization
settings for verify and commit.

Verify has a TF32 tensor-core specialization on each architecture. Both
require T=4..8, B=1..256, BF16 output, int32 indices, contiguous FP32
checkpoints and normalized Q/K; other supported inputs use the FP32 MTP
fallback. TF32 contraction changes numerical reduction precision, so outputs
need not be bitwise equal to the fallback. See the API docstrings for shape,
index uniqueness, padding and optional tracked-checkpoint contracts.

- SM100/SM103: the tcgen05 specialization uses TMA and tcgen05 MMA, and also
  requires an even value-head/key-head ratio.
- SM120 has no tcgen05. Its specialization uses warp-level ``mma.sync`` TF32
  MMA and splits each head's value rows across several CTAs. It is selected
  when ``B * HV <= 128`` or T>=7, where it is generally faster than the FP32
  fallback; larger batches at T<=6 keep the fallback.

Commit defaults to ``backend="auto"``. On SM100/SM103, with normalized K and
no tracking, T=8 uses tcgen05; T=6/7 uses tcgen05 when B>1 and
``layers * B * HV >= 256``. Other cases, including every tracked commit, use
SIMT. Explicit ``backend="tcgen05"`` supports T=4..8 and, with tracking, writes
the FP32 snapshots in a SIMT pass before the TF32 fold. The heuristic uses
tensor metadata, not device-resident accepted lengths; short accepted prefixes
may favor ``backend="simt"``. SIMT also retains FP32 contractions when
checkpoint accuracy is preferred over tensor-core throughput. On SM120, commit
always uses the FP32 SIMT kernel: ``auto`` resolves to ``simt`` and explicit
``backend="tcgen05"`` is rejected.
