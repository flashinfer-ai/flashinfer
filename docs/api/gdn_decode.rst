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

With ``ugpu_row_split=n`` the call is also localized per die: rows ``[0, n)`` run
on die 0's SMs and the rest on die 1's (green contexts, two launches forked from
and joined back to the current stream). Allocate the state pool with
:func:`flashinfer.green_ctx.empty_by_locality_domain` and order the batch so that
rows below ``n`` use slots in die 0's half; then no state byte crosses the
die-to-die link. On Rubin VR200 this takes B=512, HV=64 from ~163 us to ~131 us.

ReplaySSM speculative verification (SM100/SM103)
---------------------------------------------

Use ``gated_delta_rule_mtp(cache_replayssm=True, disable_state_update=True)``
with FP32 128x128 checkpoints. For each layer, pass contiguous views into
all-layer BF16 raw K/V windows and FP32 log-decay/beta windows. The windows are
indexed by the same pool slots as the checkpoints. Verify produces outputs
without changing the checkpoint or storing a full state for every token.

Once the accepted lengths are known, call ``gated_delta_rule_replayssm_commit``
once across all layers. It replays only the accepted prefix and updates each
live checkpoint in place. Zero acceptance leaves the checkpoint unchanged.
The next verify overwrites the raw windows; no circular cache or deferred
state update is retained between rounds. Use identical Q/K normalization
settings for verify and commit.

The tcgen05 verify specialization uses TMA and TF32 tensor cores for T=4..8,
B=1..256, BF16 output, int32 indices, contiguous FP32 checkpoints, normalized
Q/K and an even value-head/key-head ratio. Other supported SM100 inputs use
the FP32 MTP fallback. TF32 contraction changes numerical reduction precision.
See the API docstrings for shape, index uniqueness, padding and optional
tracked-checkpoint contracts.

Commit defaults to ``backend="auto"``. With normalized K and no tracking,
T=8 uses tcgen05; T=6/7 uses tcgen05 when B>1 and ``layers * B * HV >= 256``.
Other cases use SIMT. Explicit ``backend="tcgen05"`` supports T=4..8.
The heuristic uses tensor metadata, not device-resident accepted lengths;
short accepted prefixes may favor ``backend="simt"``. SIMT also retains FP32
contractions when checkpoint accuracy is preferred over tensor-core throughput.
