# Blackwell BF16 rank-major source drop

This directory contains one generated CUDA source bundle for the capacity-
bounded SM100 BF16 rank-major MoE-EP layer and the FlashInfer runtime adapter
that owns its workspace and launch sequence. Each launch accepts an active
prefix of 1 to 128 rows per rank; every rank must use the same active row count.
Up to 32 active rows per rank, `stage_inputs` already sends the rows to the
ranks that own their experts, so every rank must call `stage_inputs` and `run`
symmetrically; before the first `bind_weights`, it keeps a copy of the inputs
and `run` sends them. Every rank must therefore also call `bind_weights` at the
same point relative to `stage_inputs` and `run` (the layer API binds in
`compute`); otherwise a repeated `stage_inputs` completes the pending forward
on some ranks and replaces it on others, and the ranks pair different
forwards, so `run` hangs or returns rows computed from other inputs. Peers wait for a staged forward whose rows were sent, or
whose kernels `run` began to launch when a launch failed. The session
completes such a forward into internal scratch output on a repeated
`stage_inputs`, and on a `run` that rejects its arguments (which also sends
deferred rows), if expert weights are bound and the forward was staged in the
active CUDA graph capture or eagerly outside any capture; a later `run`
resumes it at the failed launch. Other staged inputs, including deferred rows
not sent yet, are replaced, also inside a capture. A CUDA graph that captures
`stage_inputs` must also capture the matching `run`, for every active row
count; inputs staged in a capture that is no longer active are dropped, and
`run` then reports that no inputs are staged. Inside a capture, `stage_inputs`
and `run` raise for an eagerly staged forward that peers wait for, and `run`
also for other eagerly staged inputs of up to 32 rows. Kernels that `run`
recorded into a capture before one of its launches failed never run: calls
inside that capture raise, and afterwards an eagerly staged forward restarts
at its first kernel. Do not replay such a graph.

The `src/` directory is an immutable generated pair:

- `flashinfer_blackwell_moe_ep_layer_sm100.cu` contains thirteen device
  kernels: the nine kernels of the pull path in launch order, then the four
  kernels of the flag-in-data path that the session launches instead for up
  to 32 active rows per rank. The pull path computes the route in one extra
  dispatch CTA for every active count, so it does not launch `route_fused`.
- `manifest.json` pins the source checksum, capacity/workload constraints,
  symbols, launch geometry, cluster geometry, dynamic shared memory, and PDL
  policy.

Do not edit either generated file independently. Updates must replace the
source and manifest together, preserve the public ABI, pass the
eight-rank correctness gates, and show paired no-regression measurements
against the previous source bundle.
