# TODO: MoEEpMegaLayer retains source weight pack after preprocess (OOM)

Status: analysis only; fixed downstream in the vLLM wrapper patch, not yet
upstream (2026-07-16). Found during the 2026-07-15 vLLM e2e integration
(run log: `$LUSTRE_ROOT/moe_ep_benchmark/vllm_e2e/RUNS.md`, run 7).

## Symptom

`MoEEpMegaLayer.__init__` stores `self._weights: MoEWeightPack = weights`
(`modes/mega_layer.py:64`) and never drops it after
`_preprocess_weights()` produces `self._transformed`. For the cutedsl
backends in the fp4-checkpoint flow, `weights` is a per-layer **bf16 dequant
pack (~3.2 GB at DeepSeek-V4-Flash geometry)**; across 43 MoE layers that is
~140 GB of dead weight — OOM on a 186 GB GB200 at model load. The
transformed tensors are self-sufficient: nothing reads `self._weights` after
a successful transform (verified in the 07-15 integration; the vLLM patch
sets `layer._weights = None` post-construction and everything works,
correctness smokes included).

## Why it's not just "drop it in `__init__`"

- `forward()` has a lazy `_preprocess_weights()` call
  (`modes/mega_layer.py:113-119`), but that path raises unless
  `transformed_weights` was provided at init — so in practice the only
  consumers of `self._weights` are `validate_fleet_weights` at init and the
  single `preprocess_weights()` call. Re-audit before landing (grep
  `_weights` across `modes/`, `core/validation/`, tests) so the lazy path
  can't dereference a released pack.
- The `backend.transformed_weights is not None` path never needs the pack at
  all — it's stored anyway today.

## Fix plan

1. In `MoEEpMegaLayer`, release the source pack once `self._transformed`
   exists: set `self._weights = None` at the end of `_preprocess_weights()`
   and skip storing it when `backend.transformed_weights` was provided.
   Keep it ONLY in the `preprocess_weights=False`-without-transformed error
   path (which raises anyway).
2. Same audit for the split path (`backends/split/`) and any other mode that
   copies the mega pattern.
3. Type the field `Optional[MoEWeightPack]` and make the invariant explicit
   in the class docstring: "source weights are released after transform;
   transformed tensors own the memory."
4. Test: construct a mega layer with a dummy pack, assert the pack tensors'
   refcount/storage is released post-init (e.g. `weakref` on the tensor
   storage, or check `torch.cuda.memory_allocated` drops after
   `del weights; gc.collect()`).
5. Remove the corresponding workaround from the vLLM integration patch
   (`moe_ep_benchmark/vllm_e2e/patch_0251/fi_utils.py`) once upstream.

Related: [[todo_weight_pack_union]] — if the pack becomes a discriminated
union, the release point stays the same; do this one first, it's smaller.
