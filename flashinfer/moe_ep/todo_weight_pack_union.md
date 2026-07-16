# TODO: split MoEWeightPack into a discriminated union (Unquantized | Prequantized)

Status: analysis only, nothing implemented yet (2026-07-14). Suggested in
review: replace the single optional-field dataclass with

```python
@dataclass(frozen=True)
class UnquantizedMoEWeights:
    w13: torch.Tensor  # [local_experts, 2*intermediate, hidden] bf16/fp32
    w2: torch.Tensor   # [local_experts, hidden, intermediate]

@dataclass(frozen=True)
class PrequantizedMoEWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor

MoEWeightPack = UnquantizedMoEWeights | PrequantizedMoEWeights
```

Verdict: worth doing. It's not just cosmetic — the current optional-field
shape has a live silent-misbehavior class (below), and every consumer
already discriminates on the scales anyway.

## Motivating bug: mixed scale state is silently mis-handled today

All three mega backends discriminate quantized-vs-canonical by testing the
optional scales:

- `backends/mega/kernel/nvfp4_cutedsl/weights.py:129` and
  `mxfp8_cutedsl/weights.py:161`:
  `if weights.w13_scale is not None and weights.w2_scale is not None:`
- `backends/mega/kernel/deep_gemm_mega/weights.py:74`:
  `if weights.w13_scale is None or weights.w2_scale is None:` → re-quantize

So a pack with **exactly one** scale set (user mistake, e.g. forgot
`w2_scale`) is not rejected — it silently falls into the
"unquantized, re-quantize from bf16" path in all three backends, ignoring
the one scale that was provided, and then fails later with confusing dtype/
shape errors (or worse, quietly re-quantizes already-quantized int8 data).
The union makes this state unrepresentable: `PrequantizedMoEWeights`
requires both scales, `UnquantizedMoEWeights` has none.

## Secondary wins

- Discrimination becomes `isinstance(weights, PrequantizedMoEWeights)`
  instead of two-field None tests repeated in three backends.
- Docstrings live with the variant they describe (the current
  `MoEWeightPack` docstring interleaves both recipes in one paragraph).
- `frozen=True` on both variants (current class is mutable for no reason —
  audit for mutation first, none expected).
- Recipe-specific expectations (DeepGEMM fp4/ue8m0 vs NVFP4 vs MXFP8) stay
  backend-validated at `preprocess_weights` time, as now — the union
  encodes *quantized-or-not*, and the selected backend implies the recipe.
  Going further (per-recipe weight classes with a `kind`) is possible but
  not needed; note it as a rejected-for-now extension.

## Migration plan

1. Add the two variant classes + `MoEWeightPack = Unquantized | Prequantized`
   alias in `weights.py`.
2. **Back-compat constructor**: `MoEWeightPack(...)` is called positionally/
   by-kwarg across tests and by external users; a bare union alias is not
   callable. Provide a factory shim with the old signature —
   `def MoEWeightPack(w13, w2, w13_scale=None, w2_scale=None)` returning the
   right variant — that raises a clear error on the mixed state (one scale
   set). Deprecate later; alternatively name the factory
   `make_moe_weight_pack` and keep `MoEWeightPack` purely as the type alias
   (breaking construction; decide with reviewers).
3. `isinstance(pack, MoEWeightPack)` in
   `core/validation/common.py:245` works with a PEP 604 union on
   Python 3.10+ (fine for this repo) — but re-check every isinstance site.
4. Update backend `preprocess_weights` discrimination in the three mega
   backends (nvfp4_cutedsl, mxfp8_cutedsl, deep_gemm_mega) to
   `isinstance`-dispatch; delete the None tests.
5. `dummy_moe_weights` (`weights.py`) returns `UnquantizedMoEWeights`.
6. Split path (`backends/split/kernel/fused_moe/weights.py`) and layer/mode
   plumbing (`layer.py`, `modes/*.py`, `core/kernel/base.py`) — type
   annotations only, no behavior change expected.
7. Tests: update constructors across `tests/moe_ep/` (~10 files touch
   `MoEWeightPack`); add a test that the mixed-scale state raises at
   construction instead of silently re-quantizing.

## Scope check

`MoEWeightPack` is referenced from ~30 files (backends, validation, layer,
modes, tests) but almost all are annotations/pass-through; the behavioral
touch points are the three backend discrimination branches and shared
validation. Small-to-medium, mechanical, low risk — good candidate to land
before more backends copy the None-test pattern.
