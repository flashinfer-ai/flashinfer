# Sparse MLA and the experimental V4.1 adapter

Main's generic sparse MLA remains the native BF16/E4M3, prepared-list interface
in `flashinfer/attention/prims_ts/mla_decode.py`. Its policy selection, output
quantization/group-major layout, 2CTA kernels and separate reducers are not
replaced by the model-specific adapter below.

## Private V4.1 specialization

`flashinfer/experimental/deepseek_v41/prims_ts.py` configures one joint
finalizing attention for SM100, H64, Sq1 and D512. It requires explicit E4M3 Q
and typed E4M3 SWA storage, plus optional packed FP4 main storage. CuTe-DSL
>=4.8 with its bundled CUDA 13 compiler is required. This path is experimental,
JIT-only and selected explicitly with `backend="prims-ts"`; it has no automatic
selection, BF16-Q adapter, AOT registration or per-source merge.

The fixed schedule is Keep-AB/M64/BK128, retained KV8, four load warps,
one page-offset stage, split1 and nonpersistent. A fixed 128-slot SWA HEAD
precedes `ceil(K/128)` main LOOP tiles for `K=0,64,...,512`, including when the
SWA prefix is empty. The declared capacity is at least 256; live scheduler
extent remains `128 + ceil(K/128)*128`. One softmax/PV state spans both sources.
Graph topology checks cover `K=0,64,192,512`, with/without sinks; this is not
full-model qualification.

## Routing and publication

The adapter uses main's seven-item sparse metadata ABI. Its producer-owned
binder reads the original combined int32 table `[B,128+K]` and live lengths
inside attention; it does not materialize a prepared route list. Topk lengths
include the fixed 128 SWA positions. SWA visibility clamps to `[0,128]`; main
visibility clamps `topk-128` to `[0,K]` without signed-underflow arithmetic.

Private `CombinedSparseRouteView` and `CombinedSparseTile` subclass main's
`SparseKVIndices` and `SparseSourceTile` protocols. They implement
`bind_query_row`, `select_source_tile`, `load_row`, `load_quad` and
`load_full_quad`, and use `SparseKVLengths` for scheduler extents. They never
claim `valid_prefix`: negative IDs, INT32_MAX and out-of-capacity IDs remain
masked even on full-quad staging paths. Holes and duplicate multiplicity are
preserved. Physical pointer products widen to Int64 before applying strides.

The custom producer retains K through delayed PV and publishes aligned SW128
shared-memory stores through AsyncThread-to-UMMA synchronization. All 128
writers issue their async-proxy fence and arrival. SWA intentionally uses
thread loads; Q and generic native cache paths retain their existing TMA path.
The producer receives `(main_cache, swa_cache)` in that order.

## Arithmetic and ownership

Both caches must already use common cache units. Main staging rounds/saturates
exact `E2M1 * original_group16_E4M3_scale` directly to E4M3, retaining signed
zeros/subnormals and intentional +/-448 saturation. There is no KV amax,
request-selected scale tensor, division or repeated scale restoration.

Default-false `sparse_split_softmax_scale` is restricted to this fixed producer
topology. It retains raw maxima, applies positive finite FP32 BMM1 to raw score
differences before log2 conversion, and zeros masked probabilities before their
sum and FP8 pack. The log2 constant is materialized as FP32 SSA before task
construction. Sink subtraction uses natural-domain RN/non-FTZ FMA. Internal FP32
LSE is natural-log and sink-exclusive; output factoring does not alter that path.
Exact empty rows produce O=0/LSE=-inf. Generic native arithmetic stays on its
existing branch when this specialization is disabled.

The private finalizer preserves the original one-factor normalization for
`2**-48 <= abs(BMM2) <= 2**48` and log2 sink suppression at most48. Outside that
runtime window, it combines BMM2 amplification and sink suppression before
materializing the numerator: let `delta` be the log2 sink-minus-anchor difference,
`t=max(delta,0)` and `D=S*exp2(-t)+448*exp2(min(delta,0))`. Disabled sinks use
`t=0,D=S`. With `n=log2(abs(BMM2))-t` and `r=exp2(n/2)`, output is evaluated as
`(PV*r)*(sign(BMM2)*r/D)` with two RN/non-FTZ multiplies before BF16 packing.
Subnormal BMM2 is lifted by `2**24` with non-FTZ multiplication before log2, then
its exponent is corrected. Bitwise zero classification keeps signed zero and
nonzero subnormal inputs distinct. Explicit zero factors cover empty/zero rows.

The fixed native bounds `abs(PV)<2**27` and `448<=D<2**19` keep both factors
normal for `n>=-192` and prevent overflow in the first multiply; smaller `n`
rounds to BF16 zero even at maximum PV. This avoids a tiny intermediate sink
factor or early BMM2 division destroying a recoverable result. It does not remove
true BF16 output overflow. Keep-AB's otherwise unused `final_scale1` carries the
private post-factor only; generic dual-accumulator scale semantics are unchanged.
TMEM loads remain warp-convergent, with no extra launch, scratch or scale-value
specialization. Rare-path cost has not been measured.

Task initialization and domain/cache snapshots execute at Python tracing time.
Runtime activity predicates guard task bodies, not TaskManager initialization;
main's shared `task_context_kwargs` handles the DSL task API difference.

The native API returns output only. Its workspace-size helper defaults to and
accepts only Prims-TS. Caller-owned workspace contains private LSE and the K0
dummy main storage, not prepared metadata or source states. Native decode defaults
to HND; the shared writer retains its original NHD default, so request matching
layouts explicitly. Capture requires caller output/workspace and eager warmup.
Keep addresses alive/fixed and refresh values in place; concurrent calls need
separate writable buffers or synchronization. Output/workspace must not overlap
inputs or each other. The existing default backend keeps its original plan ABI.

## Validation boundary

The native decode/cache tests under `tests/experimental/` retain independent
represented-input FP64 references, the componentwise FP8 output bound,
`rtol=0, atol=1e-5` sink-exclusive LSE, exact empty/nonfinite checks and bitwise
layout/replay parity. Host contract files cover routing, task control, defaults
and workspace ownership. Static/CLC task-cache graph coverage remains in
`tests/attention/test_attention_ts_sparse_mla.py`.

New regressions cover strong sinks with large signed BMM2, signed zero,
analytic BF16-subnormal output and live metadata/layout/replay parity. After
backend cleanup and finalizer changes, native-only frozen-source verification
on B200 (SM100) passed 197 host-contract and 100 GPU cases, with no failures or
skips. The run used PyTorch 2.11.0+cu128, CuTe-DSL 4.8.0 and bundled CUDA 13.4;
JSON/JUnit results agreed and source fingerprints remained unchanged throughout.
These historical counts predate the rebase and host-test cleanup; tests were not
rerun for those changes. The Frost decoder, examples and benchmarks were not run.
The generic sparse MLA suite, complete integration, full-model equivalence and
performance remain unqualified by this run. See
[the example README](../../../../../examples/deepseek_v41/README.md) for the
public API, cache formats and example commands.
