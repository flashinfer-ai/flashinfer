# Native sparse MLA

`BatchSparseMLADecodePagedTSWrapper` implements Dqk = Dv = 512 sparse
attention on SM100/SM103 through the existing Prims-TS MLA families. Q and KV
use matching native BF16 or E4M3; output is BF16. Calling the API opts into the
existing experimental Prims-TS backend. CuTe DSL 4.7 is required.

## Code organization

- `attention/prims_ts/sparse_mla_decode.py`: the public `plan()` / `run()`
  wrapper, input validation, workspace ownership and kernel launch.
- `sparse_policy.py`: datatype-specific schedule selection, buffer sizing and
  split/V partition sizing from planned workload bounds and SM count.
- `sparse_views.py`: live per-request storage-row metadata used by the loaders.
- `sparse_reduce.py`: split reduction and joint normalization of independent
  source passes, including sinks and LSE.
- `helpers/gather.py` and the existing 1-CTA/2-CTA families: TMA gather4,
  cached offsets, retained KV and softmax/correction pipelines.

## Interface

```python
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

wrapper = BatchSparseMLADecodePagedTSWrapper(workspace_buffer)
wrapper.plan(q.device, B, H, max_topk=K, max_extra_topk=K_extra,
             max_seq_len_q=SQ, q_data_type=q.dtype,
             has_sinks=True, return_lse=True)
# Applications prepare SparseMLAPreparedMetadata before attention.
out, lse = wrapper.run(q, kv_cache, metadata, extra_kv_cache=extra_cache,
                       sinks=sinks, softmax_scale=model_softmax_scale)
```

Omit the extra pool and use `max_extra_topk=0` for one selected source.
Neither pool is required to be SWA. Both lists contribute to one softmax;
duplicate entries retain their multiplicity. Native pools may be paged
`[pages,page_size,512]` or singleton-head NHD/HND tensors, including page
padding. Queries are `[B,SQ,H,512]`, or `[total_q,H,512]` with `packed_query=True`
and `qo_indptr`. Projection, RoPE, compression and index selection belong to
the caller; packed FP8 cache formats are not accepted.

FP8 accepts positive per-tensor Q and per-source KV descales. Independent
source descales use two attention passes and a joint merge; shared scales
use one pass. BF16 descales must be one. Sinks add denominator mass only;
returned LSE is the natural logarithm and excludes sinks. Empty rows produce
zero output and negative-infinity LSE. Selected values and effective scale
products must be finite.

## Prepared metadata

`SparseMLAPreparedMetadata` contains caller-owned CUDA tensors. Indices already
include physical page strides; attention consumes storage rows directly.

| Field | Shape/type | Meaning |
|---|---|---|
| `indices`, `extra_indices` | int32 `[rows, source_capacity]` | Storage rows; `-1` invalid |
| `lengths`, `extra_lengths` | int32 `[rows]` | Active source prefixes |
| `routes` | int32 `[passes, rows, route_capacity]` | Bit 31 selects the extra source; low 31 bits are its row; `0x7fffffff` invalid |
| `execution_lengths`, `valid_counts` | int32 `[passes, rows]` | Padded execution span and valid entry count |
| `scale_params` | FP32 `[passes, 2 + H + rows]` | QK scale, PV scale, sinks, counts |

Each source span is rounded to 128; route capacity is at least 256. Empty rows
execute one masked slot. Independent scales use two passes. Supply all
representations so metadata remains compatible with automatic dispatch;
direct-index schedules can omit the combined fields. The producer must keep
all representations consistent when indices, lengths, scales or sinks change.

Warm up before CUDA Graph capture. Use stable buffers, preallocated outputs
and `validate=False` during capture; keep all inputs, metadata and workspace
alive through replay. One wrapper/workspace supports one in-flight call.
Buffer overlap is unsupported. The workspace-size helper accepts the same
arguments as `plan`; the eager helper combines planning and execution.

## Scheduling and validation

Split-KV and V partitions fill small grids. Larger query tiles share gather
work across heads; KV reuse retains K through PV when one CTA owns V and the
buffer budget permits it. BF16 BK64 permits two retained tiles where BK128
would not fit. Selection uses host bounds; live lengths remain on the GPU.
BF16 deferred maxima use FlashMLA's six-log2 bound; FP8 uses exact maxima and
P scale 448. `assume_valid_prefix=True` promises no holes before each length
and enables analytic masking in the direct FP8 2-CTA path.

DSL 4.7's credit simulator has a bounded iteration count that can report
unconsumed commits before BF16 retained-KV schedules drain. Those schedules
use the stock task manager's warning mode: checks still run, but failures
warn rather than abort compilation. Other schedules retain strict checking.

Accuracy, graph replay, output/workspace ownership and important validation
checks are in one file. Its small Triton preparer adapts vLLM's page-stride
mapping and compaction; its FP64 reference preserves the probability/output
rounding bound used for FP8.

```bash
pytest tests/attention/test_prims_ts_sparse_mla.py
```

Benchmarks are maintained locally, outside this change. One static two-source
BF16 definition remains under `tests/trace/fi_trace_out` as an example;
automatic `fi_trace()` integration is not included.
