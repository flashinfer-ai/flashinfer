# Experimental Task-Scheduled Attention

`flashinfer.attention.prims_ts` exposes experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.

## CUTLASS DSL version policy

Published FlashInfer packages, including nightlies, retain
`nvidia-cutlass-dsl==4.6.2` for the general runtime. The experimental PrimTS
attention kernels use APIs introduced in CUTLASS DSL 4.7 and require
`nvidia-cutlass-dsl>=4.7.0`. Importing `flashinfer.attention.prims_ts` with the
default 4.6.2 dependency raises a feature-local error; importing and using
other FlashInfer APIs does not require the PrimTS override.

FlashInfer source CI and documentation builds replace the complete CUTLASS DSL
package stack with 4.7.0 before validating PrimTS. That test-only environment
does not change the dependency metadata of release or nightly artifacts, whose
consumer tests restore and verify 4.6.2.

To opt into PrimTS, replace the installed DSL stack with the package for the
environment's CUDA major version:

```bash
python -m pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-core \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13

# CUDA 12
python -m pip install "nvidia-cutlass-dsl==4.7.0"

# CUDA 13
python -m pip install "nvidia-cutlass-dsl[cu13]==4.7.0"
```

This is a feature-specific override of FlashInfer's 4.6.2 package dependency,
so `pip check` will report the intentional version difference. Keep the
override confined to environments that use or validate PrimTS.

## Guides and public APIs

Import all entries below from `flashinfer.attention.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md) | `BatchPrefillTSWrapper`, `batch_prefill`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prims_ts_batch_decode_with_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_decode_mla_with_paged_kv_cache`, `get_prims_ts_batch_decode_mla_workspace_size`, `prims_ts_batch_decode_with_kv_cache_mla` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

## Validation

After installing CUTLASS DSL 4.7.0, run the numerical, graph,
scheduler/resource, alias-safety, and public-surface contracts:

```bash
python -c 'import importlib.metadata as m; assert m.version("nvidia-cutlass-dsl") == "4.7.0"'
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_mask.py \
  tests/attention/test_attention_ts_mla_decode.py
```
