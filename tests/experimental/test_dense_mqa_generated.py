"""Dense prepared API: native-oracle numerics, exact metadata, non-default-stream
dependencies and changed-input graph replay on SM100a/SM103a.

DeepGEMM is a test oracle only; production imports and launches do not use it.
The private export campaign additionally reuses the original numerical fixtures.
"""

import pytest
import torch

from flashinfer.dense_mqa import prepare_dense_mqa_logits
from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as _runtime


def _skip_unless_exported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        arch = _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms not in _runtime.supported_num_sms(arch):
        pytest.skip(
            f"The exported {arch} schedules cover {_runtime.supported_num_sms(arch)} SMs, "
            f"this device has {sms}"
        )


def _inputs(precision, queries, keys):
    torch.manual_seed(101)
    if precision == "fp4":
        q = torch.randint(0, 256, (queries, 32, 64), device="cuda", dtype=torch.uint8)
        kv = torch.randint(0, 256, (keys, 64), device="cuda", dtype=torch.uint8)
        qs = torch.full((queries, 32, 4), 127, dtype=torch.uint8, device="cuda")
        ks = torch.full((keys, 4), 127, dtype=torch.uint8, device="cuda")
        native_q = (q.view(torch.int8), qs.view(torch.int32).view(queries, 32))
        native_kv = (kv.view(torch.int8), ks.view(torch.int32).view(keys))
        rows = queries
    else:
        rows = max(4, queries)
        q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        kv = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
        qs = None
        ks = torch.ones(keys, device="cuda")
        native_q, native_kv = (q[:queries], None), (kv, ks)
    weights = torch.randn(rows, 32, device="cuda")
    starts = torch.zeros(queries, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, keys)
    ends[1::3] = 17
    ends[2::3] = 0
    return q, kv, qs, ks, weights, starts, ends, native_q, native_kv


# The twenty scheduled model rows; the private campaign owns source/export timing.
CASES = [
    (precision, queries, keys)
    for precision in ("fp4", "fp8")
    for queries, keys in (
        (1, 4096),
        (1, 32768),
        (1, 131072),
        (16, 4096),
        (16, 32768),
        (16, 131072),
        (128, 4096),
        (128, 32768),
        (128, 131072),
        (16, 1048576),
    )
]


@pytest.mark.parametrize("precision,queries,keys", CASES)
def test_dense_mqa_stream_and_replay(precision, queries, keys):
    _skip_unless_exported()
    deep_gemm = pytest.importorskip("deep_gemm")
    q, kv, qs, ks, weights, starts, ends, nq, nk = _inputs(precision, queries, keys)
    plan = prepare_dense_mqa_logits(
        precision, q, kv, weights, starts, ends, q_scales=qs, kv_scales=ks
    )
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    old_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(sms)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            # A device-side input dependency must reach the current-stream launch.
            weights.mul_(0.5)
            plan.run()
        stream.synchronize()
        first = plan.output.clone()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.run()
        for changed in (False, True):
            with torch.cuda.stream(stream):
                if changed:
                    weights.neg_()
                    ends.copy_(keys - ends)
                plan.output.fill_(float("nan"))
                plan.metadata.fill_(0x55555555)
                graph.replay()
            stream.synchronize()
            meta = deep_gemm.get_mqa_logits_metadata(starts, ends, keys, 32)
            expected = deep_gemm.fp8_fp4_mqa_logits(
                q=nq,
                kv=nk,
                weights=weights[:queries],
                cu_seq_len_k_start=starts,
                cu_seq_len_k_end=ends,
                clean_logits=True,
                max_seqlen_k=0,
                logits_dtype=torch.float32,
                schedule_meta=meta,
            )
            torch.cuda.synchronize()
            actual = plan.logical_output
            finite = torch.isfinite(expected)
            assert torch.equal(torch.isfinite(actual), finite)
            assert bool(torch.isneginf(actual[~finite]).all())
            assert bool(torch.isneginf(plan.output[:, keys:]).all())
            torch.testing.assert_close(
                actual[finite],
                expected[finite],
                atol=1.0 if precision == "fp4" else 0.1,
                rtol=0.1,
            )
            if precision == "fp4":
                left, right = actual[finite].double(), expected[finite].double()
                denom = (left.square() + right.square()).sum()
                diff = (
                    0.0 if denom == 0 else float(1 - 2 * (left * right).sum() / denom)
                )
                assert diff < 5e-6
            span = (3 * sms + 1) // 2 * 2
            assert torch.equal(plan.metadata[: 3 * sms], meta[: 3 * sms])
            assert torch.equal(plan.metadata[span:], meta[span:])
            if not changed:
                assert torch.equal(plan.output, first)
    finally:
        deep_gemm.set_num_sms(old_sms)
