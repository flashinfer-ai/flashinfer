"""Explicit FP32 prefill precision, pooled-state continuity, and captured replay."""

from contextlib import contextmanager
from dataclasses import replace
import json
import subprocess
import sys

import pytest
import torch

from b12x.preparation import DeviceIdentity, PreparationSession, PreparedCall
from b12x.sequence import gdn_prefill
from b12x.sequence.gdn_prefill._tuning import TUNING
from b12x.testing.delta_prefill_cases import PrefillCase, make_inputs

from ..conftest import require_b12x


@pytest.fixture(autouse=True)
def _ieee_reference_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


def _caps(**changes):
    values = dict(
        device="cuda:0",
        max_tokens=512,
        max_seqs=4,
        max_state_slots=13,
        key_heads=16,
        value_heads=48,
        checkpoint_export=True,
        null_state_index=0,
        intermediate_dtype=torch.float32,
    )
    return gdn_prefill.Caps(**(values | changes))


def test_precision_is_explicit_and_separates_selection_keys():
    identity = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
    precise = gdn_prefill.plan(_caps())
    chunk = gdn_prefill.plan(_caps(intermediate_dtype=torch.bfloat16))
    assert precise.query.intermediate_dtype == "float32"
    assert chunk.query.intermediate_dtype == "bfloat16"
    selected = TUNING.configure(precise.query, device=identity, override=None)
    baseline = TUNING.configure(chunk.query, device=identity, override=None)
    assert selected.encoded_query != baseline.encoded_query
    assert (
        gdn_prefill.Caps(
            device="cuda:0",
            max_tokens=512,
            max_seqs=4,
            max_state_slots=13,
            key_heads=16,
            value_heads=48,
        ).intermediate_dtype
        == torch.bfloat16
    )
    from b12x.sequence.gdn_prefill._tuning import _tuning_parameters

    choices = list(_tuning_parameters(precise.query, identity).configurations())
    assert len(choices) == 1
    with pytest.raises(ValueError, match="fixed sequential"):
        TUNING.validate_config(
            precise.query,
            replace(
                TUNING.default_config(precise.query, identity),
                algorithm="chunk_parallel",
            ),
            identity,
        )
    with pytest.raises(ValueError, match="intermediate_dtype"):
        _caps(intermediate_dtype=torch.float16)


def test_fp32_compile_plan_is_single_program_without_pool_or_token_specialization():
    program = r"""
import json, multiprocessing
from dataclasses import replace
from b12x._lib import compile_pool
compile_pool._initialize_worker(0, (12,0), "synthetic", "synthetic SM120", 148, 232448, 232448,
    multiprocessing.get_context("spawn").Array("q", (0,0)))
from b12x._lib.compile_pool import CompileJob, describe_compilation
from b12x.preparation import DeviceIdentity
from b12x.sequence.gdn_prefill._tuning import TUNING, GdnPrefillQuery
from b12x.sequence.gdn_prefill._impl import _materialize_layout
from b12x.sequence._shared.delta_prefill.preparation import _caps_from_query
from b12x.sequence.gdn_prefill import _impl
identity = DeviceIdentity("nvidia", (12,0), 148, "synthetic SM120")
query = GdnPrefillQuery(key_heads=16, value_heads=48, head_dim=128, model_dtype="bfloat16",
    state_dtype="float32", intermediate_dtype="float32", qk_l2norm=True, checkpoint_export=True,
    max_tokens=512, max_seqs=4, max_state_slots=13, null_state_index=0)
records=[]
for tokens,slots in [(512,13),(64,26130)]:
    q=replace(query,max_tokens=tokens,max_state_slots=slots)
    config=TUNING.default_config(q,identity)
    job=CompileJob.create("b12x.sequence._shared.delta_prefill.preparation:compile_prefill",
        "gdn_prefill",q.to_dict(),TUNING.encode_config(config),0)
    programs=describe_compilation(job).programs
    layout=_materialize_layout(_caps_from_query(_impl,q,0,is_gdn=True),config)
    records.append({"programs":[[p.name,p.key] for p in programs],"scratch":layout.scratch_specs()[0].shape[0],"windows":layout.workspace_windows})
print(json.dumps(records))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = json.loads(result.stdout.strip().splitlines()[-1])
    assert records[0]["programs"] == records[1]["programs"]
    assert len(records[0]["programs"]) == 1
    assert records[0]["programs"][0][0] == "sequence.gdn_prefill.recurrent_fp32"
    assert all(value["scratch"] < 8192 and value["windows"] == 0 for value in records)


@contextmanager
def _prepared(values, *, qk_l2norm=True):
    args = dict(values)
    args["a"], args["b"] = args.pop("raw_g"), args.pop("raw_beta")
    caps = _caps(
        device=args["q"].device,
        max_tokens=args["q"].shape[0],
        max_seqs=args["cu_seqlens"].numel() - 1,
        max_state_slots=args["recurrent_state"].shape[0],
        qk_l2norm=qk_l2norm,
    )
    plan = gdn_prefill.plan(
        caps, invocation=gdn_prefill.invocation_from_tensors(**args)
    )
    saved = {}

    def prepare(state):
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=args["q"].device)
        binding = state.bind(scratch=scratch, **args)
        saved["binding"] = binding
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    with PreparationSession(
        device=args["q"].device, autotune=False, compile_workers=0
    ) as session:
        session.prepare((plan.request(name="fp32", prepare_call=prepare),))
        binding = saved["binding"]
        yield session, binding, lambda: plan.prepared.state.run(binding)


def _reference(values, *, qk_l2norm=True):
    args = {key: value.clone() for key, value in values.items()}
    args["output"].zero_()
    args["a"], args["b"] = args.pop("raw_g"), args.pop("raw_beta")
    gdn_prefill.reference.prefill_gdn(**args, null_state_index=0, qk_l2norm=qk_l2norm)
    return args["output"], args["recurrent_state"]


@pytest.mark.parametrize(
    "lengths", [(1,), (5,), (17, 1), (65, 1), (512,), (64, 0, 33, 1)]
)
@torch.no_grad()
def test_fp32_prefill_matches_unrounded_recurrence_and_checkpoints(lengths):
    device = require_b12x()
    case = PrefillCase("gdn", 16, 48, lengths)
    values = make_inputs(
        case, device=device, max_tokens=max(512, sum(lengths) + 3), max_seqs=4
    )
    values["recurrent_state"][0].fill_(float("nan"))
    for i, length in enumerate(lengths):
        values["checkpoint_offsets"][i] = (length // 16) * 16
    original = values["recurrent_state"].clone()
    expected, state = _reference(values)
    with _prepared(values) as (session, binding, run):
        values["recurrent_state"].copy_(original)
        run()
        torch.testing.assert_close(values["output"], expected, rtol=0.008, atol=2e-5)
        torch.testing.assert_close(
            values["recurrent_state"], state, rtol=1e-4, atol=2e-6, equal_nan=True
        )
        assert binding.scratch.numel() < 8192
        assert torch.isnan(values["recurrent_state"][0]).all()


@torch.no_grad()
def test_fp32_graph_replay_strides_counts_and_inplace_continuation():
    device = require_b12x()
    case = PrefillCase("gdn", 16, 48, (65, 1))
    values = make_inputs(case, device=device, max_tokens=512, max_seqs=4)
    width = 48 * 128 * 128
    storage = torch.full((13, width + 64), 19.0, device=device)
    state = storage[:, :width].view(13, 48, 128, 128)
    state.copy_(values["recurrent_state"])
    values["recurrent_state"] = state
    packed = torch.zeros((512, 10240), dtype=torch.bfloat16, device=device)
    for key, start, stop, heads in [
        ("q", 0, 2048, 16),
        ("k", 2048, 4096, 16),
        ("v", 4096, 10240, 48),
    ]:
        view = packed[:, start:stop].view(512, heads, 128)
        view.copy_(values[key])
        values[key] = view
    beta = torch.zeros((512, 96), dtype=torch.bfloat16, device=device)
    beta[:, ::2].copy_(values["raw_beta"])
    values["raw_beta"] = beta[:, ::2]
    final = torch.full((8,), 2**40, dtype=torch.int64, device=device)
    final[::2].copy_(values["final_state_indices"])
    values["final_state_indices"] = final[::2]
    for name in ("initial_state_indices", "checkpoint_state_indices"):
        values[name] = values[name].to(torch.int64)
    output = torch.full(
        (512, 48 * 128 + 64), float("nan"), dtype=torch.bfloat16, device=device
    )
    values["output"] = output[:, : 48 * 128].view(512, 48, 128)
    values["checkpoint_offsets"][0] = 64
    original = state.clone()
    with _prepared(values) as (session, _, run):
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        for lengths, initials, finals, offsets in [
            ((65, 1), (0, 1), (4, 5), (64, 0)),
            ((3, 2), (4, 5), (4, 5), (0, 0)),
            ((0, 0), (4, 5), (6, 7), (0, 0)),
            ((), (), (), ()),
        ]:
            if lengths == (65, 1):
                state.copy_(original)
            n = len(lengths)
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
            for key, data, pad in [
                ("cu_seqlens", cu, -12345),
                ("initial_state_indices", initials, 2**30),
                ("final_state_indices", finals, 2**30),
                ("checkpoint_offsets", offsets, -12345),
            ]:
                destination = values[key]
                destination.copy_(
                    torch.tensor(
                        list(data) + [pad] * (destination.numel() - len(data)),
                        dtype=destination.dtype,
                        device=device,
                    )
                )
            values["num_seqs"].fill_(n)
            values["num_tokens"].fill_(sum(lengths))
            expected, expected_state = _reference(values)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(
                values["output"], expected, rtol=0.008, atol=2e-5
            )
            torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=2e-6)
            assert torch.all(storage[:, width:] == 19)
            assert torch.isnan(output[:, 48 * 128 :]).all()
        graph.reset()


@torch.no_grad()
def test_fp32_snapshot_restore_is_bit_identical_across_prefill_capacities():
    device = require_b12x()
    values = make_inputs(
        PrefillCase("gdn", 16, 48, (92,)), device=device, max_tokens=128, max_seqs=1
    )
    values["checkpoint_offsets"][0] = 64
    original = values["recurrent_state"].clone()
    with _prepared(values) as (session, _, run):
        full_graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(full_graph):
            run()
        values["recurrent_state"].copy_(original)
        full_graph.replay()
        torch.cuda.synchronize()
        uninterrupted = values["output"][:92].clone()
        final = values["recurrent_state"][1].clone()
        snapshot = values["recurrent_state"][2].clone()
        full_graph.reset()

    tail = dict(values)
    for name in ("q", "k", "v", "raw_g", "raw_beta", "output"):
        tail[name] = values[name][64:128]
    tail["cu_seqlens"] = torch.tensor([0, 28], dtype=torch.int32, device=device)
    tail["initial_state_indices"] = torch.tensor([2], dtype=torch.int32, device=device)
    tail["final_state_indices"] = torch.tensor([3], dtype=torch.int32, device=device)
    tail["checkpoint_offsets"] = torch.zeros(1, dtype=torch.int32, device=device)
    tail["num_tokens"] = torch.tensor([28], dtype=torch.int32, device=device)
    with _prepared(tail) as (session, _, run):
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(tail["output"][:28], uninterrupted[64:])
        assert torch.equal(tail["recurrent_state"][3], final)
        assert torch.equal(tail["recurrent_state"][2], snapshot)
        graph.reset()


@torch.no_grad()
def test_fp32_prefill_accepts_bf16_parameters_without_qk_normalization():
    values = make_inputs(
        PrefillCase("gdn", 16, 48, (17,)),
        device=require_b12x(),
        max_tokens=32,
        max_seqs=1,
    )
    values["A_log"] = values["A_log"].bfloat16()
    values["dt_bias"] = values["dt_bias"].bfloat16()
    values["checkpoint_offsets"][0] = 16
    initial = values["recurrent_state"].clone()
    expected, state = _reference(values, qk_l2norm=False)
    with _prepared(values, qk_l2norm=False) as (_, _, run):
        values["recurrent_state"].copy_(initial)
        run()
        torch.testing.assert_close(values["output"], expected, rtol=0.008, atol=2e-5)
        torch.testing.assert_close(
            values["recurrent_state"], state, rtol=1e-4, atol=2e-6
        )
