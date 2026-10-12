"""Raw GDN convolution parity, accepted history, and fixed-address replay."""

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import short_conv
from b12x.sequence.short_conv._tuning import ShortConvQuery

from ..conftest import require_b12x


def test_gdn_raw_projection_geometry_is_metadata_only():
    plan = short_conv.plan(
        short_conv.Caps(
            device="cuda:0",
            mode="mixed",
            channels=10240,
            kernel_size=4,
            max_tokens=16,
            max_seqs=4,
            max_state_slots=8,
            max_speculative_tokens=3,
        )
    )
    assert plan.query.state_length == 3
    assert plan.query.state_capacity == 6
    assert plan.query.state_strides == (61440, 6, 1)


@pytest.mark.parametrize(
    "override",
    [
        {"channels": True},
        {"mode": "eager"},
        {"kernel_size": 1},
        {"max_speculative_tokens": -1},
        {"dilation": 0},
        {"max_seqs": 2**31},
        {"state_strides": (32, 6, 1)},
        {"state_strides": (65536, 1, 10240)},
        {"input_alignments": (16,)},
        {"input_row_stride": 10239},
        {"input_row_stride": True},
        {"input_row_stride": 2**63},
    ],
)
def test_invalid_convolution_contract_is_rejected_without_cuda(override):
    values = dict(
        mode="mixed",
        channels=10240,
        kernel_size=4,
        dilation=1,
        max_tokens=16,
        max_seqs=4,
        max_state_slots=8,
        max_speculative_tokens=3,
    )
    with pytest.raises(ValueError):
        ShortConvQuery(**(values | override))


class _Case:
    def __init__(
        self,
        mode,
        *,
        channels=10240,
        dilation=1,
        input_padding=0,
        speculative_tokens=3,
        kernel_size=4,
        max_tokens=12,
    ):
        device = require_b12x()
        self.caps = short_conv.Caps(
            device=device,
            mode=mode,
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            max_tokens=max_tokens,
            max_seqs=4,
            max_state_slots=6,
            max_speculative_tokens=speculative_tokens,
        )
        self.length = dilation * (kernel_size - 1)
        capacity = self.length + speculative_tokens
        stride = channels * capacity + 32
        self.storage = torch.full(
            (6 * stride,), 17, dtype=torch.bfloat16, device=device
        )
        self.state = self.storage.as_strided(
            (6, channels, capacity), (stride, capacity, 1)
        )
        self.input_storage = torch.randn(
            (max_tokens, channels + input_padding), device=device
        ).to(torch.bfloat16)
        self.x = self.input_storage[:, :channels]
        self.weight = (torch.randn((channels, kernel_size), device=device) * 0.2).to(
            torch.bfloat16
        )
        self.args = dict(
            x=self.x,
            conv_weight=self.weight,
            conv_state=self.state,
            out=torch.full_like(self.x, float("nan")),
            query_start_loc=torch.zeros(5, dtype=torch.int32, device=device),
            state_slot_ids=torch.full((4,), -1, dtype=torch.int64, device=device),
            state_is_fresh=torch.zeros(4, dtype=torch.bool, device=device),
            num_accepted_tokens=torch.ones(4, dtype=torch.int32, device=device),
            num_seqs=torch.zeros(1, dtype=torch.int32, device=device),
            num_tokens=torch.zeros(1, dtype=torch.int32, device=device),
        )
        if mode == "mixed":
            self.args["request_is_prefill"] = torch.zeros(
                4, dtype=torch.bool, device=device
            )

    def batch(self, lengths, slots, fresh, accepted, prefill=None):
        n = len(lengths)
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        for name, values, pad in (
            ("query_start_loc", starts, -12345),
            ("state_slot_ids", slots, 2**40),
            ("state_is_fresh", fresh, False),
            ("num_accepted_tokens", accepted, -12345),
        ):
            dest = self.args[name]
            dest.copy_(
                torch.tensor(
                    values + [pad] * (dest.numel() - len(values)),
                    dtype=dest.dtype,
                    device=dest.device,
                )
            )
        self.args["num_seqs"].fill_(n)
        self.args["num_tokens"].fill_(sum(lengths))
        if prefill is not None:
            self.args["request_is_prefill"].copy_(
                torch.tensor(
                    prefill + [False] * (4 - n),
                    dtype=torch.bool,
                    device=self.x.device,
                )
            )

    def oracle(self):
        expected = torch.zeros_like(self.x)
        state = self.state.clone()
        count = int(self.args["num_seqs"].item())
        for request in range(count):
            slot = int(self.args["state_slot_ids"][request].item())
            start, end = self.args["query_start_loc"][request : request + 2].tolist()
            if slot < 0 or start == end:
                continue
            prefill = self.caps.mode == "prefill" or (
                self.caps.mode == "mixed"
                and bool(self.args["request_is_prefill"][request])
            )
            accepted = int(self.args["num_accepted_tokens"][request].item())
            offset = 0 if prefill else accepted - 1
            history = state[slot, :, offset : offset + self.length].clone()
            if bool(self.args["state_is_fresh"][request]):
                history.zero_()
            sequence = torch.cat((history, self.x[start:end].T), dim=1)
            conv = F.conv1d(
                sequence.unsqueeze(0),
                self.weight.unsqueeze(1),
                groups=self.caps.channels,
                dilation=self.caps.dilation,
            )
            expected[start:end] = F.silu(conv).squeeze(0).T
            committed = end - start if prefill else 1
            state[slot, :, : self.length] = sequence[
                :, committed : committed + self.length
            ]
            state[slot, :, self.length :] = 0
            if not prefill and end - start > 1:
                state[slot, :, self.length : self.length + end - start - 1] = self.x[
                    start + 1 : end
                ].T
        return expected, state


@contextmanager
def _prepared(case):
    plan = short_conv.plan(
        case.caps, invocation=short_conv.invocation_from_tensors(**case.args)
    )
    box = []

    def prepare_call(state):
        scratch = {
            s.name: torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in plan.scratch_specs()
        }
        binding = state.bind(plan=plan, scratch=scratch, **case.args)
        box.append(binding)
        return PreparedCall(run=lambda: state.run(binding), output=binding.out)

    with PreparationSession(
        device=case.x.device, autotune=False, compile_workers=1
    ) as session:
        session.prepare((plan.request(name="short_conv", prepare_call=prepare_call),))
        session.freeze()
        yield box[0]


@pytest.mark.parametrize("max_tokens", [12, 300])
@pytest.mark.parametrize("mode", ["decode", "prefill", "mixed"])
@pytest.mark.parametrize("input_padding", [0, 6144])
def test_gdn_width_raw_history_and_device_metadata_replay(mode, input_padding, max_tokens):
    case = _Case(mode, input_padding=input_padding, max_tokens=max_tokens)
    with _prepared(case) as binding:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = short_conv.run(binding)
        batches = (
            (
                [4, 2, 0],
                [4, 1, -1],
                [False, True, False],
                [3, 1, 0],
                [False, True, False],
            ),
            ([1, 3], [-1, 4], [False, True], [0, 1], [False, False]),
            ([], [], [], [], []),
            ([0, 0, 0], [5, 0, 2], [True, True, True], [1, 1, 1], [True, False, True]),
        )
        for lengths, slots, fresh, accepted, prefill in batches:
            case.x.normal_()
            case.batch(
                lengths, slots, fresh, accepted, prefill if mode == "mixed" else None
            )
            expected, state = case.oracle()
            untouched_padding = case.storage.reshape(6, -1)[:, -32:].clone()
            untouched_input = case.input_storage.clone()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert result.data_ptr() == case.args["out"].data_ptr()
            torch.testing.assert_close(result, expected, rtol=0.008, atol=0.002)
            torch.testing.assert_close(case.state, state, rtol=0, atol=0)
            torch.testing.assert_close(
                case.input_storage, untouched_input, rtol=0, atol=0
            )
            torch.testing.assert_close(
                case.storage.reshape(6, -1)[:, -32:], untouched_padding, rtol=0, atol=0
            )
        graph.reset()


@pytest.mark.parametrize("accepted", [1, 2, 4])
def test_decode_selects_accepted_history_and_retains_raw_candidates(accepted):
    case = _Case("decode", channels=65)
    prior = torch.randn_like(case.state)
    with _prepared(case) as binding:
        case.state.copy_(prior)
        case.batch([4], [2], [False], [accepted])
        expected, state = case.oracle()
        short_conv.run(binding)
        torch.testing.assert_close(binding.out, expected, rtol=0.008, atol=0.002)
        torch.testing.assert_close(case.state, state, rtol=0, atol=0)
        torch.testing.assert_close(case.state[2, :, -3:], case.x[1:4].T, rtol=0, atol=0)


@pytest.mark.parametrize("dilation", [1, 3])
def test_prefill_chunks_match_one_canonical_convolution(dilation):
    case = _Case("prefill", channels=65, dilation=dilation)
    sequence = case.x.clone()
    full_input = F.pad(sequence.T.unsqueeze(0), (case.length, 0))
    expected = (
        F.silu(
            F.conv1d(full_input, case.weight.unsqueeze(1), groups=65, dilation=dilation)
        )
        .squeeze(0)
        .T
    )
    outputs = []
    with _prepared(case) as binding:
        start = 0
        for length in (1, 2, 5, 4):
            case.x[:length].copy_(sequence[start : start + length])
            case.batch([length], [3], [start == 0], [1])
            short_conv.run(binding)
            outputs.append(binding.out[:length].clone())
            start += length
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=0.008, atol=0.002)
        torch.testing.assert_close(
            case.state[3, :, : case.length], sequence[-case.length :].T, rtol=0, atol=0
        )



@pytest.mark.parametrize("dilation", [1, 3])
def test_tiled_prefill_matches_canonical_convolution_across_tiles(dilation):
    """Capacities that tile tokens keep requests crossing tile boundaries exact."""
    case = _Case("prefill", channels=200, dilation=dilation, max_tokens=300)
    with _prepared(case) as binding:
        for lengths, slots, fresh in (
            ([31, 70, 1, 150], [3, 1, -1, 4], [True, False, True, False]),
            ([300], [2], [False]),
            ([33, 2], [5, 0], [False, True]),
        ):
            case.x.normal_()
            case.state.normal_()
            case.batch(lengths, slots, fresh, [1] * len(lengths))
            expected, state = case.oracle()
            short_conv.run(binding)
            live = sum(lengths)
            torch.testing.assert_close(
                binding.out[:live], expected[:live], rtol=0.008, atol=0.002
            )
            torch.testing.assert_close(case.state, state, rtol=0, atol=0)

def test_binding_rejects_input_output_aliases_and_unknown_history_layout():
    case = _Case("mixed", channels=64)
    with _prepared(case) as binding:
        with pytest.raises(ValueError, match="must not overlap"):
            short_conv.bind(
                binding.plan, scratch=binding.scratch, **(case.args | {"out": case.x})
            )
        with pytest.raises(ValueError, match="strides differ"):
            short_conv.bind(
                binding.plan,
                scratch=binding.scratch,
                **(case.args | {"conv_state": case.state.contiguous()}),
            )


@pytest.mark.parametrize("speculative_tokens", [0, 3])
def test_convolution_rounds_bf16_before_silu(speculative_tokens):
    case = _Case("decode", channels=64, speculative_tokens=speculative_tokens)
    case.x.fill_(1)
    case.weight.zero_()
    case.weight[:, 0] = 1
    case.weight[:, 1] = 1 / 256
    with _prepared(case) as binding:
        case.state.fill_(1)
        case.batch([1], [0], [False], [1])
        short_conv.run(binding)
        expected = F.silu(torch.ones_like(binding.out[:1]))
        unrounded = F.silu(torch.full((1, 64), 1 + 1 / 256, device=case.x.device)).to(
            torch.bfloat16
        )
        assert not torch.equal(expected, unrounded)
        torch.testing.assert_close(binding.out[:1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("channels", [65, 6144, 12288])
@pytest.mark.parametrize("kernel_size,dilation", [(2, 1), (4, 1), (4, 3), (7, 2)])
def test_single_token_decode_updates_packed_strided_history(
    channels, kernel_size, dilation
):
    case = _Case(
        "decode",
        channels=channels,
        dilation=dilation,
        input_padding=160,
        speculative_tokens=0,
        kernel_size=kernel_size,
    )
    with _prepared(case) as binding:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            short_conv.run(binding)
        batches = (
            ([1, 0, 1, 1], [4, 3, 1, -1], [False, True, True, False]),
            ([0, 1, 0, 1], [3, 4, 0, 1], [True, False, False, False]),
            ([1], [0], [True]),
            ([0, 0, 0], [5, 0, 2], [True, True, True]),
            ([], [], []),
        )
        for lengths, slots, fresh in batches * 3:
            case.x.normal_()
            case.batch(lengths, slots, fresh, [1] * len(lengths))
            for length, slot, is_fresh in zip(lengths, slots, fresh):
                if length and slot >= 0 and is_fresh:
                    case.state[slot].fill_(float("nan"))
            expected, state = case.oracle()
            padding = case.storage.reshape(6, -1)[:, -32:].clone()
            inputs = case.input_storage.clone()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(binding.out, expected, rtol=0.008, atol=0.002)
            torch.testing.assert_close(case.state, state, rtol=0, atol=0)
            torch.testing.assert_close(case.input_storage, inputs, rtol=0, atol=0)
            torch.testing.assert_close(
                case.storage.reshape(6, -1)[:, -32:], padding, rtol=0, atol=0
            )
        graph.reset()


@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("speculative_tokens", [1, 3, 7])
def test_speculative_decode_matches_the_staged_kernels_bitwise(dilation, speculative_tokens):
    """One-launch speculative decode equals mixed mode's gather/convolve/update with all requests decoding."""
    window = speculative_tokens + 1
    lengths = [window, window - 1, 0, 1][: 4]
    slots, fresh, accepted = [1, 3, 0, 5], [False, False, False, True], [window, 1, 1, 1]
    results = {}
    for mode in ("decode", "mixed"):
        torch.manual_seed(7)
        case = _Case(mode, channels=1000, dilation=dilation, speculative_tokens=speculative_tokens,
                     max_tokens=4 * window)
        prior = torch.randn_like(case.state)
        with _prepared(case) as binding:
            case.state.copy_(prior)
            case.batch(lengths, slots, fresh, accepted, [False] * 4 if mode == "mixed" else None)
            short_conv.run(binding)
            torch.cuda.synchronize()
            results[mode] = (binding.out.clone(), case.state.clone())
    assert torch.equal(results["decode"][0], results["mixed"][0])
    assert torch.equal(results["decode"][1], results["mixed"][1])
