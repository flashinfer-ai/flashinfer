"""PLE checkpoint multipliers index token lag, independently of window width."""

import pytest
import torch

from b12x.sequence import ple_hash
from b12x.sequence.ple_hash.reference import ple_hash_packed_reference

from ..conftest import require_b12x


def _fixture(device):
    return dict(
        token_ids=torch.tensor([7, 9, 11], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        committed_history=torch.tensor([[99, 99]], dtype=torch.int64, device=device),
        eos_token_id=99,
        multipliers=torch.tensor([3, 5, 7], dtype=torch.int64, device=device),
        prime_sizes=torch.tensor([101, 103], dtype=torch.int64, device=device),
        table_offsets=torch.tensor([0, 101], dtype=torch.int64, device=device),
        heads_per_order=1,
    )


def test_checkpoint_multipliers_index_current_token_then_predecessors():
    # At token 11, the bigram hash is (11*3) XOR (9*5) = 12, and
    # the trigram hash adds XOR (7*7) = 61 before the second head offset.
    expected = torch.tensor([[1, 124], [56, 136], [12, 162]])
    actual = ple_hash_packed_reference(**_fixture("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _lag_oracle(tokens, starts, histories, eos, multipliers, sizes, offsets, heads):
    """Integer-only checkpoint equation with an explicit EOS boundary scan."""
    result = []
    order_max = len(multipliers)
    for request, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        history = list(histories[request]) + list(tokens[start:end])
        for position in range(order_max - 1, len(history)):
            lagged = [history[position]]
            crossed_eos = False
            for lag in range(1, order_max):
                token = eos if crossed_eos else history[position - lag]
                lagged.append(token)
                crossed_eos |= token == eos
            row = []
            mixed = lagged[0] * multipliers[0]
            for order in range(2, order_max + 1):
                mixed ^= lagged[order - 1] * multipliers[order - 1]
                for local_head in range(heads):
                    head = (order - 2) * heads + local_head
                    row.append(mixed % sizes[head] + offsets[head])
            result.append(row)
    return torch.tensor(result, dtype=torch.int64)


@pytest.mark.parametrize("max_order", [2, 3, 4])
def test_checkpoint_hash_crosses_chunk_boundaries_but_not_eos(max_order):
    tokens = [7, 99, 11, 99, 13, 17]
    starts = [0, 3, 6]
    histories = [[19] * (max_order - 1), [23] * (max_order - 1)]
    multipliers = [3, 5, 7, 11][:max_order]
    sizes = [101, 103, 107, 109, 113, 127][: (max_order - 1) * 2]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    actual = ple_hash_packed_reference(
        torch.tensor(tokens),
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(histories),
        eos_token_id=99,
        multipliers=torch.tensor(multipliers),
        prime_sizes=torch.tensor(sizes),
        table_offsets=torch.tensor(offsets),
        heads_per_order=2,
    )
    expected = _lag_oracle(
        tokens, starts, histories, 99, multipliers, sizes, offsets, 2
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_hash_gpu_and_graph_replay_use_lag_order():
    device = require_b12x()
    data = _fixture(device)
    plan = ple_hash.plan(
        ple_hash.Caps(
            device=device,
            max_tokens=3,
            max_seqs=1,
            vocab_size=100,
            eos_token_id=99,
            max_order=3,
            heads_per_order=1,
            dense_layer_ordinal=0,
            base_table_size=101,
        ),
        multipliers=data["multipliers"],
        prime_sizes=data["prime_sizes"],
        table_offsets=data["table_offsets"],
    )
    spec = plan.scratch_specs()[0]
    output = torch.empty(3, 2, dtype=torch.int64, device=device)
    binding = plan.bind(
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=spec.device),
        token_ids=data["token_ids"],
        query_start_loc=data["query_start_loc"],
        committed_history=data["committed_history"],
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([3], dtype=torch.int32, device=device),
        out=output,
    )
    ple_hash.run(binding)
    torch.testing.assert_close(
        output.cpu(), torch.tensor([[1, 124], [56, 136], [12, 162]]), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ple_hash.run(binding)
    for tokens in ([11, 9, 7], [99, 13, 17], [7, 9, 11]):
        data["token_ids"].copy_(torch.tensor(tokens, device=device))
        graph.replay()
        expected = _lag_oracle(
            tokens, [0, 3], [[99, 99]], 99, [3, 5, 7], [101, 103], [0, 101], 1
        )
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
