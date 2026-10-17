# RoCEnante: experimental neighbor rings

The switchless DGX Spark topology is inspired by
[FujitsuPolycom/SparkRing](https://github.com/FujitsuPolycom/sparkring), which
describes directly cabled GB10 clusters and collective forwarding between
non-neighbor hosts. Credit to that project for the topology and design
inspiration. This RoCEnante experiment uses native CPU proxies and RDMA
mailboxes for relaying; it does not implement SparkRing's ConnectX hardware
forwarding or establish equivalent performance.

Both the API and this transport implementation are experimental. The explicit
`ring=True` constructor argument enables it; it is never selected automatically.
Omitting the argument keeps the direct all-peer transport.

## Configuration

Construct `b12x.comm.roce.AllReduce` as usual, with the additional keyword:

```python
runtime = roce.AllReduce(
    exchange_group=control_group,
    device="cuda:0",
    max_size=1 << 20,
    pipe_bytes=8 << 20,
    ring=True,
)
```

The exchange group's rank order defines the cycle `0,1,...,N-1,0`, with
2 through 16 ranks. All ranks must select the same mode and capacities. Only
the two neighbors' queue pairs connect (one for a two-rank cycle). This also
works on a switched fabric, allowing neighbor-only connectivity to be emulated
without changing host routes, firewall rules, or cabling.

On a physical ring, enumerate the active RDMA functions on both cabled ports
using `hca_names` and set the appropriate `gid_index`. The existing route planner
pairs endpoints on the same subnet for each neighbor, with up to two rails per
link. It trusts device order for endpoints without IPv4 GIDs, so verify those
links explicitly. A TCP control network is still needed for setup. A Gloo group
can perform that exchange without requiring an NCCL data transport.

One-shot reductions, two-shot reductions, fixed gathers, and variable-length
gathers share the same native write path. Non-neighbor transfers use the
shorter direction around the ring; equal distances choose increasing ranks.
Relays split each rail into at most 64 KiB packets, with one packet per original
sender per rail in flight. Each sender has private mailbox credit at every
relay. Acknowledgments propagate back only after the downstream transfer has
finished reading its source, preserving staging lifetimes without circular
credit dependencies between senders.

The proxy owns approximately 2.01 MiB of extra registered host memory per rank,
allocated before connection exchange. No relay allocation or Python callback
occurs during replay. Existing captured buffers and kernel layouts are retained.
Proxy ABI 9 extends the connection blob; native consumers must use the matching
C source and call `roce_enable_ring` before exchanging blobs. The returned
`native_description()` records `ring`; private mailbox addresses are exchanged
fresh at startup rather than restored from a CUDA graph.

Collective generations range from 1 through 2,147,483,647. The proxy fails
with `RoCE sequence space exhausted` before dispatching a larger generation;
recreate every rank's runtime to reset the sequence. The bound preserves the
meaning of zero-initialized chunk-ready words and the healthy error sentinel.
It applies to the direct and ring transports without changing proxy ABI 9.

## Runnable checks

The transport validator uses real libibverbs registrations and RDMA writes,
with CPU-produced doorbells and byte-exact reference checks. It needs Python,
a C compiler, libibverbs development files, and at least two active RDMA functions per
rank. It creates no CUDA context. Run one command on each host, changing `R`
to that host's rank and `N` to the total rank count:

```sh
python3 tests/experimental/b12x/comm/validate_roce_ring_native.py \
  --rank R --world-size N --rendezvous RANK_ZERO_HOST:29783 \
  --hca-names roceP2p1s0f0,rocep1s0f0 --gid-index 3 --iterations 120
```

The default source directory is this checkout. Use `--source PATH` to validate
another proxy source directory, `--skew-rank R` to delay one CPU driver between
operations, or `--mesh` to check the ordinary direct transport. The validator
alternates one-shot, two-shot, and variable-length operations, including changing
payload sizes, uneven shards, and empty contributions. Each rank reports routes,
connected-peer count, forwarding count, and completed writes as JSON.

The GPU test is a separate torchrun target. From each host, with its rank `R`:

```sh
torchrun --nnodes=N --nproc-per-node=1 --node-rank=R \
  --master-addr=RANK_ZERO_HOST --master-port=29650 \
  -m pytest -x tests/experimental/b12x/comm/test_roce_ring_gpu.py
```

This uses a Gloo control group, captures alternating two-shot/one-shot reductions,
and checks changed BF16 inputs over repeated graph replay. It skips without a
2..16-rank torchrun launch. Use the existing explicit `hca_names` and `gid_index`
constructor arguments if the machine's automatic discovery is insufficient.

The protocol and generation-limit tests use the same launch, with these pytest
targets:

```sh
torchrun --nnodes=N --nproc-per-node=1 --node-rank=R \
  --master-addr=RANK_ZERO_HOST --master-port=29650 \
  -m pytest -x tests/experimental/b12x/comm/test_roce_protocol_gpu.py \
  tests/experimental/b12x/comm/test_roce_sequence_exhaustion_gpu.py
```

The protocol tests alternate 2, 8, 4, and 16 MiB reductions within one captured
graph, change inputs between replays, mix variable-length gathers with both
reduction paths, and delay individual proxies. Both direct and ring transports
must preserve every operation's payload. The generation-limit tests initialize
stopped proxies through a test-only C wrapper: a two-shot graph completes at
generation 2,147,483,647, then the next replay must report sequence exhaustion
without advancing the epoch or posting another transfer.

## Validation limits

Validation used four physical DGX Sparks through a switch, with two RoCE rails
per host and non-neighbor queue pairs absent in ring mode:

| Check | Configuration | Result |
| --- | --- | --- |
| GPU collective and failure checks | Two ranks, sequence-limited proxy | 110 passed and one capacity-related skip per rank |
| GPU collective and failure checks | Four ranks, proxy and CuTe source at revision `c46a0b5` | 105 collective cases plus four protocol cases passed, with one capacity-related skip per rank |
| CPU-driven RDMA ring | Four ranks, sequence-limited proxy | 120 mixed operations per rank, with driver skew; every byte matched |
| CPU-driven RDMA ring | Six logical ranks on three hosts, sequence-limited proxy | 90 mixed operations per rank, with driver skew; every byte matched |

Revision `c46a0b5` also passed CPU-driven ring checks with four ranks and 240
operations per rank, seven ranks with 90, eight ranks with 120, and sixteen
ranks with 30. Larger worlds used multiple processes on each of the four hosts.
All payloads matched; its four-rank mesh passed 60 operations per rank.

The GPU checks cover changed-input graph replay, BF16 reductions, changing
variable-length counts, temporary proxy pauses, stopped-proxy timeouts, invalid
counts, and fail-stop handling. The ABI 9 proxy and CuTe source at revision
`c46a0b5`, including the two-shot `pipe_done` wait, passed four-rank graph replay.
The sequence-limited proxy passed generation-boundary GPU checks with two ranks
in both direct and ring modes, plus four- and six-rank CPU transport checks.
Every rank in those native ring checks forwarded traffic and connected only its
two neighbors.

A separate four-rank GPU negative control removed only the two-shot
`pipe_done` wait. Its changing-shard ring graph produced corrupt output; the
matching graph passed with the wait present. The wait therefore remains part
of the send-buffer lifetime contract.

Actual switchless cabling, full-model ring serving, and ring performance remain
unqualified. These finite tests do not establish progress under every possible
failure or scheduling pattern. CPU relays add overhead; throughput and latency
depend on the topology and workload.
