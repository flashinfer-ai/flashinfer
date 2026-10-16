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

## Validation limits

CPU-driven RDMA validation passed on four DGX Sparks through a switch: four
ranks with 240 mixed operations per rank and driver skew, seven ranks with 90,
eight ranks with 120 and skew, and sixteen ranks with 30. Larger worlds placed
multiple native processes on each of the four physical hosts. Every byte
matched, each ring rank connected only its two neighbors, and every rank
forwarded traffic. The ordinary four-rank mesh also passed 60 mixed operations
per rank with zero relays.

Those checks validate transport progress and buffer lifetimes on the switched
fabric. GPU graph replay, floating-point reduction correctness on the ring,
failure injection, actual switchless cabling, and model serving remain to be
qualified. CPU relays add overhead; these tests make no throughput, latency,
or physical ring performance claim.
