# RoCE source ownership and relay progress models

These experimental Quint models check source-buffer ownership, payload publication,
and finite-work progress for the b12x RoCE transport. They execute on CPUs and do
not import FlashInfer, initialize CUDA, or use an RDMA device. Hardware tests must
also check the CUDA, CPU, and NIC visibility assumptions used by the models.

The model toolchain is Quint 0.33.0 and Apalache 0.62.1. TLC runs through Quint's
TLA+ translation. An Apalache success covers the requested transition bound. A TLC
success covers the reachable state graph of the selected finite configuration.
The six-transition Apalache cases check source reads with three collective
ranks or four ring origins. Ring checks can reach a relay's NIC read, but
cannot complete an acknowledgment chain.
The exhaustive TLC cases check acknowledgment ordering, finite queue bounds,
deadlock freedom, and fair progress in their smaller configurations.

## Source mapping

Paths in this table are relative to the repository root.

| Protocol operation | Model transition | Implementation |
| --- | --- | --- |
| Stage non-owned shards and publish chunk readiness | `twoshot.stage` | `flashinfer/experimental/b12x/comm/roce/_twoshot_cute.py`, kernel section 2 |
| Reduce an owned shard and publish all-gather readiness | `twoshot.reduce` | `_twoshot_cute.py`, kernel section 3 |
| Post chunk source addresses | `twoshot.post` | `flashinfer/experimental/b12x/comm/roce/_roce_proxy.c`, `post_pipe_chunk`, `post_striped` |
| Read source bytes, deliver the ordered stripe flag, consume CQ completion | `twoshot.readPayload`, `publishFlag`, `completion` | `_roce_proxy.c`, RDMA write chain and `drain_cq` |
| Release all two-shot source storage | `twoshot.release` | `_roce_proxy.c`, `run_pipe_op` drains outstanding writes and stores `CTRL_PIPE_DONE`, control word 48 |
| Publish the device epoch after the release wait | `twoshot.finish` | `_twoshot_cute.py`, kernel section 5 |
| Route one non-neighbor packet through private mailboxes | `ring.admit`, `forward` | `_roce_proxy.c`, `ring_next`, `ring_post`, `ring_submit`, `ring_service` |
| Publish payload before mailbox serial or destination flag | `ring.readSource`, `deliverPayload`, `publishFlag` | `_roce_proxy.c`, ordered work requests in `ring_submit` and final-hop forwarding |
| Return acknowledgments after final-hop completion or downstream acknowledgment | `ring.sendAck`, `deliverAck` | `_roce_proxy.c`, `ring_complete`, `ring_service`, `drain_cq` |
| Continue servicing relay traffic while an origin waits | `SERVICE_WHILE_WAITING` | `_roce_proxy.c`, `ring_post` calls `drain_all`, which polls CQs and calls `ring_service` |

`_routes.py:plan_routes` plans routes only for rank-order neighbors when
`ring=True`; the proxy's `roce_connect` connects those QPs and leaves
non-neighbor QPs unconnected. The ring model starts after that route plan and
reliable connections exist.

## Two-shot ownership

`twoshot.qnt` has independently scheduled GPU, proxy, and NIC transitions. Each
operation has one chunk per shard, one source buffer per rank, and one or two
rails. An address represents a 16-byte pack. Source values carry their operation
generation and whether they contain staged input or a reduced shard. Posting an
RDMA write records an address; the NIC reads that address in a later transition.
The receiver can observe a flag before the sender consumes the completion.

The positive configurations use the source-release wait. Removing that wait is
the only protocol mutation in `twoshot_without_release`. Its
`geometryOverwriteTest` is an executable witness:

1. With two ranks and four packs, rank zero's reduced shard occupies packs
   `[0,2)`.
2. Rank zero receives rank one's all-gather and advances its epoch while its own
   all-gather remains unsent or unread by the NIC.
3. A two-pack operation starts. Rank zero stages rank one's shard into pack 1,
   overwriting part of the previous reduced shard.
4. The delayed all-gather reads that overwritten pack and violates `sourceSafe`.

The changing geometry matters: staging skips the locally owned shard, so an
arbitrary overwrite of that same shard at identical geometry would not match the
kernel. The release wait prevents the epoch transition until all local writes,
including the local all-gather, have completed.

`ownership_schedule.qnt` supplies that legal 18-transition schedule to Apalache
using the same actions as the unrestricted model. With the release wait, the
schedule stops at the attempted early epoch publication. Without the wait, the
schedule reaches the overwritten source read. This regression check has a
restricted scheduler; exhaustive interleaving checks use the two-shot TLC
configurations. The unrestricted Apalache mutation case remains available with
`--include-expensive`.

The two-shot model combines payload read and remote payload delivery into one
transition, then separates flag visibility and CQ polling. It permits additional
inter-packet reorderings beyond reliable-QP ordering; payload-before-own-flag
ordering remains mandatory. Positive safety checks therefore include those
additional interleavings. The deterministic overwrite witness uses legal QP
ordering.

## Ring ownership and queue admission

`ring.qnt` represents each origin, rail, serial, direction, and hop explicitly.
Every relay has a private mailbox for each origin and rail. A packet occupies
that mailbox while forwarding or waiting for its downstream acknowledgment. The
origin admits the following serial only after the acknowledgment returns along
the full reverse path. Payload source reads, remote payload delivery, serial or
flag visibility, and CQ polling are separate transitions.

The model orders visibility within each peer/rail QP and shares each QP's
admission counter across origins. Data admission stops at `ADMISSION_LIMIT`;
inline acknowledgment writes use reserved capacity. `queueBound` counts up to
three work requests per admitted data write and one per acknowledgment. Its
finite reserve includes every modeled origin and serial because acknowledgment
completions can remain unpolled after an origin admits another serial. It does
not establish the production queue bound for indefinitely repeated traffic.

Four-rank opposite destinations follow the clockwise tie used by `ring_next`.
`ring_mixed_directions` uses a five-rank ring with two opposing two-hop routes;
data and reverse acknowledgments then contend in both directions. The six-rank
`ring_ack_chain` configuration includes two relay mailboxes per route. These
five- and six-rank configurations execute on a single model-checker process;
they do not require that many physical hosts. Direct neighbor traffic and the
two-rank ring fast path are outside the relay model.

Each origin may have one packet per rail in flight in the model. The native
`ring_post` loop serializes rails and target packets more tightly. Relay service
and CQ polling can run between native waiting-loop iterations. The model permits
additional origin concurrency and abstracts an RDMA payload/header chain as one
source-lifetime unit; it does not model individual header cache lines, provider
queue internals, or recursive C call stacks.

## Progress and failures

`complete` requires successful operation epochs and proxy release in the
two-shot model, or end-to-end acknowledgments and drained CQ work in the ring
model. Polling is a stuttering transition. `noProtocolDeadlock` requires useful
work to remain enabled until that successful terminal state; spinning does not
satisfy it.

`healthyProgress` assumes weak fairness of the aggregate useful scheduler. There
are finitely many operations and packets, and every useful transition advances
their state. This fairness condition rules out endless polling while useful work
is enabled. It does not establish progress under infinite arrivals, retries,
arbitrary process failure, or a scheduler that permanently starves useful work.

`ring_without_service` disables relay and CQ service while each origin waits for
its acknowledgment. `ring_failed_rank` permanently blocks transfers involving a
selected rank. Both are expected to violate `noProtocolDeadlock`. The fault
configuration also permits a timeout record through `faultStep`; a timeout never
populates the delivered or acknowledged sets. Failure detection, safe teardown,
and outstanding-memory lifetime during shutdown require separate models and
native tests.

The models assume one ordered collective stream per runtime, matching
`roce_oneshot.py:_order_stream`. All ranks issue matching collective geometry and
order. Generations are bounded positive integers; device epoch exhaustion,
sequence wrap, and ring serial wrap are outside the checked configurations.
CUDA system-scope fences, coherent host mappings, and reliable ordered RDMA
visibility are platform assumptions.

## Running checks

Install the pinned CLI outside the repository:

```sh
npm install --prefix /tmp/roce-quint @informalsystems/quint@0.33.0
export PATH="/tmp/roce-quint/node_modules/.bin:$PATH"
```

Java 21 or later is required. Quint downloads the selected Apalache distribution
into its user cache. Run one case at a time per host to give each backend an
independent default server port:

```sh
python3 tests/experimental/b12x/comm/formal/run.py list
python3 tests/experimental/b12x/comm/formal/run.py twoshot-tlc-2r1h \
  --output /tmp/roce-formal-results --timeout 300
quint test tests/experimental/b12x/comm/formal/twoshot.qnt \
  --main twoshot_without_release --match geometryOverwriteTest --seed 17
```

`cases.json` records each module, invariant, backend, bound, and expected result.
The runner records the exact command, source hashes, tool version, raw log,
elapsed time, and TLC state counts. An expected mutation counterexample is an
accepted test result; a backend error or time limit is an unsuccessful check.
`tlc.json` caps TLC at a 2 GiB heap and four workers. The runner also supplies
bounded JVM arguments for the Apalache server. Hardware measurements and model
results should be stored outside the repository with the source hashes recorded
by the runner.

The default exhaustive four-rank ring case has adjacent active origins zero and
one, whose own and relayed data share a QP. Larger four-origin, two-rail, and
three-rank configurations also have short six-transition Apalache checks; those
bounds do not reach complete operations or establish behavior after serial
reuse. Deeper bounds and the unrestricted four-origin TLC graph are optional
cases selected with `--include-expensive`. Backend time limits remain
inconclusive even when shorter or more restricted checks pass.

Separate exhaustive single-origin cases cover two consecutive serials through
the same mailbox and independent progress of two rails. Their smaller state
graphs complement the adjacent-origin QP contention and mixed-direction cases.
