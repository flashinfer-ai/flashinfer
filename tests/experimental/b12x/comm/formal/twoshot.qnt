// Two-shot source-buffer ownership with independently scheduled GPU, proxy and NIC.
module twoshot {
  const WORLD: int
  const RAILS: int
  const SIZES: List[int]
  const WAIT_RELEASE: bool
  const MAX_ADDRESSES: int
  const OP_COUNT: int

  pure val Ranks = 0.to(WORLD - 1)
  pure val Rails = 0.to(RAILS - 1)
  pure val Ops = 1.to(OP_COUNT)
  pure val Addresses = 0.to(MAX_ADDRESSES - 1)
  assume validSizes = Ops.forall(op => SIZES[op - 1] > 0 and SIZES[op - 1] <= MAX_ADDRESSES)
  assume matchingCount = SIZES.length() == OP_COUNT
  // (generation, phase: 0 reduce-scatter / 1 all-gather, source, destination, rail).
  type Packet = (int, int, int, int, int)
  pure val Packets = tuples(Ops, Set(0, 1), Ranks, Ranks, Rails)
    .filter(p => p._3 != p._4)
  type State = {
    epoch: int -> int,
    gpu: int -> int,
    proxy: int -> int,
    done: int -> int,
    readyRS: int -> int,
    readyAG: int -> int,
    send: (int, int) -> int,
    wire: Packet -> int,
    corrupt: bool
  }
  var s: State

  // One address represents one 16-byte pack. Empty stripes still carry a flag.
  pure def first(op: int, shard: int): int = (SIZES[op - 1] * shard) / WORLD
  pure def inShard(op: int, shard: int, address: int): bool =
    first(op, shard) <= address and address < first(op, shard + 1)
  pure def stripe(p: Packet): Set[int] = {
    val shard = if (p._2 == 0) p._4 else p._3
    val count = first(p._1, shard + 1) - first(p._1, shard)
    val base = count / RAILS
    val extra = count % RAILS
    val lo = first(p._1, shard) + p._5 * base + (if (p._5 < extra) p._5 else extra)
    val n = base + (if (p._5 < extra) 1 else 0)
    Addresses.filter(a => lo <= a and a < lo + n)
  }
  pure def code(op: int, phase: int): int = 2 * op + phase
  def incoming(rank: int, op: int, phase: int): bool =
    Packets.filter(p => p._1 == op and p._2 == phase and p._4 == rank)
      .forall(p => s.wire.get(p) >= 3)
  def posted(rank: int, op: int, phase: int): bool =
    Packets.filter(p => p._1 == op and p._2 == phase and p._3 == rank)
      .forall(p => s.wire.get(p) >= 1)
  def drained(rank: int, op: int): bool =
    Packets.filter(p => p._1 == op and p._3 == rank)
      .forall(p => s.wire.get(p) == 4)

  action init = s' = {
    epoch: Ranks.mapBy(_ => 0), gpu: Ranks.mapBy(_ => 0),
    proxy: Ranks.mapBy(_ => 0), done: Ranks.mapBy(_ => 0),
    readyRS: Ranks.mapBy(_ => 0), readyAG: Ranks.mapBy(_ => 0),
    send: tuples(Ranks, Addresses).mapBy(_ => 0),
    wire: Packets.mapBy(_ => 0), corrupt: false
  }

  def canStage(r: int): bool = s.gpu.get(r) == 0 and s.epoch.get(r) < OP_COUNT
  action stage(r: int): bool = all {
    canStage(r),
    s' = {
      gpu: s.gpu.set(r, 1),
      readyRS: s.readyRS.set(r, s.epoch.get(r) + 1),
      send: tuples(Ranks, Addresses).mapBy(k =>
        if (k._1 == r and k._2 < SIZES[s.epoch.get(r)] and
            not(inShard(s.epoch.get(r) + 1, r, k._2)))
          code(s.epoch.get(r) + 1, 0) else s.send.get(k)),
      ...s
    }
  }
  def canReduce(r: int): bool = s.gpu.get(r) == 1 and incoming(r, s.epoch.get(r) + 1, 0)
  action reduce(r: int): bool = all {
    canReduce(r),
    s' = {
      gpu: s.gpu.set(r, 2),
      readyAG: s.readyAG.set(r, s.epoch.get(r) + 1),
      send: tuples(Ranks, Addresses).mapBy(k =>
        if (k._1 == r and inShard(s.epoch.get(r) + 1, r, k._2))
          code(s.epoch.get(r) + 1, 1) else s.send.get(k)),
      ...s
    }
  }
  def canGather(r: int): bool = s.gpu.get(r) == 2 and incoming(r, s.epoch.get(r) + 1, 1)
  action gather(r: int): bool = all {
    canGather(r),
    s' = { gpu: s.gpu.set(r, 3), ...s }
  }
  def canFinish(r: int): bool = s.gpu.get(r) == 3 and
    (not(WAIT_RELEASE) or s.done.get(r) == s.epoch.get(r) + 1)
  action finish(r: int): bool = all {
    canFinish(r),
    s' = { epoch: s.epoch.set(r, s.epoch.get(r) + 1), gpu: s.gpu.set(r, 0), ...s }
  }
  // A ready word survives GPU completion until the following generation writes
  // that word. Each proxy finishes one generation before inspecting the next.
  def canPost(p: Packet): bool = all {
    s.wire.get(p) == 0,
    s.proxy.get(p._3) + 1 == p._1,
    (if (p._2 == 0) s.readyRS.get(p._3) else s.readyAG.get(p._3)) == p._1,
    p._2 == 0 or posted(p._3, p._1, 0)
  }
  action post(p: Packet): bool = all {
    canPost(p), s' = { wire: s.wire.set(p, 1), ...s }
  }
  // Posting describes an address; the NIC reads its contents later.
  action readPayload(p: Packet): bool = all {
    s.wire.get(p) == 1,
    s' = {
      wire: s.wire.set(p, 2),
      corrupt: s.corrupt or stripe(p).exists(a => s.send.get((p._3, a)) != code(p._1, p._2)),
      ...s
    }
  }
  // Reliable QP ordering publishes a stripe flag only after its payload.
  action publishFlag(p: Packet): bool = all {
    s.wire.get(p) == 2, s' = { wire: s.wire.set(p, 3), ...s }
  }
  // CQ consumption may happen after a receiver has already consumed its flag.
  action completion(p: Packet): bool = all {
    s.wire.get(p) == 3, s' = { wire: s.wire.set(p, 4), ...s }
  }
  def canRelease(r: int): bool = s.proxy.get(r) < OP_COUNT and
    drained(r, s.proxy.get(r) + 1)
  action release(r: int): bool = all {
    canRelease(r),
    s' = {
      done: s.done.set(r, s.proxy.get(r) + 1),
      proxy: s.proxy.set(r, s.proxy.get(r) + 1), ...s
    }
  }

  action gpuStep = {
    nondet r = Ranks.oneOf()
    any { stage(r), reduce(r), gather(r), finish(r) }
  }
  action proxyStep = any {
    { nondet p = Packets.oneOf(); post(p) },
    { nondet r = Ranks.oneOf(); release(r) }
  }
  action nicStep = {
    nondet p = Packets.oneOf()
    any { readPayload(p), publishFlag(p), completion(p) }
  }
  action usefulStep = any { gpuStep, proxyStep, nicStep }
  // Polling can repeat forever; it does not release storage or complete an op.
  action poll = s' = s
  action step = any { usefulStep, poll }

  val complete = Ranks.forall(r => s.epoch.get(r) == OP_COUNT and
    s.proxy.get(r) == OP_COUNT)
  val usefulEnabled = Ranks.exists(r => any {
      canStage(r), canReduce(r), canGather(r), canFinish(r), canRelease(r)
    }) or Packets.exists(p => canPost(p) or (1 <= s.wire.get(p) and s.wire.get(p) < 4))
  val sourceSafe = not(s.corrupt)
  val noProtocolDeadlock = complete or usefulEnabled
  val releaseBeforeEpoch = not(WAIT_RELEASE) or
    Ranks.forall(r => s.epoch.get(r) <= s.done.get(r))
  val safety = sourceSafe and noProtocolDeadlock and releaseBeforeEpoch
  // Every useful transition advances a finite operation/packet state. Weak
  // fairness of the aggregate non-stuttering scheduler therefore suffices.
  temporal healthyProgress = weakFair(usefulStep, s) implies eventually(complete)
}

module twoshot_release_2r1h {
  import twoshot(WORLD = 2, RAILS = 1, SIZES = List(4, 2), WAIT_RELEASE = true, MAX_ADDRESSES = 4, OP_COUNT = 2).*
}
module twoshot_release_2r2h {
  import twoshot(WORLD = 2, RAILS = 2, SIZES = List(4, 2), WAIT_RELEASE = true, MAX_ADDRESSES = 4, OP_COUNT = 2).*
}
module twoshot_release_3r1h {
  import twoshot(WORLD = 3, RAILS = 1, SIZES = List(6, 3), WAIT_RELEASE = true, MAX_ADDRESSES = 6, OP_COUNT = 2).*
}
module twoshot_without_release {
  import twoshot(WORLD = 2, RAILS = 1, SIZES = List(4, 2), WAIT_RELEASE = false, MAX_ADDRESSES = 4, OP_COUNT = 2).*

  // Rank zero advances after receiving rank one's all-gather. Its own
  // all-gather still refers to packs [0,2), and the shorter op stages pack 1.
  run geometryOverwriteTest = init
    .then(stage(0)).then(stage(1))
    .then(post((1, 0, 0, 1, 0))).then(readPayload((1, 0, 0, 1, 0)))
    .then(publishFlag((1, 0, 0, 1, 0)))
    .then(post((1, 0, 1, 0, 0))).then(readPayload((1, 0, 1, 0, 0)))
    .then(publishFlag((1, 0, 1, 0, 0)))
    .then(reduce(0)).then(reduce(1))
    .then(post((1, 1, 1, 0, 0))).then(readPayload((1, 1, 1, 0, 0)))
    .then(publishFlag((1, 1, 1, 0, 0)))
    .then(gather(0)).then(finish(0)).then(stage(0))
    .then(post((1, 1, 0, 1, 0))).then(readPayload((1, 1, 0, 1, 0)))
    .then(all { assert(not(sourceSafe)), s' = s })
}
