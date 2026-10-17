// Neighbor ring packets, per-origin mailbox ownership, shared QP admission and ACK chains.
module ring {
  const WORLD: int
  const RAILS: int
  const PATH_LENGTH: int
  const DIRECTIONS: int -> int
  const ORIGINS: Set[int]
  const SERIALS: int
  const ADMISSION_LIMIT: int
  const FAILED_RANK: int
  const SERVICE_WHILE_WAITING: bool

  pure val Ranks = 0.to(WORLD - 1)
  pure val Rails = 0.to(RAILS - 1)
  pure val Distance = PATH_LENGTH
  type Packet = (int, int, int) // origin, rail, serial
  type Work = (int, int, int, int, int) // origin, rail, serial, kind: 0 data / 1 ACK, hop
  pure val Packets = tuples(ORIGINS, Rails, 1.to(SERIALS))
  pure val Data = tuples(ORIGINS, Rails, 1.to(SERIALS), Set(0), 1.to(Distance))
  pure val Acks = tuples(ORIGINS, Rails, 1.to(SERIALS), Set(1), 1.to(Distance - 1))
  pure val Works = Data.union(Acks)
  pure val Boxes = tuples(ORIGINS, Rails, 1.to(Distance - 1))
  type State = {
    admitted: Set[Packet],
    acknowledged: Set[Packet],
    delivered: Set[Packet],
    wire: Work -> int,
    ticket: Work -> int,
    mailbox: (int, int, int) -> int,
    pending: Set[Work],
    seen: (int, int, int) -> int,
    corrupt: bool,
    timedOut: bool
  }
  var s: State

  pure def packet(w: Work): Packet = (w._1, w._2, w._3)
  pure def node(origin: int, hop: int): int =
    (origin + DIRECTIONS.get(origin) * hop + WORLD * WORLD) % WORLD
  pure def sender(w: Work): int = node(w._1, if (w._4 == 0) w._5 - 1 else w._5)
  pure def receiver(w: Work): int = node(w._1, if (w._4 == 0) w._5 else w._5 - 1)
  pure def sameQP(a: Work, b: Work): bool =
    sender(a) == sender(b) and receiver(a) == receiver(b) and a._2 == b._2
  pure def data(p: Packet, hop: int): Work = (p._1, p._2, p._3, 0, hop)
  pure def ack(p: Packet, hop: int): Work = (p._1, p._2, p._3, 1, hop)
  pure val QPWorks = Works.mapBy(w => Works.filter(v => sameQP(v, w)))
  def outstanding(w: Work): int = QPWorks.get(w).filter(v =>
    s.wire.get(v) > 0 and s.wire.get(v) < 5).size()
  def healthy(w: Work): bool = sender(w) != FAILED_RANK and receiver(w) != FAILED_RANK
  def orderReady(w: Work, level: int): bool = QPWorks.get(w).forall(v =>
    s.ticket.get(v) == 0 or
    s.ticket.get(v) >= s.ticket.get(w) or s.wire.get(v) >= level)
  def nextTicket(w: Work): int =
    QPWorks.get(w).filter(v => s.ticket.get(v) > 0).size() + 1
  def servicing(rank: int): bool = SERVICE_WHILE_WAITING or
    not(s.admitted.exists(p => p._1 == rank and not(s.acknowledged.contains(p))))

  action init = s' = {
    admitted: Set(), acknowledged: Set(), delivered: Set(),
    wire: Works.mapBy(_ => 0), ticket: Works.mapBy(_ => 0),
    mailbox: Boxes.mapBy(_ => 0), pending: Set(), seen: Boxes.mapBy(_ => 0),
    corrupt: false, timedOut: false
  }
  def canAdmit(p: Packet): bool = all {
    not(s.admitted.contains(p)),
    p._3 == 1 or s.acknowledged.contains((p._1, p._2, p._3 - 1)),
    outstanding(data(p, 1)) < ADMISSION_LIMIT
  }
  action admit(p: Packet): bool = all {
    canAdmit(p),
    s' = { admitted: s.admitted.union(Set(p)),
      wire: s.wire.set(data(p, 1), 1), ticket: s.ticket.set(data(p, 1), nextTicket(data(p, 1))),
      ...s }
  }
  def canForward(w: Work): bool = all {
    w._4 == 0, w._5 > 1, s.wire.get(w) == 0,
    s.wire.get(data(packet(w), w._5 - 1)) >= 4,
    s.seen.get((w._1, w._2, w._5 - 1)) < w._3,
    not(s.pending.contains(w)),
    servicing(sender(w)), outstanding(w) < ADMISSION_LIMIT
  }
  action forward(w: Work): bool = all {
    canForward(w),
    s' = { pending: s.pending.union(Set(w)), wire: s.wire.set(w, 1),
      ticket: s.ticket.set(w, nextTicket(w)), ...s }
  }
  // Payload and descriptor source storage remain owned until downstream ACK.
  action readSource(w: Work): bool = all {
    w._4 == 0, s.wire.get(w) == 1, healthy(w), orderReady(w, 2),
    s' = { wire: s.wire.set(w, 2),
      corrupt: s.corrupt or (if (w._5 == 1)
        s.admitted.exists(p => p._1 == w._1 and p._2 == w._2 and p._3 > w._3)
        else s.mailbox.get((w._1, w._2, w._5 - 1)) != w._3), ...s }
  }
  action deliverPayload(w: Work): bool = all {
    w._4 == 0, s.wire.get(w) == 2, healthy(w), orderReady(w, 3),
    s' = { wire: s.wire.set(w, 3),
      mailbox: (if (w._5 < Distance)
        s.mailbox.set((w._1, w._2, w._5), w._3) else s.mailbox), ...s }
  }
  action publishFlag(w: Work): bool = all {
    w._4 == 0, s.wire.get(w) == 3, healthy(w), orderReady(w, 4),
    s' = { wire: s.wire.set(w, 4),
      delivered: (if (w._5 == Distance) s.delivered.union(Set(packet(w))) else s.delivered), ...s }
  }
  action deliverAck(w: Work): bool = all {
    w._4 == 1, s.wire.get(w) == 1, healthy(w), orderReady(w, 4),
    s' = { wire: s.wire.set(w, 4),
      acknowledged: (if (w._5 == 1) s.acknowledged.union(Set(packet(w))) else s.acknowledged), ...s }
  }
  action pollCompletion(w: Work): bool = all {
    s.wire.get(w) == 4, servicing(sender(w)), orderReady(w, 5),
    s' = { wire: s.wire.set(w, 5), ...s }
  }
  def canAck(w: Work): bool = all {
    w._4 == 1, s.wire.get(w) == 0,
    s.pending.contains(data(packet(w), w._5 + 1)),
    (if (w._5 == Distance - 1) s.wire.get(data(packet(w), Distance)) == 5
      else s.wire.get(ack(packet(w), w._5 + 1)) >= 4),
    servicing(sender(w))
  }
  // ACK writes are inline and do not reserve the data admission credit. The
  // real proxy leaves three quarters of the hardware queue for WQEs and ACKs.
  action sendAck(w: Work): bool = all {
    canAck(w),
    s' = { wire: s.wire.set(w, 1), ticket: s.ticket.set(w, nextTicket(w)),
      pending: s.pending.exclude(Set(data(packet(w), w._5 + 1))),
      seen: s.seen.set((w._1, w._2, w._5), w._3), ...s }
  }
  action usefulStep = any {
    { nondet p = Packets.oneOf(); admit(p) },
    { nondet w = Works.oneOf(); any {
      forward(w), readSource(w), deliverPayload(w), publishFlag(w),
      deliverAck(w), pollCompletion(w), sendAck(w)
    }}
  }
  action poll = s' = s
  action step = any { usefulStep, poll }
  // Fault checks may record a timeout, but a timeout never fills delivered or ACK sets.
  action timeout = all { not(s.timedOut), s' = { timedOut: true, ...s } }
  action faultStep = any { step, timeout }

  val complete = s.acknowledged == Packets and Works.forall(w => s.wire.get(w) == 5)
  val usefulEnabled = Packets.exists(p => canAdmit(p)) or Works.exists(w => any {
    canForward(w), canAck(w),
    w._4 == 0 and s.wire.get(w) == 1 and healthy(w) and orderReady(w, 2),
    w._4 == 0 and s.wire.get(w) == 2 and healthy(w) and orderReady(w, 3),
    w._4 == 0 and s.wire.get(w) == 3 and healthy(w) and orderReady(w, 4),
    w._4 == 1 and s.wire.get(w) == 1 and healthy(w) and orderReady(w, 4),
    s.wire.get(w) == 4 and servicing(sender(w)) and orderReady(w, 5)
  })
  val noProtocolDeadlock = complete or usefulEnabled
  val sourceSafe = not(s.corrupt)
  val ackAfterPayload = s.acknowledged.subseteq(s.delivered)
  val creditsSafe = Packets.forall(p =>
    not(s.admitted.contains(p)) or p._3 == 1 or
      s.acknowledged.contains((p._1, p._2, p._3 - 1)))
  // Each admitted data WR can use three WQEs. Unpolled ACK completions may
  // survive serial reuse, so the finite bound includes every modeled serial.
  val queueBound = Works.forall(w =>
    Works.filter(v => sameQP(v, w) and 0 < s.wire.get(v) and s.wire.get(v) < 5)
      .fold(0, (n, v) => n + (if (v._4 == 0) 3 else 1)) <=
        3 * ADMISSION_LIMIT + ORIGINS.size() * SERIALS)
  val safety = sourceSafe and ackAfterPayload and creditsSafe and queueBound
  val healthySafety = safety and noProtocolDeadlock
  temporal healthyProgress = weakFair(usefulStep, s) implies eventually(complete)
}

module ring_4r1h {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 1 -> 1, 2 -> 1, 3 -> 1), ORIGINS = Set(0, 1, 2, 3), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_4r2h {
  import ring(WORLD = 4, RAILS = 2, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 1 -> 1, 2 -> 1, 3 -> 1), ORIGINS = Set(0, 1, 2, 3), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_reuse {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 1 -> 1), ORIGINS = Set(0, 1), SERIALS = 2,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_ack_chain {
  import ring(WORLD = 6, RAILS = 1, PATH_LENGTH = 3,
    DIRECTIONS = Map(0 -> 1, 3 -> 1), ORIGINS = Set(0, 3), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_without_service {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 1 -> 1, 2 -> 1, 3 -> 1), ORIGINS = Set(0, 1, 2, 3), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = false).*
}
module ring_failed_rank {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1), ORIGINS = Set(0), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = 1, SERVICE_WHILE_WAITING = true).*
}

module ring_mixed_directions {
  import ring(WORLD = 5, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 2 -> -1), ORIGINS = Set(0, 2), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_4r_two_origins {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 1 -> 1), ORIGINS = Set(0, 1), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_reuse_one_origin {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1), ORIGINS = Set(0), SERIALS = 2,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
module ring_two_rails_one_origin {
  import ring(WORLD = 4, RAILS = 2, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1), ORIGINS = Set(0), SERIALS = 1,
    ADMISSION_LIMIT = 1, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true).*
}
