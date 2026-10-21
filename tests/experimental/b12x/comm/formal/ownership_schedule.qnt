// An 18-transition legal NIC-delay witness using the same protocol actions.
// TLC checks the unrestricted small instances in twoshot.qnt; this schedule
// gives the SMT backend a compact source-ownership regression check.
module ownership_schedule {
  const WAIT_RELEASE: bool
  import twoshot(WORLD = 2, RAILS = 1, SIZES = List(4, 2),
    MAX_ADDRESSES = 4, OP_COUNT = 2, WAIT_RELEASE = WAIT_RELEASE) as P from "./twoshot"
  var pc: int
  action init = all { P::init, pc' = 0 }
  action scheduled = any {
    all { pc == 0, P::stage(0) },
    all { pc == 1, P::stage(1) },
    all { pc == 2, P::post((1, 0, 0, 1, 0)) },
    all { pc == 3, P::readPayload((1, 0, 0, 1, 0)) },
    all { pc == 4, P::publishFlag((1, 0, 0, 1, 0)) },
    all { pc == 5, P::post((1, 0, 1, 0, 0)) },
    all { pc == 6, P::readPayload((1, 0, 1, 0, 0)) },
    all { pc == 7, P::publishFlag((1, 0, 1, 0, 0)) },
    all { pc == 8, P::reduce(0) },
    all { pc == 9, P::reduce(1) },
    all { pc == 10, P::post((1, 1, 1, 0, 0)) },
    all { pc == 11, P::readPayload((1, 1, 1, 0, 0)) },
    all { pc == 12, P::publishFlag((1, 1, 1, 0, 0)) },
    all { pc == 13, P::gather(0) },
    all { pc == 14, P::finish(0) },
    all { pc == 15, P::stage(0) },
    all { pc == 16, P::post((1, 1, 0, 1, 0)) },
    all { pc == 17, P::readPayload((1, 1, 0, 1, 0)) }
  }
  action step = any {
    all { scheduled, pc' = pc + 1 },
    all { pc == 18 or (WAIT_RELEASE and pc == 14), P::poll, pc' = pc }
  }
  val sourceSafe = P::sourceSafe
}
module ownership_schedule_without_release {
  import ownership_schedule(WAIT_RELEASE = false).*
}
module ownership_schedule_with_release {
  import ownership_schedule(WAIT_RELEASE = true).*
}
