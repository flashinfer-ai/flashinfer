// Enumerate six canonical transitions, including stuttering polls.
module ring_prefix6 {
  import ring(WORLD = 4, RAILS = 2, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1, 1 -> 1, 2 -> 1, 3 -> 1), ORIGINS = Set(0, 1, 2, 3),
    SERIALS = 1, ADMISSION_LIMIT = 1, FAILED_RANK = -1,
    SERVICE_WHILE_WAITING = true, WINDOW = 1, SLOT_REUSE_WAIT = true) as P from "./ring"
  var transitions: int
  action init = all { P::init, transitions' = 0 }
  action step = any {
    all { transitions < 6, P::step, transitions' = transitions + 1 },
    all { transitions == 6, P::poll, transitions' = transitions }
  }
  val sourceSafe = P::sourceSafe
  val counterBound = 0 <= transitions and transitions <= 6
}
