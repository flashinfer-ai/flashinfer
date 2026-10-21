// A window instance of the ring model: WINDOW mailbox slots per origin and rail at every relay,
// serials beyond the window reuse slots. One module per file keeps the check's request small.
module ring_window2_ack_chain {
  import ring(WORLD = 6, RAILS = 1, PATH_LENGTH = 3,
    DIRECTIONS = Map(0 -> 1), ORIGINS = Set(0), SERIALS = 3,
    ADMISSION_LIMIT = 2, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true,
    WINDOW = 2, SLOT_REUSE_WAIT = true).* from "./ring"
}
