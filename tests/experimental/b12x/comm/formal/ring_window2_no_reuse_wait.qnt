// A window instance of the ring model: WINDOW mailbox slots per origin and rail at every relay,
// serials beyond the window reuse slots. One module per file keeps the check's request small.
// Mutation: slot reuse without waiting for the acknowledgment of the packet that last used it.
module ring_window2_no_reuse_wait {
  import ring(WORLD = 4, RAILS = 1, PATH_LENGTH = 2,
    DIRECTIONS = Map(0 -> 1), ORIGINS = Set(0), SERIALS = 3,
    ADMISSION_LIMIT = 2, FAILED_RANK = -1, SERVICE_WHILE_WAITING = true,
    WINDOW = 2, SLOT_REUSE_WAIT = false).* from "./ring"
}
