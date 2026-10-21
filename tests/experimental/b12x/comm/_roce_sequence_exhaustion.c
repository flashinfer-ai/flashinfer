// Exercise the actual proxy doorbell loop without opening devices or posting RDMA.
#include ROCE_PROXY_SOURCE

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  uint32_t sequence = (uint32_t)strtoull(argv[1], NULL, 0);
  int exhausted = atoi(argv[2]);
  uint32_t control[ROCE_CTRL_BYTES / sizeof(uint32_t)] = {0};
  roce_ctx_t context = {0};
  context.world = 2;
  context.region = (uint8_t*)control;
  context.last_seq = sequence - 1;
  control[CTRL_SEQ] = sequence;
  // An invalid kind stops admitted sequences at dispatch before any NIC access.
  control[CTRL_SLOT_KIND + (sequence & 1)] = UINT32_MAX;
  atomic_init(&context.running, 1);
  atomic_init(&context.failed, 0);
  proxy_main(&context);
  if (!roce_failed(&context) || context.last_seq != sequence - 1 || context.ops_posted != 0)
    return 3;
  const char* error = roce_error(&context);
  if (exhausted) {
    if (!strstr(error, "sequence space exhausted") || !strstr(error, "recreate the transport"))
      return 4;
  } else if (!strstr(error, "unknown op kind")) {
    return 5;
  }
  printf("%u: %s\n", sequence, error);
  return 0;
}
