// RDMA proxy for the b12x RoCE one-shot all-reduce.
//
// One rank owns one pinned host region laid out as:
//
//   recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
//   flag[src][slot][rail] sequence number written on each rail after that rail's
//                         payload stripe
//   send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
//   ctrl             (FLAG_STRIDE)                 {u32 seq, u32 nbytes, u32 error,
//                                                   u32 missing_peer} doorbell; the
//                                                  last two are set by the kernel
//                                                  when a wait times out
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on flag[peer][seq & 1][rail] for every peer and rail.  The
// proxy thread
// below spins on ctrl.seq and, for every peer, stripes the payload across every
// rail.  A rail is one (local HCA, peer HCA) link chosen per peer at connect time:
// on a switched fabric rail h is HCA h on both ends; on a switchless ring each
// peer is reached through the HCAs cabled to it.  Each stripe is followed by its own 4-byte seq
// write on the same reliable QP, so its flag cannot become visible before its payload.  The GPU
// waits for every stripe flag before consuming the receive slot.  Nothing on the receive path
// involves the host.
//
// Messages too large for one-shot latency to matter take the pipelined two-shot
// path instead (ctrl kind[slot] = 1).  The region then also holds
//
//   pipe_send          (pipe_bytes)                 staged input; our reduced shard
//   pipe_recv[src][2]  (world * 2 * shard_cap)       peers' reduce-scatter and
//                                                   all-gather chunks
//   pipe_flag[2][src][chunk][rail]                   seq after each chunk stripe
//
// Shard j covers packs [j * P / world, (j + 1) * P / world) of a P-pack message and
// is split into chunks of chunk_packs.  The kernel publishes ready_rs[c] = seq
// once every block staged chunk c of every peer's shard and ready_ag[c] = seq once
// chunk c of its own shard is reduced; the proxy posts each chunk as its word
// turns, so staging, the wire and the reduction overlap.  The two-shot buffers
// need one copy only: a rank leaves op N after every peer's all-gather of N
// arrived, which every peer sent after reducing all of our N chunks, and a peer
// sends its N + 1 all-gather only after our N + 1 reduce-scatter, which we post
// after our N kernel finished reading.  pipe_send is the exception: our own
// all-gather of N reads our reduced shard from it, and that send is not part of
// what lets our kernel N finish.  So once every write of op N completed, this
// thread sets ctrl pipe_done = seq, and the kernel publishes its epoch only after
// seeing it; otherwise our next two-shot op could stage over the shard while it
// was still to be sent, and peers would gather that input as our reduced values.
//
// A variable-length all-gather (ctrl kind[slot] = 2) streams each rank's contribution, whose size
// the kernel reads from device memory, through
//
//   varlen_recv[slot][src]  (2 * world * varlen_bytes)  src's contribution of the slot's op;
//                                                       our own row is the posting source
//   varlen_flag[slot][src][chunk][rail]                 seq after each chunk stripe
//
// The kernel stages chunk c of its contribution into its own row and publishes
// ready_vl[c] = seq; the proxy posts the chunk to the same row of every peer.  Ops alternate
// slots: a kernel finishes on its peers' rows alone, so our op N + 1 may stage (and overwrite
// ready_vl) before this thread posted our op N.  Its rows and flags are the other slot's, a
// ready word of N + 1 also releases chunk c of N, and no peer can begin op N + 2, which would
// reuse N's slot, before receiving our N.
//
// This file is compiled by b12x.comm.roce._proxy at first use with the host
// gcc and libibverbs; it must stay plain C with no CUDA dependency.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 4
#define ROCE_MAX_RAILS 2
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 512
#define ROCE_ABI_VERSION 10
#define ROCE_MAX_CHUNKS 64
#define ROCE_VARLEN_MAX_CHUNKS 512
#define ROCE_CTRL_BYTES 4096
// ctrl words (u32): the one-shot record, then per-slot op descriptors, the
// two-shot chunk-ready words and the variable-length all-gather's.
#define CTRL_SEQ 0
#define CTRL_ERROR 2
#define CTRL_SLOT_NBYTES 4
#define CTRL_SLOT_KIND 8
#define CTRL_SLOT_CHUNK_PACKS 10
#define CTRL_SLOT_CHUNKS 12
#define CTRL_PIPE_DONE 48
#define CTRL_READY_RS 64
#define CTRL_READY_AG 128
#define CTRL_READY_VL 256
#define KIND_ONESHOT 0
#define KIND_TWOSHOT 1
#define KIND_VARLEN 2
// Model graphs leave sub-millisecond gaps between collectives.  Keep the
// proxy hot across those gaps; sleeping there adds one scheduler wakeup to
// every collective on the graph's critical path.
#define ROCE_IDLE_SPINS 20000000

// Experimental neighbor transport in rank order. Only the two cable-neighbor QPs
// connect. Non-neighbor writes cross CPU-driven, bounded RDMA mailboxes.
// No GPU program or captured address depends on these startup-owned buffers.
//
// Every original sender owns RING_WINDOW mailbox slots per rail at every rank,
// so up to RING_WINDOW packets per sender and rail are in flight at once.
// The startup region, sized for the world, holds per (origin, rail, slot):
// mailboxes, then 8-byte acknowledgment words, then outgoing descriptors.
#define RING_CHUNK_BYTES 65536
#define RING_WINDOW 8
#define RING_COMPLETION (UINT64_C(1) << 63)
typedef struct {
  uint64_t serial, target, offset, flag_offset, flag_value, nbytes, previous;
  uint8_t padding[72];
} ring_descriptor_t;
typedef struct {
  ring_descriptor_t descriptor;
  uint8_t data[RING_CHUNK_BYTES];
} ring_mailbox_t;

static size_t ring_slot(int origin, int rail, int slot) {
  return ((size_t)origin * ROCE_MAX_RAILS + (size_t)rail) * RING_WINDOW + (size_t)slot;
}
static size_t ring_slots(int world) { return (size_t)world * ROCE_MAX_RAILS * RING_WINDOW; }
static size_t ring_inbox_offset(size_t i) { return i * sizeof(ring_mailbox_t); }
static size_t ring_ack_offset(int world, size_t i) {
  return ring_slots(world) * sizeof(ring_mailbox_t) + i * sizeof(uint64_t);
}
static size_t ring_outgoing_offset(int world, size_t i) {
  return ring_slots(world) * (sizeof(ring_mailbox_t) + sizeof(uint64_t)) +
         i * sizeof(ring_descriptor_t);
}
static size_t ring_bytes(int world) {
  return ring_slots(world) *
         (sizeof(ring_mailbox_t) + sizeof(uint64_t) + sizeof(ring_descriptor_t));
}

typedef struct {
  uint64_t region_addr;
  uint64_t ring_addr;
  uint32_t ring_rkey[ROCE_MAX_HCAS];
  uint32_t rkey[ROCE_MAX_HCAS];
  uint16_t lid[ROCE_MAX_HCAS];
  uint8_t gid[ROCE_MAX_HCAS][16];
  uint32_t mtu[ROCE_MAX_HCAS];
  uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
} roce_blob_t;

typedef struct {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_mr* mr;
  struct ibv_cq* cq;
  struct ibv_qp* qp[ROCE_MAX_PEERS];
  uint32_t outstanding[ROCE_MAX_PEERS];
  uint64_t writes_completed;
  uint64_t bytes_posted;
  union ibv_gid gid;
  uint16_t lid;
  enum ibv_mtu mtu;
} roce_hca_t;

typedef struct {
  int world;
  int rank;
  int n_hca;
  int n_rail;
  int gid_index;
  int traffic_class;
  roce_hca_t hca[ROCE_MAX_HCAS];
  uint8_t* region;
  size_t region_bytes;
  size_t slot_bytes;
  size_t recv_off;
  size_t flag_off;
  size_t send_off;
  size_t ctrl_off;
  size_t pipe_bytes;
  size_t pipe_send_off;
  size_t pipe_recv_off;
  size_t pipe_flag_off;
  size_t shard_cap;
  size_t varlen_bytes;
  size_t varlen_recv_off;
  size_t varlen_flag_off;
  int started;
  uint64_t peer_addr[ROCE_MAX_PEERS];
  // Per peer and rail: the local HCA that carries it and the peer's rkey on the
  // remote HCA at the other end of that link.
  int rail_hca[ROCE_MAX_PEERS][ROCE_MAX_RAILS];
  uint32_t peer_rkey[ROCE_MAX_RAILS][ROCE_MAX_PEERS];
  pthread_t thread;
  atomic_int running;
  atomic_int failed;
  uint32_t last_seq;
  uint64_t ops_posted;
  uint64_t writes_completed;
  uint8_t* ring;
  struct ibv_mr* ring_mr[ROCE_MAX_HCAS];
  uint64_t ring_addr[ROCE_MAX_PEERS];
  uint32_t ring_rkey[ROCE_MAX_RAILS][ROCE_MAX_PEERS];
  uint64_t ring_serial[ROCE_MAX_RAILS];
  uint64_t ring_seen[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_WINDOW];
  int ring_pending[ROCE_MAX_PEERS][ROCE_MAX_RAILS][RING_WINDOW];
  uint64_t ring_forwarded;
  char err[512];
} roce_ctx_t;

static int drain_all(roce_ctx_t* c);
static int ring_complete(roce_ctx_t* c, int src, int rail, int slot);
static ring_mailbox_t* ring_inbox(const roce_ctx_t* c, int origin, int rail, int slot) {
  return (ring_mailbox_t*)(c->ring + ring_inbox_offset(ring_slot(origin, rail, slot)));
}
static uint64_t* ring_ack(const roce_ctx_t* c, int origin, int rail, int slot) {
  return (uint64_t*)(c->ring + ring_ack_offset(c->world, ring_slot(origin, rail, slot)));
}
static ring_descriptor_t* ring_outgoing(const roce_ctx_t* c, int origin, int rail, int slot) {
  return (ring_descriptor_t*)(c->ring +
                              ring_outgoing_offset(c->world, ring_slot(origin, rail, slot)));
}
static int ring_next(const roce_ctx_t* c, int target) {
  int clockwise = (target - c->rank + c->world) % c->world;
  int counter = (c->rank - target + c->world) % c->world;
  return clockwise <= counter ? (c->rank + 1) % c->world : (c->rank + c->world - 1) % c->world;
}
static int ring_relayed(const roce_ctx_t* c, int peer) {
  return c->ring != NULL && peer != c->rank && peer != (c->rank + 1) % c->world &&
         peer != (c->rank + c->world - 1) % c->world;
}

void roce_destroy(roce_ctx_t* c);

static void set_err(roce_ctx_t* c, const char* what, int e) {
  snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

int roce_layout(int world, uint64_t slot_bytes, uint64_t pipe_bytes, uint64_t varlen_bytes,
                uint64_t* out) {
  // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots,
  //        pipe_send_off, pipe_recv_off, pipe_flag_off, shard_cap, max_chunks, pipe_bytes,
  //        varlen_recv_off, varlen_flag_off, varlen_bytes, varlen_max_chunks}
  if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0 ||
      (pipe_bytes % 4096) != 0 || (varlen_bytes % 4096) != 0 ||
      varlen_bytes > ((uint64_t)1 << 32)) {
    return -1;
  }
  // Reject a layout whose arithmetic would wrap; the caller sizes slots from
  // configuration, so a wrapped region must fail here rather than at the NIC.
  uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, pipe_send_off, pipe_recv_off,
      pipe_flag_off, pipe_recv_bytes, total;
  if (slot_bytes > ((uint64_t)1 << 40) || pipe_bytes > ((uint64_t)1 << 32) ||
      __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
      __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS * ROCE_MAX_RAILS,
                             (uint64_t)ROCE_FLAG_STRIDE, &flag_bytes) ||
      __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
      __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
      __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
      __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_CTRL_BYTES, &pipe_send_off)) {
    return -1;
  }
  // A shard holds at most ceil(packs / world) packs.
  uint64_t shard_cap = (pipe_bytes / 16 + (uint64_t)world - 1) / (uint64_t)world * 16;
  shard_cap = (shard_cap + 4095) / 4096 * 4096;
  pipe_recv_off = pipe_send_off + pipe_bytes;
  pipe_recv_bytes = (uint64_t)world * 2 * shard_cap;
  pipe_flag_off = pipe_recv_off + pipe_recv_bytes;
  uint64_t pipe_flag_bytes =
      pipe_bytes == 0 ? 0
                      : (uint64_t)2 * world * ROCE_MAX_CHUNKS * ROCE_MAX_RAILS * ROCE_FLAG_STRIDE;
  // The variable-length area follows; without one the region is unchanged.
  uint64_t varlen_recv_off = pipe_flag_off + pipe_flag_bytes;
  uint64_t varlen_flag_off = varlen_recv_off + (uint64_t)ROCE_SLOTS * world * varlen_bytes;
  uint64_t varlen_flag_bytes = varlen_bytes == 0
                                   ? 0
                                   : (uint64_t)ROCE_SLOTS * world * ROCE_VARLEN_MAX_CHUNKS *
                                         ROCE_MAX_RAILS * ROCE_FLAG_STRIDE;
  total = varlen_flag_off + varlen_flag_bytes;
  uint64_t recv_off = 0;
  uint64_t flag_off = recv_off + recv_bytes;
  out[0] = recv_off;
  out[1] = flag_off;
  out[2] = send_off;
  out[3] = ctrl_off;
  out[4] = total;
  out[5] = ROCE_FLAG_STRIDE;
  out[6] = ROCE_SLOTS;
  out[7] = pipe_send_off;
  out[8] = pipe_recv_off;
  out[9] = pipe_flag_off;
  out[10] = shard_cap;
  out[11] = ROCE_MAX_CHUNKS;
  out[12] = pipe_bytes;
  out[13] = varlen_recv_off;
  out[14] = varlen_flag_off;
  out[15] = varlen_bytes;
  out[16] = ROCE_VARLEN_MAX_CHUNKS;
  return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

static int open_hca(roce_ctx_t* c, int h, const char* name) {
  int num = 0;
  struct ibv_device** list = ibv_get_device_list(&num);
  if (list == NULL) {
    set_err(c, "ibv_get_device_list", errno);
    return -1;
  }
  struct ibv_device* dev = NULL;
  for (int i = 0; i < num; i++) {
    if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
      dev = list[i];
      break;
    }
  }
  if (dev == NULL) {
    ibv_free_device_list(list);
    snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
    return -1;
  }
  roce_hca_t* hca = &c->hca[h];
  hca->ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (hca->ctx == NULL) {
    set_err(c, "ibv_open_device", errno);
    return -1;
  }
  struct ibv_port_attr port;
  if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
    set_err(c, "ibv_query_port", errno);
    return -1;
  }
  if (port.state != IBV_PORT_ACTIVE) {
    snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
    return -1;
  }
  hca->lid = port.lid;
  hca->mtu = port.active_mtu;
  if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index, &hca->gid) != 0) {
    set_err(c, "ibv_query_gid", errno);
    return -1;
  }
  hca->pd = ibv_alloc_pd(hca->ctx);
  if (hca->pd == NULL) {
    set_err(c, "ibv_alloc_pd", errno);
    return -1;
  }
  hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (hca->mr == NULL) {
    set_err(c, "ibv_reg_mr(pinned region)", errno);
    return -1;
  }
  hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS, NULL, NULL, 0);
  if (hca->cq == NULL) {
    set_err(c, "ibv_create_cq", errno);
    return -1;
  }
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = hca->cq;
    attr.recv_cq = hca->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = ROCE_SEND_DEPTH;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 16;
    hca->qp[p] = ibv_create_qp(hca->pd, &attr);
    if (hca->qp[p] == NULL) {
      set_err(c, "ibv_create_qp", errno);
      return -1;
    }
    struct ibv_qp_attr init;
    memset(&init, 0, sizeof(init));
    init.qp_state = IBV_QPS_INIT;
    init.pkey_index = 0;
    init.port_num = ROCE_PORT;
    init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    int rc = ibv_modify_qp(hca->qp[p], &init,
                           IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (rc != 0) {
      set_err(c, "ibv_modify_qp(INIT)", rc);
      return -1;
    }
  }
  return 0;
}

roce_ctx_t* roce_create(int world, int rank, const char* const* hca_names, int n_hca, int gid_index,
                        int traffic_class, void* region, uint64_t region_bytes, uint64_t slot_bytes,
                        uint64_t pipe_bytes, uint64_t varlen_bytes, char* err, uint64_t err_len) {
  uint64_t layout[17];
  if (roce_layout(world, slot_bytes, pipe_bytes, varlen_bytes, layout) != 0 ||
      layout[4] > region_bytes || rank < 0 || rank >= world || n_hca < 1 || n_hca > ROCE_MAX_HCAS ||
      traffic_class < 0 || traffic_class > 255) {
    snprintf(err, err_len, "invalid roce runtime geometry");
    return NULL;
  }
  roce_ctx_t* c = calloc(1, sizeof(*c));
  if (c == NULL) {
    snprintf(err, err_len, "out of memory");
    return NULL;
  }
  c->world = world;
  c->rank = rank;
  c->n_hca = n_hca;
  c->gid_index = gid_index;
  c->traffic_class = traffic_class;
  c->region = region;
  c->region_bytes = region_bytes;
  c->slot_bytes = slot_bytes;
  c->recv_off = layout[0];
  c->flag_off = layout[1];
  c->send_off = layout[2];
  c->ctrl_off = layout[3];
  c->pipe_send_off = layout[7];
  c->pipe_recv_off = layout[8];
  c->pipe_flag_off = layout[9];
  c->shard_cap = layout[10];
  c->pipe_bytes = layout[12];
  c->varlen_recv_off = layout[13];
  c->varlen_flag_off = layout[14];
  c->varlen_bytes = layout[15];
  for (int h = 0; h < n_hca; h++) {
    if (open_hca(c, h, hca_names[h]) != 0) {
      snprintf(err, err_len, "%s", c->err);
      roce_destroy(c);
      return NULL;
    }
  }
  return c;
}

// Enable before exchanging blobs; startup allocation, never during replay.
int roce_enable_ring(roce_ctx_t* c) {
  if ((c->world < 2 || c->world > ROCE_MAX_PEERS) || c->started || c->ring != NULL) {
    snprintf(c->err, sizeof(c->err), "experimental ring requires 2..16 ranks before connect");
    return -1;
  }
  void* memory = NULL;
  int rc = posix_memalign(&memory, 4096, ring_bytes(c->world));
  if (rc != 0) {
    set_err(c, "ring mailbox allocation", rc);
    return -1;
  }
  c->ring = memory;
  memset(c->ring, 0, ring_bytes(c->world));
  for (int h = 0; h < c->n_hca; h++) {
    c->ring_mr[h] = ibv_reg_mr(c->hca[h].pd, c->ring, ring_bytes(c->world),
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (c->ring_mr[h] == NULL) {
      set_err(c, "register ring mailboxes", errno);
      return -1;
    }
  }
  return 0;
}

int roce_local_blob(roce_ctx_t* c, void* out, uint64_t out_len) {
  if (out_len < sizeof(roce_blob_t)) {
    return -1;
  }
  roce_blob_t blob;
  memset(&blob, 0, sizeof(blob));
  blob.region_addr = (uint64_t)(uintptr_t)c->region;
  blob.ring_addr = (uint64_t)(uintptr_t)c->ring;
  for (int h = 0; h < c->n_hca; h++) {
    blob.rkey[h] = c->hca[h].mr->rkey;
    blob.ring_rkey[h] = c->ring == NULL ? 0 : c->ring_mr[h]->rkey;
    blob.lid[h] = c->hca[h].lid;
    blob.mtu[h] = (uint32_t)c->hca[h].mtu;
    memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
    for (int p = 0; p < c->world; p++) {
      blob.qp_num[h][p] = (p == c->rank) ? 0 : c->hca[h].qp[p]->qp_num;
    }
  }
  memcpy(out, &blob, sizeof(blob));
  return 0;
}

// Move local HCA h's queue pair for rank p to RTS, connected to the peer's HCA r.
static int connect_qp(roce_ctx_t* c, int h, int r, int p, const roce_blob_t* peer) {
  roce_hca_t* hca = &c->hca[h];
  struct ibv_qp_attr rtr;
  memset(&rtr, 0, sizeof(rtr));
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu =
      (enum ibv_mtu)(peer->mtu[r] < (uint32_t)hca->mtu ? peer->mtu[r] : (uint32_t)hca->mtu);
  rtr.dest_qp_num = peer->qp_num[r][c->rank];
  rtr.rq_psn = 0;
  rtr.max_dest_rd_atomic = 1;
  rtr.min_rnr_timer = 12;
  rtr.ah_attr.is_global = 1;
  rtr.ah_attr.dlid = peer->lid[r];
  rtr.ah_attr.sl = 0;
  rtr.ah_attr.src_path_bits = 0;
  rtr.ah_attr.port_num = ROCE_PORT;
  memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[r], 16);
  rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
  rtr.ah_attr.grh.hop_limit = 64;
  rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
  rtr.ah_attr.grh.flow_label = 0;
  int rc = ibv_modify_qp(hca->qp[p], &rtr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (rc != 0) {
    set_err(c, "ibv_modify_qp(RTR)", rc);
    return -1;
  }
  struct ibv_qp_attr rts;
  memset(&rts, 0, sizeof(rts));
  rts.qp_state = IBV_QPS_RTS;
  rts.timeout = 14;
  rts.retry_cnt = 7;
  rts.rnr_retry = 7;
  rts.sq_psn = 0;
  rts.max_rd_atomic = 1;
  rc = ibv_modify_qp(hca->qp[p], &rts,
                     IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                         IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (rc != 0) {
    set_err(c, "ibv_modify_qp(RTS)", rc);
    return -1;
  }
  return 0;
}

// routes holds world * n_rail * 2 ints: for peer p and rail k, the local HCA
// index then the peer's HCA index of the link that carries the rail (the
// entries for this rank are ignored).  Both ends must pass mirrored routes.
int roce_connect(roce_ctx_t* c, const void* blobs, uint64_t blobs_len, const int32_t* routes,
                 int n_rail) {
  if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
    snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
    return -1;
  }
  if (n_rail < 1 || n_rail > ROCE_MAX_RAILS) {
    snprintf(c->err, sizeof(c->err), "invalid rail count %d", n_rail);
    return -1;
  }
  const roce_blob_t* all = (const roce_blob_t*)blobs;
  for (int p = 0; p < c->world; p++) {
    if ((all[p].ring_addr != 0) != (c->ring != NULL)) {
      snprintf(c->err, sizeof(c->err), "ranks disagree on experimental ring transport");
      return -1;
    }
    c->peer_addr[p] = all[p].region_addr;
    c->ring_addr[p] = all[p].ring_addr;
  }
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank || ring_relayed(c, p)) {
      continue;
    }
    for (int k = 0; k < n_rail; k++) {
      int h = routes[(p * n_rail + k) * 2];
      int r = routes[(p * n_rail + k) * 2 + 1];
      if (h < 0 || h >= c->n_hca || r < 0 || r >= ROCE_MAX_HCAS) {
        snprintf(c->err, sizeof(c->err), "invalid route to rank %d rail %d: %d -> %d", p, k, h, r);
        return -1;
      }
      for (int j = 0; j < k; j++) {
        if (c->rail_hca[p][j] == h) {
          snprintf(c->err, sizeof(c->err), "rank %d rails %d and %d share HCA %d", p, j, k, h);
          return -1;
        }
      }
      c->rail_hca[p][k] = h;
    }
  }
  c->n_rail = n_rail;
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank || ring_relayed(c, p)) {
      continue;
    }
    c->peer_addr[p] = all[p].region_addr;
    for (int k = 0; k < n_rail; k++) {
      int h = c->rail_hca[p][k];
      int r = routes[(p * n_rail + k) * 2 + 1];
      c->peer_rkey[k][p] = all[p].rkey[r];
      c->ring_rkey[k][p] = all[p].ring_rkey[r];
      if (connect_qp(c, h, r, p, &all[p]) != 0) {
        return -1;
      }
    }
  }
  return 0;
}

static int drain_cq(roce_ctx_t* c, int h) {
  struct ibv_wc wc[32];
  int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
  if (n < 0) {
    set_err(c, "ibv_poll_cq", errno);
    return -1;
  }
  for (int i = 0; i < n; i++) {
    int relayed = (wc[i].wr_id & RING_COMPLETION) != 0;
    int src = (int)((wc[i].wr_id >> 16) & 255);
    int slot = (int)((wc[i].wr_id >> 8) & 255);
    int rail = (int)(wc[i].wr_id & 255);
    int peer = relayed ? (int)ring_inbox(c, src, rail, slot)->descriptor.target : (int)wc[i].wr_id;
    if (wc[i].status != IBV_WC_SUCCESS) {
      snprintf(c->err, sizeof(c->err), "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
               (unsigned)peer, ibv_wc_status_str(wc[i].status), wc[i].vendor_err, c->last_seq);
      return -1;
    }
    c->hca[h].outstanding[peer] -= 1;
    c->hca[h].writes_completed += 1;
    c->writes_completed += 1;
    if (relayed && ring_complete(c, src, rail, slot) != 0) return -1;
  }
  return 0;
}

// Wait until HCA h can take one more stripe to peer p. Direct stripes need two
// requests and relay packets need three. A quarter of the queue leaves room
// for provider overhead and per-origin acks posted while draining completions.
static int reserve(roce_ctx_t* c, int h, int p) {
  roce_hca_t* hca = &c->hca[h];
  while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {
    if (drain_all(c) != 0) {
      return -1;
    }
    // A peer that stopped acknowledging keeps the QP retrying for
    // a long time; honour a stop request instead of blocking
    // roce_stop (and so teardown) behind it.
    if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
      snprintf(c->err, sizeof(c->err), "RoCE proxy stopped with %u writes outstanding to rank %d",
               hca->outstanding[p], p);
      return -1;
    }
  }
  return 0;
}

// Each original sender has RING_WINDOW private mailbox slots per rail at every
// relay. It reuses a slot only after the end-to-end ack of the packet that last
// used it returns. Different senders never hold each other's mailbox credit, so
// a cycle of active senders cannot deadlock. Credit waits still progress
// incoming relay work and shared-QP completions; ACK posting relies on the
// queue slack left by reserve. A packet's remaining shortest-path distance
// decreases at every hop. Its header and payload remain live until the
// downstream ack.
static int ring_complete(roce_ctx_t* c, int origin, int rail, int slot) {
  ring_descriptor_t* d = &ring_inbox(c, origin, rail, slot)->descriptor;
  int previous = (int)d->previous;
  int h = c->rail_hca[previous][rail];
  uint64_t serial = d->serial;
  struct ibv_sge sge = {.addr = (uintptr_t)&serial, .length = 8};
  struct ibv_send_wr wr = {0}, *bad = NULL;
  wr.wr_id = (uint64_t)previous;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr =
      c->ring_addr[previous] + ring_ack_offset(c->world, ring_slot(origin, rail, slot));
  wr.wr.rdma.rkey = c->ring_rkey[rail][previous];
  int rc = ibv_post_send(c->hca[h].qp[previous], &wr, &bad);
  if (rc != 0) {
    set_err(c, "ring ack", rc);
    return -1;
  }
  c->hca[h].outstanding[previous]++;
  c->ring_seen[origin][rail][slot] = serial;
  c->ring_pending[origin][rail][slot] = 0;
  c->ring_forwarded++;
  return 0;
}

// The caller reserves queue capacity; this function never progresses another
// packet recursively. In particular a forwarding mailbox is marked pending
// before any subsequent progress call can inspect it again.
static int ring_submit(roce_ctx_t* c, int origin, int rail, int slot, int next,
                       const uint8_t* payload, uint32_t lkey, const ring_descriptor_t* descriptor) {
  int h = c->rail_hca[next][rail];
  ring_descriptor_t* d = ring_outgoing(c, origin, rail, slot);
  *d = *descriptor;
  d->previous = (uint64_t)c->rank;
  uint64_t serial = d->serial;
  uint64_t mailbox = c->ring_addr[next] + ring_inbox_offset(ring_slot(origin, rail, slot));
  struct ibv_sge data = {.addr = (uintptr_t)payload, .length = (uint32_t)d->nbytes, .lkey = lkey};
  struct ibv_sge header = {
      .addr = (uintptr_t)d + 8, .length = sizeof(*d) - 8, .lkey = c->ring_mr[h]->lkey};
  struct ibv_sge bell = {.addr = (uintptr_t)&serial, .length = 8};
  struct ibv_send_wr a = {0}, b = {0}, z = {0}, *bad = NULL;
  a.wr_id = b.wr_id = z.wr_id = (uint64_t)next;
  a.sg_list = &data;
  a.num_sge = 1;
  a.opcode = IBV_WR_RDMA_WRITE;
  a.wr.rdma.remote_addr = mailbox + offsetof(ring_mailbox_t, data);
  a.wr.rdma.rkey = c->ring_rkey[rail][next];
  a.next = &b;
  b.sg_list = &header;
  b.num_sge = 1;
  b.opcode = IBV_WR_RDMA_WRITE;
  b.wr.rdma.remote_addr = mailbox + 8;
  b.wr.rdma.rkey = c->ring_rkey[rail][next];
  b.next = &z;
  z.sg_list = &bell;
  z.num_sge = 1;
  z.opcode = IBV_WR_RDMA_WRITE;
  z.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  z.wr.rdma.remote_addr = mailbox;
  z.wr.rdma.rkey = c->ring_rkey[rail][next];
  int rc = ibv_post_send(c->hca[h].qp[next], d->nbytes ? &a : &b, &bad);
  if (rc != 0) {
    set_err(c, "ring submit", rc);
    return -1;
  }
  c->hca[h].outstanding[next]++;
  c->hca[h].bytes_posted += d->nbytes;
  return 0;
}

static int ring_distance(int a, int b, int world) {
  int cw = (b - a + world) % world, ccw = (a - b + world) % world;
  return cw < ccw ? cw : ccw;
}

static int ring_service_slot(roce_ctx_t* c, int origin, int rail, int slot) {
  ring_mailbox_t* box = ring_inbox(c, origin, rail, slot);
  ring_descriptor_t* d = &box->descriptor;
  uint64_t serial = __atomic_load_n(&d->serial, __ATOMIC_ACQUIRE);
  int* pending = &c->ring_pending[origin][rail][slot];
  if (*pending == 2) {
    if (__atomic_load_n(ring_ack(c, origin, rail, slot), __ATOMIC_ACQUIRE) == serial &&
        ring_complete(c, origin, rail, slot) != 0)
      return -1;
    return 0;
  }
  uint64_t seen = c->ring_seen[origin][rail][slot];
  if (serial == seen || *pending) return 0;
  int target = (int)d->target, previous = (int)d->previous;
  if (serial < seen || (serial - 1) % RING_WINDOW != (uint64_t)slot ||
      d->target >= (uint64_t)c->world || d->previous >= (uint64_t)c->world || previous == c->rank ||
      ring_relayed(c, previous) || target == c->rank ||
      ring_distance(previous, target, c->world) != ring_distance(c->rank, target, c->world) + 1 ||
      d->nbytes > RING_CHUNK_BYTES || d->offset > c->region_bytes ||
      d->nbytes > c->region_bytes - d->offset ||
      (d->flag_offset != UINT64_MAX &&
       (d->flag_offset > c->region_bytes - 4 || d->flag_value > UINT32_MAX))) {
    snprintf(c->err, sizeof(c->err), "invalid ring mailbox from origin %d rail %d slot %d", origin,
             rail, slot);
    return -1;
  }
  int next = ring_next(c, target);
  int h = c->rail_hca[next][rail];
  if (c->hca[h].outstanding[next] >= ROCE_SEND_DEPTH / 4) return 0;
  if (next != target) {
    if (ring_submit(c, origin, rail, slot, next, box->data, c->ring_mr[h]->lkey, d) != 0) return -1;
    *pending = 2;
    return 0;
  }
  struct ibv_sge data = {
      .addr = (uintptr_t)box->data, .length = (uint32_t)d->nbytes, .lkey = c->ring_mr[h]->lkey};
  uint32_t value = (uint32_t)d->flag_value;
  struct ibv_sge flag = {.addr = (uintptr_t)&value, .length = 4};
  struct ibv_send_wr write = {0}, mark = {0}, *bad = NULL;
  uint64_t id = RING_COMPLETION | ((uint64_t)origin << 16) | ((uint64_t)slot << 8) | (uint64_t)rail;
  write.wr_id = id;
  write.sg_list = &data;
  write.num_sge = 1;
  write.opcode = IBV_WR_RDMA_WRITE;
  write.send_flags = IBV_SEND_SIGNALED;
  write.wr.rdma.remote_addr = c->peer_addr[target] + d->offset;
  write.wr.rdma.rkey = c->peer_rkey[rail][target];
  struct ibv_send_wr* first = &write;
  if (d->flag_offset != UINT64_MAX) {
    mark.wr_id = id;
    mark.sg_list = &flag;
    mark.num_sge = 1;
    mark.opcode = IBV_WR_RDMA_WRITE;
    mark.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    mark.wr.rdma.remote_addr = c->peer_addr[target] + d->flag_offset;
    mark.wr.rdma.rkey = c->peer_rkey[rail][target];
    if (d->nbytes) {
      write.next = &mark;
      write.send_flags = 0;
    } else
      first = &mark;
  } else if (!d->nbytes) {
    snprintf(c->err, sizeof(c->err), "empty ring write without a flag");
    return -1;
  }
  int rc = ibv_post_send(c->hca[h].qp[target], first, &bad);
  if (rc != 0) {
    set_err(c, "ring forward", rc);
    return -1;
  }
  *pending = 1;
  c->hca[h].outstanding[target]++;
  c->hca[h].bytes_posted += d->nbytes;
  return 0;
}

static int ring_service(roce_ctx_t* c) {
  if (c->ring == NULL) return 0;
  for (int origin = 0; origin < c->world; origin++) {
    if (origin == c->rank) continue;
    for (int rail = 0; rail < c->n_rail; rail++) {
      for (int slot = 0; slot < RING_WINDOW; slot++) {
        if (ring_service_slot(c, origin, rail, slot) != 0) return -1;
      }
    }
  }
  return 0;
}

// Post without waiting for delivery. A slot is reused only after its previous
// packet's end-to-end ack, which also releases its outgoing descriptor. The
// payload source is read by the first-hop write alone; its lifetime follows
// the same completion rules as a direct write.
static int ring_post(roce_ctx_t* c, int target, const uint8_t* send, uint64_t offset,
                     uint32_t packs, uint64_t flag_offset, uint32_t value) {
  int next = ring_next(c, target);
  uint32_t pack_offset = 0;
  for (int rail = 0; rail < c->n_rail; rail++) {
    uint32_t count = packs / (uint32_t)c->n_rail + ((uint32_t)rail < packs % (uint32_t)c->n_rail);
    uint32_t bytes = count * 16;
    uint64_t stripe_offset = (uint64_t)pack_offset * 16;
    uint32_t posted = 0;
    do {
      uint32_t length = bytes - posted;
      if (length > RING_CHUNK_BYTES) length = RING_CHUNK_BYTES;
      int last = posted + length == bytes;
      int h = c->rail_hca[next][rail];
      uint64_t serial = ++c->ring_serial[rail];
      int slot = (int)((serial - 1) % RING_WINDOW);
      uint64_t prior = serial > RING_WINDOW ? serial - RING_WINDOW : 0;
      while (__atomic_load_n(ring_ack(c, c->rank, rail, slot), __ATOMIC_ACQUIRE) != prior) {
        if (drain_all(c) != 0) return -1;
        uint32_t* ctrl = (uint32_t*)(c->region + c->ctrl_off);
        if (!atomic_load_explicit(&c->running, memory_order_relaxed) ||
            __atomic_load_n(&ctrl[CTRL_ERROR], __ATOMIC_ACQUIRE)) {
          snprintf(c->err, sizeof(c->err), "ring stopped waiting for rank %d via rank %d", target,
                   next);
          return -1;
        }
      }
      if (reserve(c, h, next) != 0) return -1;
      ring_descriptor_t d = {
          .serial = serial,
          .target = (uint64_t)target,
          .offset = offset + stripe_offset + posted,
          .nbytes = length,
          .flag_offset = last ? flag_offset + (uint64_t)rail * ROCE_FLAG_STRIDE : UINT64_MAX,
          .flag_value = value};
      if (ring_submit(c, c->rank, rail, slot, next, send + stripe_offset + posted,
                      c->hca[h].mr->lkey, &d) != 0)
        return -1;
      posted += length;
    } while (posted < bytes);
    pack_offset += count;
  }
  return 0;
}

// Write `packs` 16-byte packs from local `send` to peer p at region offset
// `remote_off`, striped across every rail, each stripe followed on its QP by a
// 4-byte seq write to `flag_off + rail * ROCE_FLAG_STRIDE`.  An empty payload
// still writes every flag, so receivers wait the same way for every chunk.
static int post_striped(roce_ctx_t* c, int p, const uint8_t* send, uint64_t remote_off,
                        uint32_t packs, uint64_t flag_off, uint32_t seq) {
  if (ring_relayed(c, p)) return ring_post(c, p, send, remote_off, packs, flag_off, seq);
  uint32_t seq_copy = seq;
  uint32_t pack_offset = 0;
  uint64_t remote = c->peer_addr[p];
  for (int k = 0; k < c->n_rail; k++) {
    int h = c->rail_hca[p][k];
    roce_hca_t* hca = &c->hca[h];
    if (reserve(c, h, p) != 0) {
      return -1;
    }
    uint32_t stripe_packs = packs / (uint32_t)c->n_rail;
    if ((uint32_t)k < packs % (uint32_t)c->n_rail) {
      stripe_packs += 1;
    }
    uint32_t stripe_bytes = stripe_packs * 16;
    uint64_t byte_offset = (uint64_t)pack_offset * 16;
    struct ibv_sge flag_sge = {
        .addr = (uint64_t)(uintptr_t)&seq_copy,
        .length = 4,
        .lkey = 0,
    };
    struct ibv_send_wr flag_wr;
    memset(&flag_wr, 0, sizeof(flag_wr));
    flag_wr.wr_id = (uint64_t)p;
    flag_wr.sg_list = &flag_sge;
    flag_wr.num_sge = 1;
    flag_wr.opcode = IBV_WR_RDMA_WRITE;
    flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    flag_wr.wr.rdma.remote_addr = remote + flag_off + (uint64_t)k * ROCE_FLAG_STRIDE;
    flag_wr.wr.rdma.rkey = c->peer_rkey[k][p];

    struct ibv_send_wr data_wr;
    struct ibv_sge data_sge;
    struct ibv_send_wr* first_wr = &flag_wr;
    if (stripe_bytes != 0) {
      data_sge = (struct ibv_sge){
          .addr = (uint64_t)(uintptr_t)(send + byte_offset),
          .length = stripe_bytes,
          .lkey = hca->mr->lkey,
      };
      memset(&data_wr, 0, sizeof(data_wr));
      data_wr.wr_id = (uint64_t)p;
      data_wr.next = &flag_wr;
      data_wr.sg_list = &data_sge;
      data_wr.num_sge = 1;
      data_wr.opcode = IBV_WR_RDMA_WRITE;
      data_wr.wr.rdma.remote_addr = remote + remote_off + byte_offset;
      data_wr.wr.rdma.rkey = c->peer_rkey[k][p];
      first_wr = &data_wr;
    }
    struct ibv_send_wr* bad = NULL;
    int rc = ibv_post_send(hca->qp[p], first_wr, &bad);
    if (rc != 0) {
      set_err(c, "ibv_post_send", rc);
      return -1;
    }
    hca->outstanding[p] += 1;
    hca->bytes_posted += stripe_bytes;
    pack_offset += stripe_packs;
  }
  return 0;
}

static int drain_all(roce_ctx_t* c) {
  for (int h = 0; h < c->n_hca; h++) {
    if (drain_cq(c, h) != 0) {
      return -1;
    }
  }
  return ring_service(c);
}

// Post one-shot op seq to every peer: the whole payload, one stripe plus its flag
// write per rail, each on the HCA routed to that peer for the rail.
static int post_op(roce_ctx_t* c, uint32_t seq, uint32_t nbytes) {
  if (nbytes == 0 || nbytes % 16 != 0 || nbytes > c->slot_bytes) {
    snprintf(c->err, sizeof(c->err),
             "RoCE payload size must be a positive multiple of 16 bytes within the slot, got %u",
             nbytes);
    return -1;
  }
  uint32_t slot = seq & 1u;
  const uint8_t* send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    uint64_t remote_off = c->recv_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes;
    uint64_t flag_off = c->flag_off + ((uint64_t)c->rank * ROCE_SLOTS + slot) *
                                          (uint64_t)c->n_rail * ROCE_FLAG_STRIDE;
    if (post_striped(c, p, send, remote_off, nbytes / 16, flag_off, seq) != 0) {
      return -1;
    }
  }
  c->ops_posted += 1;
  return drain_all(c);
}

// First pack of shard j of a total-pack message (the kernel uses the same split).
static inline uint32_t shard_start(const roce_ctx_t* c, uint32_t total, int j) {
  return (uint32_t)(((uint64_t)j * total) / (uint64_t)c->world);
}

// Post chunk `chunk` of two-shot phase `phase` (0 reduce-scatter: peer p's shard
// of our staged input; 1 all-gather: our reduced shard) to every peer.
static int post_pipe_chunk(roce_ctx_t* c, uint32_t seq, int phase, uint32_t chunk, uint32_t total,
                           uint32_t chunk_packs) {
  const uint8_t* send = c->region + c->pipe_send_off;
  for (int p = 0; p < c->world; p++) {
    if (p == c->rank) {
      continue;
    }
    int shard = phase == 0 ? p : c->rank;
    uint32_t first = shard_start(c, total, shard);
    uint32_t end = shard_start(c, total, shard + 1);
    uint64_t lo = (uint64_t)first + (uint64_t)chunk * chunk_packs;
    uint64_t hi = lo + chunk_packs;
    if (lo > end) {
      lo = end;
    }
    if (hi > end) {
      hi = end;
    }
    uint64_t remote_off = c->pipe_recv_off +
                          ((uint64_t)c->rank * 2 + (uint64_t)phase) * c->shard_cap +
                          (lo - first) * 16;
    uint64_t flag_off =
        c->pipe_flag_off +
        ((((uint64_t)phase * c->world + c->rank) * ROCE_MAX_CHUNKS + chunk) * (uint64_t)c->n_rail) *
            ROCE_FLAG_STRIDE;
    if (post_striped(c, p, send + lo * 16, remote_off, (uint32_t)(hi - lo), flag_off, seq) != 0) {
      return -1;
    }
  }
  return 0;
}

// Run two-shot op seq: post each reduce-scatter chunk once the kernel staged it
// and each all-gather chunk once the kernel reduced it.
static int run_pipe_op(roce_ctx_t* c, uint32_t seq, uint32_t nbytes, uint32_t chunk_packs,
                       uint32_t chunks) {
  volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
  uint32_t total = nbytes / 16;
  uint32_t widest = (total + (uint32_t)c->world - 1) / (uint32_t)c->world;
  if (nbytes == 0 || nbytes % 16 != 0 || nbytes > c->pipe_bytes || chunks == 0 ||
      chunks > ROCE_MAX_CHUNKS || chunk_packs == 0 || (uint64_t)chunks * chunk_packs < widest) {
    snprintf(c->err, sizeof(c->err),
             "invalid two-shot op %u: %u bytes in %u chunks of %u packs (capacity %zu)", seq,
             nbytes, chunks, chunk_packs, c->pipe_bytes);
    return -1;
  }
  uint32_t rs = 0;
  uint32_t ag = 0;
  uint64_t idle = 0;
  while (rs < chunks || ag < chunks) {
    int posted = 0;
    if (rs < chunks && __atomic_load_n(&ctrl[CTRL_READY_RS + rs], __ATOMIC_ACQUIRE) == seq) {
      if (post_pipe_chunk(c, seq, 0, rs, total, chunk_packs) != 0) {
        return -1;
      }
      rs++;
      posted = 1;
    }
    if (ag < chunks && __atomic_load_n(&ctrl[CTRL_READY_AG + ag], __ATOMIC_ACQUIRE) == seq) {
      if (post_pipe_chunk(c, seq, 1, ag, total, chunk_packs) != 0) {
        return -1;
      }
      ag++;
      posted = 1;
    }
    if (posted) {
      idle = 0;
      continue;
    }
    if (++idle % 64 == 0) {
      if (drain_all(c) != 0) {
        return -1;
      }
      // A kernel whose peer wait timed out publishes no further chunks.
      if (ctrl[CTRL_ERROR] != 0) {
        snprintf(c->err, sizeof(c->err), "two-shot op %u abandoned: kernel timed out at seq %u",
                 seq, ctrl[CTRL_ERROR]);
        return -1;
      }
      if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
        snprintf(c->err, sizeof(c->err), "RoCE proxy stopped inside two-shot op %u", seq);
        return -1;
      }
    }
  }
  c->ops_posted += 1;
  // Release pipe_send: wait until every posted write, our all-gather included,
  // completed, so the NIC no longer reads the shard the next op will overwrite.
  for (;;) {
    if (drain_all(c) != 0) {
      return -1;
    }
    uint32_t pending = 0;
    for (int h = 0; h < c->n_hca; h++) {
      for (int p = 0; p < c->world; p++) {
        pending += c->hca[h].outstanding[p];
      }
    }
    if (pending == 0) {
      break;
    }
    if (ctrl[CTRL_ERROR] != 0) {
      snprintf(c->err, sizeof(c->err), "two-shot op %u abandoned: kernel timed out at seq %u", seq,
               ctrl[CTRL_ERROR]);
      return -1;
    }
    if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
      snprintf(c->err, sizeof(c->err), "RoCE proxy stopped releasing two-shot op %u", seq);
      return -1;
    }
  }
  __atomic_store_n(&ctrl[CTRL_PIPE_DONE], seq, __ATOMIC_RELEASE);
  return 0;
}

// Run variable-length all-gather op seq: post each chunk of our contribution, from our row of
// the op's slot to the same row of every peer, once the kernel staged it.
static int run_varlen_op(roce_ctx_t* c, uint32_t seq, uint32_t nbytes, uint32_t chunk_packs,
                         uint32_t chunks) {
  volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
  uint32_t total = nbytes / 16;
  uint32_t expected = chunk_packs == 0 ? 0 : (total + chunk_packs - 1) / chunk_packs;
  if (expected == 0) {
    expected = 1;  // an empty contribution still sends one chunk of flags
  }
  if (c->varlen_bytes == 0 || nbytes % 16 != 0 || nbytes > c->varlen_bytes || chunk_packs == 0 ||
      chunks != expected || chunks > ROCE_VARLEN_MAX_CHUNKS) {
    snprintf(c->err, sizeof(c->err),
             "invalid variable-length op %u: %u bytes in %u chunks of %u packs (capacity %zu)", seq,
             nbytes, chunks, chunk_packs, c->varlen_bytes);
    return -1;
  }
  uint32_t slot = seq & 1u;
  uint64_t row_off =
      c->varlen_recv_off + ((uint64_t)slot * c->world + (uint64_t)c->rank) * c->varlen_bytes;
  uint32_t posted = 0;
  uint64_t idle = 0;
  while (posted < chunks) {
    // A ready word of a later op releases this one's chunk too (see the header).
    uint32_t ready = __atomic_load_n(&ctrl[CTRL_READY_VL + posted], __ATOMIC_ACQUIRE);
    if ((int32_t)(ready - seq) >= 0) {
      uint64_t lo = (uint64_t)posted * chunk_packs;
      uint64_t hi = lo + chunk_packs;
      if (lo > total) {
        lo = total;
      }
      if (hi > total) {
        hi = total;
      }
      for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
          continue;
        }
        uint64_t flag_off =
            c->varlen_flag_off +
            ((((uint64_t)slot * c->world + c->rank) * ROCE_VARLEN_MAX_CHUNKS + posted) *
             (uint64_t)c->n_rail) *
                ROCE_FLAG_STRIDE;
        if (post_striped(c, p, c->region + row_off + lo * 16, row_off + lo * 16,
                         (uint32_t)(hi - lo), flag_off, seq) != 0) {
          return -1;
        }
      }
      posted++;
      idle = 0;
      continue;
    }
    if (++idle % 64 == 0) {
      if (drain_all(c) != 0) {
        return -1;
      }
      if (ctrl[CTRL_ERROR] != 0) {
        snprintf(c->err, sizeof(c->err),
                 "variable-length op %u abandoned: kernel stopped at seq %u", seq,
                 ctrl[CTRL_ERROR]);
        return -1;
      }
      if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
        snprintf(c->err, sizeof(c->err), "RoCE proxy stopped inside variable-length op %u", seq);
        return -1;
      }
    }
  }
  c->ops_posted += 1;
  return drain_all(c);
}

static void* proxy_main(void* arg) {
  roce_ctx_t* c = (roce_ctx_t*)arg;
  volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
  // Spin while ops are flowing.  After ROCE_IDLE_SPINS polls without a
  // doorbell, request a short nanosleep between polls (the OS decides the
  // actual delay) so an idle runtime does not hold a core next to the
  // serving process.  The missed-doorbell catch-up below keeps the protocol
  // correct however long the thread is away.
  uint64_t idle = 0;
  const struct timespec nap = {0, 20000};
  while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
    uint32_t seq = __atomic_load_n(&ctrl[CTRL_SEQ], __ATOMIC_ACQUIRE);
    // Dormant chunk-ready words remain zero. Signed ready/sequence comparisons
    // cease to distinguish them after INT32_MAX, and wrapping to zero would
    // also collide with the unpoisoned error sentinel. Fail before dispatch.
    if (seq > INT32_MAX) {
      snprintf(c->err, sizeof(c->err),
               "RoCE sequence space exhausted at %u; recreate the transport", seq);
      atomic_store(&c->failed, 1);
      return NULL;
    }
    if (seq == c->last_seq) {
      idle++;
      if (idle % 64 == 0) {
        if (drain_all(c) != 0) {
          atomic_store(&c->failed, 1);
          return NULL;
        }
      }
      if (idle >= ROCE_IDLE_SPINS) {
        nanosleep(&nap, NULL);
      }
      continue;
    }
    idle = 0;
    // The doorbell holds only the newest sequence.  Our kernel for op N
    // completes on the peers' payloads alone, so op N+1 can ring before
    // this thread has seen op N (it slept, or the scheduler moved it).
    // Peers cannot get further than one op ahead of us, so at most
    // ROCE_SLOTS doorbells are pending and every slot descriptor is intact:
    // post each missed sequence in order using its per-slot record.
    uint32_t pending = seq - c->last_seq;
    if (pending > ROCE_SLOTS) {
      snprintf(c->err, sizeof(c->err), "doorbell skipped %u ops (last %u, now %u)", pending,
               c->last_seq, seq);
      atomic_store(&c->failed, 1);
      return NULL;
    }
    for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
      uint32_t slot = s & 1u;
      uint32_t nbytes = ctrl[CTRL_SLOT_NBYTES + slot];
      uint32_t kind = ctrl[CTRL_SLOT_KIND + slot];
      int rc;
      if (kind == KIND_ONESHOT) {
        rc = post_op(c, s, nbytes);
      } else if (kind == KIND_TWOSHOT) {
        rc = run_pipe_op(c, s, nbytes, ctrl[CTRL_SLOT_CHUNK_PACKS + slot],
                         ctrl[CTRL_SLOT_CHUNKS + slot]);
      } else if (kind == KIND_VARLEN) {
        rc = run_varlen_op(c, s, nbytes, ctrl[CTRL_SLOT_CHUNK_PACKS + slot],
                           ctrl[CTRL_SLOT_CHUNKS + slot]);
      } else {
        snprintf(c->err, sizeof(c->err), "unknown op kind %u at seq %u", kind, s);
        rc = -1;
      }
      if (rc != 0) {
        atomic_store(&c->failed, 1);
        return NULL;
      }
      c->last_seq = s;
    }
  }
  return NULL;
}

int roce_start(roce_ctx_t* c) {
  if (atomic_load(&c->running)) {
    return 0;
  }
  if (!c->started) {
    // A restart continues from the last posted sequence so ops that rang
    // the doorbell while the thread was stopped are still posted.
    volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
    c->last_seq = ctrl[0];
    c->started = 1;
  }
  atomic_store(&c->failed, 0);
  atomic_store(&c->running, 1);
  int rc = pthread_create(&c->thread, NULL, proxy_main, c);
  if (rc != 0) {
    atomic_store(&c->running, 0);
    set_err(c, "pthread_create", rc);
    return -1;
  }
  return 0;
}

void roce_stop(roce_ctx_t* c) {
  if (atomic_exchange(&c->running, 0)) {
    pthread_join(c->thread, NULL);
  }
}

int roce_failed(roce_ctx_t* c) { return atomic_load(&c->failed); }

const char* roce_error(roce_ctx_t* c) { return c->err; }

uint64_t roce_stat(roce_ctx_t* c, int which) {
  switch (which) {
    case 0:
      return c->ops_posted;
    case 1:
      return c->writes_completed;
    case 2:
      return c->last_seq;
    case 3:
      return c->ring_forwarded;
    case 4:
      return c->ring == NULL ? c->world - 1 : (c->world == 2 ? 1 : 2);
    default:
      return 0;
  }
}

uint64_t roce_hca_stat(roce_ctx_t* c, int hca, int which) {
  if (hca < 0 || hca >= c->n_hca) {
    return -1;
  }
  switch (which) {
    case 0:
      return c->hca[hca].writes_completed;
    case 1:
      return c->hca[hca].bytes_posted;
    default:
      return 0;
  }
}

void roce_destroy(roce_ctx_t* c) {
  if (c == NULL) {
    return;
  }
  roce_stop(c);
  for (int h = 0; h < ROCE_MAX_HCAS; h++) {
    roce_hca_t* hca = &c->hca[h];
    for (int p = 0; p < ROCE_MAX_PEERS; p++) {
      if (hca->qp[p] != NULL) {
        ibv_destroy_qp(hca->qp[p]);
      }
    }
    if (hca->cq != NULL) {
      ibv_destroy_cq(hca->cq);
    }
    if (hca->mr != NULL) {
      ibv_dereg_mr(hca->mr);
    }
    if (c->ring_mr[h] != NULL) ibv_dereg_mr(c->ring_mr[h]);
    if (hca->pd != NULL) {
      ibv_dealloc_pd(hca->pd);
    }
    if (hca->ctx != NULL) {
      ibv_close_device(hca->ctx);
    }
  }
  free(c->ring);
  free(c);
}
