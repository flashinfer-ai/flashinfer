#ifndef FLASHINFER_COMM_PCIE_IPC_SPIN_GUARD_CUH_
#define FLASHINFER_COMM_PCIE_IPC_SPIN_GUARD_CUH_

// Every wait on a peer below spins until the peer arrives. A peer that never
// arrives (crashed, OOM-killed, a rank that took a different dispatch) leaves
// the others spinning forever, inside a CUDA graph where the host cannot see
// or interrupt it. FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS bounds that wait: past
// the deadline one thread per module reports where it was waiting and traps,
// turning a silent hang into a CUDA error a serving stack can restart from.
// 0 (the default) compiles the guard out and keeps the unbounded wait.
#ifndef FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS
#define FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS 0ULL
#endif

namespace flashinfer {
namespace comm {
namespace pcie_ipc {

#if FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS > 0
static __device__ int spin_timeout_reported = 0;
#endif

struct SpinGuard {
#if FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS > 0
  uint32_t iters = 0;
  unsigned long long start = 0;
#endif
  // Called once per failed poll. Reads the clock on the first failed poll (the
  // deadline counts from there; globaltimer is never 0) and then every 4096
  // polls, so a wrapping counter cannot restart the deadline.
  __device__ __forceinline__ void tick(const char* where) {
#if FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS > 0
    if (start == 0) {
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      return;
    }
    if ((++iters & 0xFFFu) != 0) return;
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - start > FLASHINFER_PCIE_IPC_SPIN_TIMEOUT_NS) {
      if (atomicExch(&spin_timeout_reported, 1) == 0) {
        printf(
            "flashinfer pcie_ipc: %s waited %llu ms for a peer (block %d, thread %d); trapping\n",
            where, (now - start) / 1000000ULL, blockIdx.x, threadIdx.x);
        __trap();
      }
    }
#else
    (void)where;
#endif
  }
};

}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_PCIE_IPC_SPIN_GUARD_CUH_
