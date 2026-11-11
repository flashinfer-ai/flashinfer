#pragma once

__forceinline__ __device__ void role_barrier() { asm volatile("bar.sync 0;" ::: "memory"); }

__forceinline__ __device__ void qk_warpgroup_barrier() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

__forceinline__ __device__ void v0_score_ready() {
  asm volatile("bar.arrive 8, 640;" ::: "memory");
}

__forceinline__ __device__ void v0_wait_helper_done() {
  asm volatile("bar.sync 9, 640;" ::: "memory");
}

__forceinline__ __device__ void m32_p0_qk_warpgroup_barrier() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

__forceinline__ __device__ void m32_p0_score_ready_producer() {
  asm volatile("bar.arrive 8, 512;" ::: "memory");
}

__forceinline__ __device__ void m32_p0_p_ready() { asm volatile("bar.sync 9, 512;" ::: "memory"); }

__forceinline__ __device__ void m32_p0_p_free() { asm volatile("bar.sync 10, 512;" ::: "memory"); }

__forceinline__ __device__ void p1_wg_sync(int wg) {
  asm volatile("bar.sync %0, 128;" ::"r"(wg + 12) : "memory");
}

__forceinline__ __device__ void p1_release(cutlass::arch::ClusterBarrier const& event, int peer) {
  const uint32_t address = cute::cast_smem_ptr_to_uint(&event);
  asm volatile(
      "{ .reg .b32 remote; mapa.shared::cluster.u32 remote, %0, %1; "
      "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote]; }" ::"r"(address),
      "r"(peer)
      : "memory");
}

__forceinline__ __device__ void p1_acquire(cutlass::arch::ClusterBarrier const& event) {
  const uint32_t address = cute::cast_smem_ptr_to_uint(&event);
  asm volatile(
      "{ .reg .pred ready; P1_WAIT: "
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [%0], 0, %1; "
      "@!ready bra P1_WAIT; }" ::"r"(address),
      "r"(0x989680)
      : "memory");
}

__forceinline__ __device__ void v0_wait_score_ready() {
  asm volatile("bar.sync 8, 640;" ::: "memory");
}

__forceinline__ __device__ void v0_helper_done() {
  asm volatile("bar.arrive 9, 640;" ::: "memory");
}

__forceinline__ __device__ void v0_pv_warpgroups_barrier() {
  asm volatile("bar.sync 10, 512;" ::: "memory");
}

__forceinline__ __device__ void m32_p0_score_ready_consumer() {
  asm volatile("bar.sync 8, 512;" ::: "memory");
}
