// TODO (yiakwy) : to be integrated into libhipcxx; POC purpose, will be moved out soon
#pragma once

// TODO (yiakwy) : only mi300x supported, other archs will be supported soon
#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif

#include <hip/hip_runtime.h>

// helpers
// ported from llvm project

template <typename MaskT, typename T>
static __device__ inline
unsigned long long __match_any_sync(MaskT mask, T value) {
    static_assert(
        __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
        "The mask must be a 64-bit integer. "
        "Implicitly promoting a smaller integer is almost always an error.");
    __hip_adjust_mask_for_wave32(mask);
    __hip_check_mask(mask);
    return __match_any(value) & mask;
}

#ifdef HIP_ENABLE_WARP_SYNC_BUILTINS
static __device__ inline
unsigned long long __activemask() {
    return __ballot(true);
}
#endif // HIP_ENABLE_WARP_SYNC_BUILTINS

// ported from <hip/amd_detail/amd_warp_functions.h> in SDK 6.2
struct __pipeline_asm_helper {
    __device__ static inline 
    uint32_t __lane_id() {
        return  __builtin_amdgcn_mbcnt_hi(
            -1, __builtin_amdgcn_mbcnt_lo(-1, 0));
    }
};

__device__ static inline unsigned int __ffs(uint64_t input) {
    return ( input == 0 ? -1 : __builtin_ctzll(input) ) + 1;
}

// TODO (yiakwy) : these headers may not find relevant functions
#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/amd_detail/amd_warp_sync_functions.h>

#include <amd/target.h>

// install from libhipcxx
#include <hip/atomic>
// #include <hip/barrier>

#include "hip/barrier.h"

#include "flashinfer/hip_warp_sync_functions.h"


namespace libhipcxx {
    using namespace hip;

    using thread_scope = hip::thread_scope;

    template<hip::thread_scope _Scope, class _CompletionF>
    class barrier;

    /*
    template<hip::thread_scope _Scope>
    using barrier = hip::barrier<_Scope>;
     */

    /*
enum thread_scope {
    thread_scope_system = __ATOMIC_SYSTEM,
    thread_scope_device = __ATOMIC_DEVICE,
    thread_scope_block = __ATOMIC_BLOCK,
    thread_scope_thread = __ATOMIC_THREAD
};
    */
    template<hip::thread_scope _Scope>
    struct __pipeline_stage {
        barrier<_Scope> __produced;
        barrier<_Scope> __consumed;
    };

    template<hip::thread_scope _Scope>
    class pipeline;

    // AMD uses 64 (__AMDGCN_WAVEFRONT_SIZE) threads wave, while NVIDIA uses 32 threads wave 
    using WAVE_MASK_TYPE=uint64_t;

    // TODO (yiakwy) : implement hip/pipline
    // We mimic a pair barriers used by NVIDIA to synchronize device threads accessing to shared memroy or registers.
    //
    // Consumer threads wait on “consumer barrier” (no need proceed to the barrier) until data is available and arrive to "producer barriers"
    // to notify the shared resources can be reuse.
    // 
    // Once data is prepared, producer threads arrive to "consumer barrier" to notify consumer threads and wait on "producer barrier" (no need
    // proceed to the barrier) to continue data production loop.
    //
    // Details can be found here : https://eel.is/c++draft/thread.barrier#class-1.3
    template <hip::thread_scope _Scope>
    class pipeline {
    private:
        uint8_t __head;
        uint8_t __tail;
        const uint8_t __stages_count;
        bool __consumed_phase_parity;
        bool __produced_phase_parity;
        bool __active;
        const bool __partitioned;
        char * const __shared_state;

    public:
        // forbidden R-Val copies
        pipeline(pipeline &&) = default;
        pipeline & operator=(pipeline &&) = delete;

        pipeline();
        
        void init() {

        }

        void copy() {

        }

        void clear() {

        }


        __host__ __device__ ~pipeline() {
            if (__active) quit();
        };

        pipeline& operator=(pipeline const&) = delete;

        __host__ __device__ void producer_acquire();

        __host__ __device__ void producer_commit();

        __host__ __device__ void consumer_wait();

        template <typename Rep, typename Period>
        __host__ __device__ bool consumer_wait_for(hip::std::chrono::duration<Rep, Period> const& duration);

        template <typename Clock, typename Duration>
        __host__ __device__
        bool consumer_wait_until(hip::std::chrono::time_point<Clock, Duration> const& time_point);

        __host__ __device__ void consumer_release();

        __host__ __device__ bool quit();

        private:
            atomic<WAVE_MASK_TYPE, _Scope> * __shared_state_get_refcount() {
                ptrdiff_t __refcount_offset = __stages_count * sizeof(__pipeline_stage<_Scope>);
                return reinterpret_cast<atomic<WAVE_MASK_TYPE, _Scope>*>(__shared_state + __refcount_offset);
            }

            __pipeline_stage<_Scope> * __shared_state_get_stage(uint8_t __stage)
            {
                ptrdiff_t __stage_offset = __stage * sizeof(__pipeline_stage<_Scope>);
                return reinterpret_cast<__pipeline_stage<_Scope>*>(__shared_state + __stage_offset);
            }

    };

} //  namespace libhipcxx

// TODO (yiakwy) : move implementation specialization to implementation folder (e.g. : impl/pipeline ) 
namespace libhipcxx {

// TODO (yiakwy)
template<hip::thread_scope _Scope>
pipeline<_Scope>::pipeline() {

}

template<hip::thread_scope _Scope>
__host__ __device__ 
bool pipeline<_Scope>::quit() {
    bool __elected;
    WAVE_MASK_TYPE __sub_count;
    const WAVE_MASK_TYPE __match_mask = __match_any_sync(__activemask(), reinterpret_cast<uintptr_t>(__shared_state_get_refcount()));
    const WAVE_MASK_TYPE __elected_id = __ffs(__match_mask) - 1;
    __elected = (__pipeline_asm_helper::__lane_id() == __elected_id);
    __sub_count = __popc(__match_mask);

    __elected = true;
    __sub_count = 1;

    bool __released = false;
    if (__elected) {
        const WAVE_MASK_TYPE __old = __shared_state_get_refcount()->fetch_sub(__sub_count);
        const bool __last = (__old == __sub_count);
        if (__last) {
            for (uint8_t __stage = 0; __stage < __stages_count; ++__stage) {
                __shared_state_get_stage(__stage)->__produced.~barrier();
                __shared_state_get_stage(__stage)->__consumed.~barrier();
            }
            __released = true;
        }
    }
    __active = false;
    return __released;
}

template<hip::thread_scope _Scope>
__host__ __device__
void pipeline<_Scope>::producer_acquire() {
    // wait for producer barrier that used resources can be reused
    barrier<_Scope> & __stage_barrier = __shared_state_get_stage(__head)->__consumed;
    __stage_barrier.wait_parity(__consumed_phase_parity);
}

template<hip::thread_scope _Scope>
__host__ __device__
void pipeline<_Scope>::producer_commit() {
    // arrive to consumer barrier to notfiy the sources are available to use
    barrier<_Scope> & __stage_barrier = __shared_state_get_stage(__head)->__produced;
    __memcpy_arrive_on_impl::__arrive_on(__stage_barrier, async_contract_fulfillment::async);
    (void)__stage_barrier.arrive();
    if (++__head == __stages_count) {
        __head = 0;
        __consumed_phase_parity = !__consumed_phase_parity;
    }
}

template<hip::thread_scope _Scope>
__host__ __device__
void pipeline<_Scope>::consumer_wait() {
    // wait for consumer barrier that data is available
    barrier<_Scope> & __stage_barrier = __shared_state_get_stage(__tail)->__produced;
    __stage_barrier.wait_parity(__produced_phase_parity);
}

template<hip::thread_scope _Scope>
__host__ __device__
void pipeline<_Scope>::consumer_release() {
    // arrive producer barrier that the resources can be reused
    (void)__shared_state_get_stage(__tail)->__consumed.arrive();
    if (++__tail == __stages_count) {
        __tail = 0;
        __produced_phase_parity = !__produced_phase_parity;
    }
}

template<hip::thread_scope _Scope>
template<class _Rep, class _Period>
__host__ __device__
bool pipeline<_Scope>::consumer_wait_for(const hip::std::chrono::duration<_Rep, _Period> & __duration) {
    // wait for at most __duration for producer to arrive consumer barrier
    barrier<_Scope> & __stage_barrier = __shared_state_get_stage(__tail)->__produced;
    return hip::std::__libcpp_thread_poll_with_backoff(
                hip::std::__barrier_poll_tester_parity<barrier<_Scope>>(
                    &__stage_barrier,
                    __produced_phase_parity),
                hip::std::chrono::duration_cast<hip::std::chrono::nanoseconds>(__duration)
    );
}

template<hip::thread_scope _Scope>
template<class _Clock, class _Duration>
__host__ __device__
bool pipeline<_Scope>::consumer_wait_until(const hip::std::chrono::time_point<_Clock, _Duration> & __time_point) {
    return consumer_wait_for(__time_point - _Clock::now());
}

} // namespace libhipcxx