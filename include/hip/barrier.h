#pragma once

#include <new>
#include <type_traits>

#include <hip/hip_runtime.h>

#include <hip/atomic>

// libcxx/include/barrier.h
#include <hip/std/detail/libcxx/include/barrier>


namespace libhipcxx {
    using namespace hip;

    using thread_scope = hip::thread_scope;

template<thread_scope _Scope>
class pipeline;

enum async_contract_fulfillment
{
    none,
    async
};

template <typename ... _Ty>
static inline __device__ constexpr bool __unused(_Ty&&...) {return true;}

template<thread_scope _Scope, class _CompletionF = hip::std::__empty_completion>
class barrier : public hip::std::__barrier_base<_CompletionF, _Scope> {
public:
    barrier() = default;

    barrier(const barrier &) = delete;
    barrier & operator=(const barrier &) = delete;

    __host__ __device__ constexpr
    barrier(ptrdiff_t __expected, _CompletionF __completion = _CompletionF())
        : hip::std::__barrier_base<_CompletionF, _Scope>(__expected, __completion) {
    }

    __host__ __device__ constexpr
    friend void init(barrier * __b, ptrdiff_t __expected) {
        new (__b) barrier(__expected);
    }

    __host__ __device__ constexpr
    friend void init(barrier * __b, ptrdiff_t __expected, _CompletionF __completion) {
        new (__b) barrier(__expected, __completion);
    }
};

// TODO (yiakwy) : verification, see MI300X ISA
__device__ void __trap(void) { __asm__ __volatile__("s_trap;"); }

__device__ void __wait_all(void) { __asm__ volatile("s_barrier" ::); }

// TODO (yiakwy) : __memorycpy_arrive_on_impl interface API for MI300x
struct __memcpy_arrive_on_impl {
    template<thread_scope _Scope, typename _CompF, bool _Is_mbarrier = (_Scope >= thread_scope_block) && hip::std::is_same<_CompF, hip::std::__empty_completion>::value>
    static inline __host__ __device__ void __arrive_on(barrier<_Scope, _CompF> & __barrier, async_contract_fulfillment __is_async) {
        // TODO (yiakwy) : add impl for MI300X
        // see details in // see details https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html
        if (__is_async == async_contract_fulfillment::async) {
            __wait_all();
        }
    }

    template<thread_scope _Scope>
    static inline __host__ __device__ void __arrive_on(pipeline<_Scope> & __pipeline, async_contract_fulfillment __is_async) {
        // pipeline does not sync on memcpy_async, defeat pipeline purpose otherwise
        __unused(__pipeline);
        __unused(__is_async);
    }
};


} // namespace libhipcxx