/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// One e4m3 wgmma m64nNk32 with both operands in shared memory, C = A * B (the shared expert's FC2
// tail k-block). Machine-generated inline PTX (one specialization per accumulator width the kernels
// use); the register / barrier protocol is part of the kernel's validated schedule.
#pragma once
#include <cstdint>

namespace flashinfer {
namespace tp_moe {
namespace sm90 {
namespace chain_ss {

template <int N>
struct ChainSS1;

template <>
struct ChainSS1<32> {
  __device__ __forceinline__ static void run(float (&C)[16], uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p0;\n"
        "setp.ne.b32 p0, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
        "%9, %10, %11, %12, %13, %14, %15}, %16, %17, p0, 1, 1;\n"
        "}\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]),
          "+f"(C[7]), "+f"(C[8]), "+f"(C[9]), "+f"(C[10]), "+f"(C[11]), "+f"(C[12]), "+f"(C[13]),
          "+f"(C[14]), "+f"(C[15])
        : "l"(da), "l"(db));
  }
};

}  // namespace chain_ss
}  // namespace sm90
}  // namespace tp_moe
}  // namespace flashinfer
