#pragma once
#include "config.cuh"
#include "primitives.cuh"
// Shared executable code; structural branches preserve lexical scope.
#include <cuda_runtime.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <vector>
namespace sm90::dense_swa {
using namespace cute;
using TMABarrier = cutlass::arch::ClusterTransactionBarrier;
template <typename ShapeQ, typename TMAQ, typename ShapeK, typename TMAK>
struct TmaParams {
  ShapeQ shape_q;
  TMAQ tma_q;
  ShapeK shape_k;
  TMAK tma_k;
};

#include "device/storage.cuh"
#include "device/traits.cuh"
}  // namespace sm90::dense_swa
namespace sm90::dense_swa::roles {
using namespace cute;
// These implementation fragments are included in declaration dependency order.
// clang-format off
#include "device/contracts.cuh"
#include "device/wgmma.cuh"
#include "device/accumulator.cuh"
#include "device/memory.cuh"
#include "device/sync.cuh"
#include "device/tma.cuh"
#include "device/gemm.cuh"
#include "device/softmax.cuh"
#include "device/mailbox.cuh"
#include "device/owner_pv.cuh"
#include "device/qk_sequence.cuh"
#include "device/native_softmax.cuh"
#include "device/pv.cuh"
#include "device/pv_role.cuh"
#include "device/qk.cuh"
#include "device/attention.cuh"
#include "device/entries.cuh"
#include "device/combine.cuh"
// clang-format on
}  // namespace sm90::dense_swa::roles
