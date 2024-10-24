/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"

namespace flashinfer {

// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
  kQueryEmpty = 0,
  kValueEmpty = 1,
  kWarpSchedulerWG1 = 2,
  kWarpSchedulerWG2 = 3,
  kWarpSchedulerWG3 = 4,
};

}  // namespace flashinfer
