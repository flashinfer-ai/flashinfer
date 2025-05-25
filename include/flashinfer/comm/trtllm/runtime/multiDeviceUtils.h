/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// #include "assert.h"
// #include "stringUtils.h"

#include <mpi.h>
#include <nccl.h>

#include "../../../exception.h"

#define TLLM_MPI_CHECK(cmd)                                                                    \
  do {                                                                                         \
    auto e = cmd;                                                                              \
    FLASHINFER_CHECK(e == MPI_SUCCESS, "Failed: MPI error %s:%d '%d'", __FILE__, __LINE__, e); \
  } while (0)

#define TLLM_NCCL_CHECK(cmd)                                                                  \
  do {                                                                                        \
    ncclResult_t r = cmd;                                                                     \
    FLASHINFER_CHECK(r == ncclSuccess, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                     ncclGetErrorString(r));                                                  \
  } while (0)
