#pragma once
#ifndef MOE_DEBUG_H
#define MOE_DEBUG_H

#ifdef DEBUG_MOE_PRINT

#include <cstdio>

// Print from block 0, threadIdx.x == 0 only
#define DBG_B0T0(fmt, ...)                                                    \
  do {                                                                        \
    if (blockIdx.x == 0 && threadIdx.x == 0) printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)

// Print from block 0, any thread (use sparingly)
#define DBG_B0(fmt, ...)                                                             \
  do {                                                                               \
    if (blockIdx.x == 0) printf("[blk0 t%u] " fmt "\n", threadIdx.x, ##__VA_ARGS__); \
  } while (0)

// Print from block 0, specific threadIdx.x
#define DBG_B0_THREAD(tid, fmt, ...)                                              \
  do {                                                                            \
    if (blockIdx.x == 0 && threadIdx.x == (tid)) printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)

#else

#define DBG_B0T0(fmt, ...)
#define DBG_B0(fmt, ...)
#define DBG_B0_THREAD(tid, fmt, ...)

#endif  // DEBUG_MOE_PRINT

#endif  // MOE_DEBUG_H
