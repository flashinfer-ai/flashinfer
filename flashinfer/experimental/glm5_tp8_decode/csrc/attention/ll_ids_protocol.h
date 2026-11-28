/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
// Publication half of the constant-tag LL consume/clear protocol.  Keep this
// paired with the LL consume/clear of the q4 and q1 attention consumers.
#pragma once

#ifdef __CUDACC__
#define LL_IDS_HD __host__ __device__
#else
#define LL_IDS_HD
#endif

namespace flashinfer::glm5_tp8::ll_wire {
constexpr int kIdsPerQuery = 2048;
constexpr int kRecordsPerQuery = kIdsPerQuery / 2;

// Dense attention generates arange locally and never reads or clears LL.
// Publishing its records would leave tag=1 valid for a later sparse launch.

LL_IDS_HD constexpr bool should_publish_record(int cur_pos, int record, int n_queries,
                                               bool consume_clear) {
  const int query = record / kRecordsPerQuery;
  return !consume_clear || (record >= 0 && query < n_queries && cur_pos + query + 1 > kIdsPerQuery);
}
}  // namespace flashinfer::glm5_tp8::ll_wire

#undef LL_IDS_HD
