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
// Shared argument structs for the FuseMoE Blackwell helper kernels.
// Used by csrc/fusemoe_blackwell.cu (caller),
// csrc/fusemoe_blackwell_cutlass_bw.cu (CUTLASS group GEMM),
// and csrc/fusemoe_blackwell_tcgen05.cu (tcgen05 grouped GEMM).
#pragma once

struct GemmArgs {
  int num_groups;
  int N, K;
  void* A;
  void* B;
  void* D;
  void* SFA;
  void* SFB;
  int* m_indptr;
  int* expert_ids;
};

struct GemmArgsDual {
  int num_groups;
  int N1, K1;
  void *A1, *B1, *D1, *SFA1, *SFB1;
  int N2, K2;
  void *A2, *B2, *D2, *SFA2, *SFB2;
  int* m_indptr;
  int* expert_ids;
};
