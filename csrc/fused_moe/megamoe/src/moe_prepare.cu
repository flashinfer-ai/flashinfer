
#pragma once
#ifndef MOE_PREPARE_CU
#define MOE_PREPARE_CU

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>

#include "moe_internal.h"

// This file previously contained the BS64 prepare_moe_topk_BSx_Ey function.
// The BS64 path has been removed; only the BS8 path (prepare_moe_topk_BS8
// in moe_routing.cu) remains.

#endif
