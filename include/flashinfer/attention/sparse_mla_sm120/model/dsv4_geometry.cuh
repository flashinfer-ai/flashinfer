// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

struct Dsv4Geometry {
  static constexpr int D_NOPE = 448;
  static constexpr int D_ROPE = 64;
  static constexpr int D_QK = D_NOPE + D_ROPE;
  static constexpr int D_V = D_QK;
  static constexpr bool V_HAS_ROPE = true;
};
