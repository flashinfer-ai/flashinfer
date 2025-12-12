#pragma once

namespace flat::collective {

struct FlatSharedNamedBarriers {
  static constexpr int AllMmaThreadsSync   = 0;
  static constexpr int AllLdStThreadsSync  = 1;
  static constexpr int MmaCooperativeStore = 2;

protected:
  static constexpr int NumBarriersUsed = 4;
};

}
