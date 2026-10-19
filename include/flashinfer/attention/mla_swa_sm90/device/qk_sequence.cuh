#pragma once
template <class Program>
struct QKExecutor;
template <class... Ops>
struct QKExecutor<mla_swa::QKSequence<Ops...>> {
  template <class Op, class T, class MMA, class A, class B, class Score, class Plan>
  __forceinline__ __device__ static void issue(MMA& mma, A const& a, B const& b, Score& score,
                                               Plan& plan, bool phase) {
    if constexpr (Op::second < 0)
      issue_qk_ready_single<Op::first, Op::zero, T>(mma, a, b, score, plan, phase);
    else
      issue_qk_ready_pair<Op::first, Op::second, Op::zero, T>(mma, a, b, score, plan, phase);
  }
  template <class T, class MMA, class A, class B, class Score, class Plan>
  __forceinline__ __device__ static void execute(MMA& mma, A const& a, B const& b, Score& score,
                                                 Plan& plan, bool phase) {
    warpgroup_fence_operand(score);
    (issue<Ops, T>(mma, a, b, score, plan, phase), ...);
    warpgroup_wait<0>();
    warpgroup_fence_operand(score);
  }
};
