#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace moe_activation {

enum class Kind { SWIGLU = 0, RELU_SQ = 1, RELU_CLAMP_SQ = 2 };

using ParamDict = std::unordered_map<std::string, double>;

struct NoParams {};

struct ReluClampParams {
  float clamp;
};

template <Kind ACTIVATION>
struct Traits;

template <>
struct Traits<Kind::SWIGLU> {
  using Params = NoParams;
  static constexpr Kind KIND = Kind::SWIGLU;
  static constexpr bool GATED = true;
};

template <>
struct Traits<Kind::RELU_SQ> {
  using Params = NoParams;
  static constexpr Kind KIND = Kind::RELU_SQ;
  static constexpr bool GATED = false;
};

template <>
struct Traits<Kind::RELU_CLAMP_SQ> {
  using Params = ReluClampParams;
  static constexpr Kind KIND = Kind::RELU_CLAMP_SQ;
  static constexpr bool GATED = false;
};

template <Kind ACTIVATION>
typename Traits<ACTIVATION>::Params make_params(ParamDict params) {
  const auto pop = [&](const char* name) {
    auto entry = params.extract(name);
    if (entry.empty())
      throw std::invalid_argument(std::string("Missing activation parameter: ") + name);
    return entry.mapped();
  };

  typename Traits<ACTIVATION>::Params result{};
  if constexpr (ACTIVATION == Kind::RELU_CLAMP_SQ) {
    result.clamp = static_cast<float>(pop("clamp"));
    if (!std::isfinite(result.clamp) || result.clamp <= 0.0f)
      throw std::invalid_argument("activation clamp must be positive and finite in float32");
  }
  if (!params.empty())
    throw std::invalid_argument("Unexpected activation parameter: " + params.begin()->first);
  return result;
}

}  // namespace moe_activation
