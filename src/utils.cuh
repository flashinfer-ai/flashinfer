#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

namespace utils {

template <typename T>
void thrust_normal_init(thrust::device_vector<T>& vec, float mean = 0.f, float std = 1.f) {
  thrust::host_vector<T> host_vec(vec.size());
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    host_vec[i] = T(d(gen));
  }
  vec = host_vec;
}

template <typename T>
void thrust_zero_init(thrust::device_vector<T>& vec) {
  thrust::fill(vec.begin(), vec.end(), T(0));
}

template <typename T>
void thrust_normal_init(thrust::host_vector<T>& vec, float mean = 0.f, float std = 1.f) {
  thrust::counting_iterator<unsigned int> counter(0);
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = d(gen);
  }
}

template <typename T>
void thrust_zero_init(thrust::host_vector<T>& vec) {
  thrust::fill(vec.begin(), vec.end(), T(0));
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

}  // namespace utils