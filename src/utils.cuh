#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

namespace utils {

template <typename T>
struct thrust_prg {
  float mean, std;

  __host__ __device__ thrust_prg(float mean = 0.f, float std = 1.f) : mean(mean), std(std){};

  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::random::normal_distribution<float> dist(mean, std);
    rng.discard(n);

    return T(dist(rng));
  }
};

template <typename T>
void thrust_normal_init(thrust::device_vector<T>& vec, float mean = 0.f, float std = 1.f) {
  thrust::counting_iterator<unsigned int> counter(0);
  thrust::transform(counter, counter + vec.size(), vec.begin(), thrust_prg<T>(mean, std));
}

template <typename T>
void thrust_zero_init(thrust::device_vector<T>& vec) {
  thrust::fill(vec.begin(), vec.end(), T(0));
}

template <typename T>
void thrust_normal_init(thrust::host_vector<T>& vec, float mean = 0.f, float std = 1.f) {
  thrust::counting_iterator<unsigned int> counter(0);
  thrust::transform(counter, counter + vec.size(), vec.begin(), thrust_prg<T>(mean, std));
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