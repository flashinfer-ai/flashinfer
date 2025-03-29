#pragma once
#include <string>
#include <cutlass/half.h>
#include "spdlog/spdlog.h"
#include "tensor.cuh"
#include <cuda_fp16.h>

template <typename T>
inline std::string _myToString(T value) {
	return std::to_string(value);
}

template <>
inline std::string _myToString(half value) {
	return std::to_string((float)__half2float(value));
}

template <>
inline std::string _myToString(cutlass::half_t value) {
	return std::to_string((float)__half2float((half)value));
}


template <typename T>
inline void log_tensor(std::shared_ptr<spdlog::logger> private_logger, std::string name, T* device_ptr, int dimN, int rows, int cols, int start_row, int start_col, char delim=',', std::string majorString = "ROW_MAJOR"){
    cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		private_logger->error("Error before copying tensor from device to host: {}", cudaGetErrorString(error));
		private_logger->error("Tensor {} (row {}->{}, col {}->{}) logged", name, start_row, start_row + rows, start_col, start_col + cols);
	}
    T* host_ptr = new T[rows * cols];
	for (int i = 0; i < rows; i++) {
		// spdlog::info("Copying ptr {} {} from device to host with size {}", (size_t) (device_ptr) , (size_t) (start_row + i) * dimN + start_col, cols * sizeof(T));
		cudaMemcpy(host_ptr + i * cols, device_ptr + (start_row + i) * dimN + start_col, cols * sizeof(T), cudaMemcpyDeviceToHost);
	}

	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		private_logger->error("Error in copying tensor from device to host: {}", cudaGetErrorString(error));
		private_logger->error("Tensor {} (row {}->{}, col {}->{}) not logged", name, start_row, start_row + rows, start_col, start_col + cols);
		private_logger->error("ptr = {}, dimN = {}, rows = {}, cols = {}, start_row = {}, start_col = {}", (size_t)device_ptr, dimN, rows, cols, start_row, start_col);
	}


	private_logger->info("--------Tensor {} (row {}->{}, col {}->{}) ------------", name, start_row, start_row + rows, start_col, start_col + cols);
	for (int i = 0; i < rows; i++) {
		std::string s = "";
		for (int j = 0; j < cols; j++) {
			s += _myToString(host_ptr[i * cols + j]);
			if (j != cols - 1) s += delim;
		}
		private_logger->info(s);
	}
	delete[] host_ptr;
	private_logger->info("");
}


template <typename T>
inline void log_tensor(std::shared_ptr<spdlog::logger> private_logger, std::string name, pllmTensor<T> device_tensor, int rows = 1, int cols = 20, int start_row = 0, int start_col = 0, char delim = ','){
    private_logger->info("--------Tensor {} ({}) ------------", name, device_tensor.shapeString());
	log_tensor(private_logger, name, device_tensor.ptr, device_tensor.dim2, rows, cols, start_row, start_col, delim, toString(device_tensor.layout));
}
