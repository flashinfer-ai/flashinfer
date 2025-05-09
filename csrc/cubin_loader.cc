/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <pybind11/functional.h>

#include <iostream>

#undef Py_LIMITED_API  // pybind11 blows up otherwise

#include <pybind11/pybind11.h>

namespace py = pybind11;

std::function<std::string(std::string, std::string)> getCubinCb;
void register_cubin_loader(std::function<std::string(std::string, std::string)> cb) {
  getCubinCb = cb;
}

void shutdown_cubin_loader() { getCubinCb = nullptr; }

std::string getCubin(const std::string& kernelName, const std::string& sha256) {
  return getCubinCb(kernelName, sha256);
}

PYBIND11_MODULE(cubin_utils, m) {
  m.def("register_cubin_loader", &register_cubin_loader, "Register cubin callback");
  // Register an automatic cleanup via Python's atexit
  // This lambda will be called when the Python interpreter is shutting down.
  py::module atexit = py::module::import("atexit");
  atexit.attr("register")(py::cpp_function([]() {
    // The cleanup is performed while the interpreter is still alive.
    shutdown_cubin_loader();
  }));
}
