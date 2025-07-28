# This file contains a non-exhaustive collection of useful development and build workloads.

# We recognize three types of builds:
# - Debug, unoptimized, for in-depth debugging.
# - Development, optimized, for development, with CUDA lineinfo, useful for profiling.
# - Release, optimized, for production, without CUDA lineinfo.

# Wipes ccache state entirely. This should not be necessary if the caching was implemented correctly, but it is not really.
ccache-clean:
	ccache --clear

# Resets the development debug build directory.
debug-devel-clean:
	rm -rf build_debug_devel/ && mkdir build_debug_devel

# Resets the optimized (release) development build directory.
release-devel-clean:
	rm -rf build_release_devel/ && mkdir build_release_devel

# Configures the optimized (release) development build.
release-devel-configure:
	cmake --preset=gcc_release_devel

# Builds the optimized (release) development artifacts.
release-devel-build:
	cmake --build --preset=gcc_release_devel

# Runs the tests for the optimized (release) development build.
release-devel-bench: release-devel-build
	ctest --preset=gcc_release_devel --output-on-failure --verbose --tests-regex "^bench_"

# Runs the benchmarks for the optimized (release) development build.
release-devel-test: release-devel-build
	ctest --preset=gcc_release_devel --output-on-failure --verbose --tests-regex "^test_"

# Well-known combinations of the individual targets above.
release-devel-configure-build: release-devel-configure release-devel-build
release-devel-clean-configure: release-devel-clean release-devel-configure
release-devel-clean-configure-build: release-devel-clean release-devel-configure release-devel-build
release-devel-clean-configure-build-test: release-devel-clean release-devel-configure release-devel-build release-devel-test

# Cleaning up the build directories.
all-clean: release-devel-clean debug-devel-clean

# Cleans up the build directories and ccache state.
all-deep-clean: all-clean ccache-clean