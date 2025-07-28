ccache-clean:
	ccache --clear

debug-devel-clean:
	rm -rf build_debug/ && mkdir build_debug

release-devel-clean:
	rm -rf build/ && mkdir build

release-devel-configure:
	cmake --preset=gcc_release_devel

release-devel-build:
	cmake --build --preset=gcc_release_devel

release-devel-test:
	ctest --preset=gcc_release_devel --output-on-failure --verbose

release-devel-configure-build: release-devel-configure release-devel-build
release-devel-clean-configure: release-devel-clean release-devel-configure
release-devel-clean-configure-build: release-devel-clean release-devel-configure release-devel-build
release-devel-clean-configure-build-test: release-devel-clean release-devel-configure release-devel-build release-devel-test

all-clean: release-devel-clean debug-devel-clean

all-deep-clean: all-clean ccache-clean