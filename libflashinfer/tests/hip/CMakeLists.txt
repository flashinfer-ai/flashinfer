cmake_minimum_required(VERSION 3.21)

project(FlashInferTests LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5 # release-1.16.0
  FIND_PACKAGE_ARGS NAMES GTest)
# )
FetchContent_MakeAvailable(googletest)

# GoogleTest has to be included so that gtest_discover_tests works
include(GoogleTest)
# Thread support is needed for gtest
find_package(Threads REQUIRED)

find_package(HIP REQUIRED)
set(CMAKE_HIP_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} --offload-arch=gfx942")

#Add flashinfer directory here
set(FLASHINFER_INCLUDE_DIRS Enter the path to the FlashInfer source code)
# Example: set(FLASHINFER_INCLUDE_DIRS /root/flashinfer/include/flashinfer/hip)

include_directories(${HIP_INCLUDE_DIRS})
include_directories("/opt/rocm/include") # Add this explicit include path
include_directories(${FLASHINFER_INCLUDE_DIRS})
include_directories(${CMAKE_CURRNET_SOURCE_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add definition to use HIP API
add_definitions(-D__HIP_PLATFORM_AMD__)

add_library(math STATIC ${FLASHINFER_INCLUDE_DIRS}/math.hip.h)
add_library(layout STATIC ${FLASHINFER_INCLUDE_DIRS}/layout.hip.h)
add_library(pos_enc STATIC ${FLASHINFER_INCLUDE_DIRS}/pos_enc.hip.h)
add_library(state STATIC ${FLASHINFER_INCLUDE_DIRS}/attention/state.hip.h)
add_library(cascade STATIC ${FLASHINFER_INCLUDE_DIRS}/attention/cascade.hip.h)
add_library(page STATIC ${FLASHINFER_INCLUDE_DIRS}/page.hip.h)

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -g -D__HIP_PLATFORM_AMD__")

set_target_properties(math PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

set_target_properties(layout PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

set_target_properties(pos_enc PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

set_target_properties(state PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

set_target_properties(cascade PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

set_target_properties(page PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

target_link_libraries(math PUBLIC hip::host hip::device)
target_link_libraries(layout PUBLIC hip::host hip::device)
target_link_libraries(pos_enc PUBLIC hip::host hip::device)
target_link_libraries(state PUBLIC hip::host hip::device)
target_link_libraries(cascade PUBLIC hip::host hip::device)
target_link_libraries(page PUBLIC hip::host hip::device)
message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_math ${CMAKE_CURRNET_SOURCE_DIR}test_math.cpp)
target_link_libraries(test_math PRIVATE
math
"${TORCH_LIBRARIES}" 
gtest
gtest_main
pthread
)

add_executable(test_pos_enc ${CMAKE_CURRNET_SOURCE_DIR}test_pos_enc.cpp)
target_link_libraries(test_pos_enc PRIVATE
pos_enc
"${TORCH_LIBRARIES}" 
gtest
gtest_main
pthread
)

add_executable(test_cascade ${CMAKE_CURRNET_SOURCE_DIR}test_cascade.cpp)
target_link_libraries(test_cascade PRIVATE
state
cascade
"${TORCH_LIBRARIES}" 
gtest
gtest_main
pthread
)

add_executable(test_page ${CMAKE_CURRNET_SOURCE_DIR}test_page.cpp)
target_link_libraries(test_page PRIVATE
page
"${TORCH_LIBRARIES}" 
layout
gtest
gtest_main
pthread
)

if(HIP_FOUND)
    set_target_properties(test_math PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    set_target_properties(test_pos_enc PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    set_target_properties(test_cascade PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    set_target_properties(test_page PROPERTIES HIP_SEPARABLE_COMPILATION ON)
endif()

enable_testing()
add_test(NAME MathTest COMMAND test_math)
add_test(NAME PosEncTest COMMAND test_pos_enc)
add_test(NAME CascadeTest COMMAND test_cascade)
add_test(NAME PageTest COMMAND test_page)
 