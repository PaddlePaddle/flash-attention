# cmake .. -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc -DAMDGPU_TARGETS=gfx928 -DCMAKE_INSTALL_PREFIX=fa2_install

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX)

find_package(HIP REQUIRED)

set(CK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel)

file(
  GLOB_RECURSE
  FA2_SOURCES_HIP
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(flashattn SHARED
    flash_attn.cpp
    ${FA2_SOURCES_HIP}
  )

target_include_directories(flashattn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CK_DIR}/include
    ${CK_DIR}/library/include
    ${HIP_INCLUDE_DIRS})

target_link_libraries(flashattn PRIVATE hip::device)

target_compile_options(flashattn PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -Wno-deprecated-builtins
  -Wno-deprecated
  -DNDEBUG
  -U__HIP_NO_HALF_OPERATORS__
  -U__HIP_NO_HALF_CONVERSIONS__
  -fPIC
  -O3
  -std=c++17
  -D__HIP_PLATFORM_HCC__=1
  --offload-arch=gfx928
  -D__gfx940__
  -mllvm 
  -enable-num-vgprs-512=true
  >)

INSTALL(TARGETS flashattn
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

INSTALL(FILES flash_attn.h DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
