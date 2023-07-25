cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flashattn LANGUAGES CXX CUDA)

find_package(Git QUIET REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

#
# CUTLASS 3.x requires C++17
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#if(CUTLASS_NATIVE_CUDA)
#  set(CMAKE_CUDA_STANDARD 17)
#  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
#else()
#  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++17)
#endif()

include_directories(
    src
    ../cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

#file(GLOB SOURCES_CU "src/*.cu")
#file(GLOB SOURCES_CPP "src/*.cpp")
#set(SOURCES_CU
#    src/fmha_fwd_hdim32.cu
#    src/fmha_fwd_hdim64.cu
#    src/fmha_fwd_hdim128.cu
#    src/fmha_bwd_hdim32.cu
#    src/fmha_bwd_hdim64.cu
#    src/fmha_bwd_hdim128.cu
#    src/fmha_fwd_with_mask_bias_hdim32.cu
#    src/fmha_fwd_with_mask_bias_hdim64.cu
#    src/fmha_fwd_with_mask_bias_hdim128.cu
#    src/fmha_bwd_with_mask_bias_hdim32.cu
#    src/fmha_bwd_with_mask_bias_hdim64.cu
#    src/fmha_bwd_with_mask_bias_hdim128.cu
#    src/utils.cu)
#set(SOURCES_CPP src/cuda_utils.cpp)
#
#add_library(flashattn OBJECT
set(SOURCES_CU
     flash_attn.cu
     src/cuda_utils.cu
     src/flash_fwd_hdim32_fp16_sm80.cu)
#     src/flash_fwd_hdim32_bf16_sm80.cu
#     src/flash_fwd_hdim64_fp16_sm80.cu
#     src/flash_fwd_hdim64_bf16_sm80.cu
#     src/flash_fwd_hdim96_fp16_sm80.cu
#     src/flash_fwd_hdim96_bf16_sm80.cu
#     src/flash_fwd_hdim128_fp16_sm80.cu
#     src/flash_fwd_hdim128_bf16_sm80.cu
#     src/flash_fwd_hdim160_fp16_sm80.cu
#     src/flash_fwd_hdim160_bf16_sm80.cu
#     src/flash_fwd_hdim192_fp16_sm80.cu
#     src/flash_fwd_hdim192_bf16_sm80.cu
#     src/flash_fwd_hdim224_fp16_sm80.cu
#     src/flash_fwd_hdim224_bf16_sm80.cu
#     src/flash_fwd_hdim256_fp16_sm80.cu
#     src/flash_fwd_hdim256_bf16_sm80.cu)
#     src/flash_bwd_hdim32_fp16_sm80.cu
#     src/flash_bwd_hdim32_bf16_sm80.cu
#     src/flash_bwd_hdim64_fp16_sm80.cu
#     src/flash_bwd_hdim64_bf16_sm80.cu
#     src/flash_bwd_hdim96_fp16_sm80.cu
#     src/flash_bwd_hdim96_bf16_sm80.cu
#     src/flash_bwd_hdim128_fp16_sm80.cu
#     src/flash_bwd_hdim128_bf16_sm80.cu
#     src/flash_bwd_hdim160_fp16_sm80.cu
#     src/flash_bwd_hdim160_bf16_sm80.cu
#     src/flash_bwd_hdim192_fp16_sm80.cu
#     src/flash_bwd_hdim192_bf16_sm80.cu
#     src/flash_bwd_hdim224_fp16_sm80.cu
#     src/flash_bwd_hdim224_bf16_sm80.cu
#     src/flash_bwd_hdim256_fp16_sm80.cu
#     src/flash_bwd_hdim256_bf16_sm80.cu)
#set(SOURCES_CPP src/cuda_utils.cpp)
add_library(flashattn SHARED
    .
#   flash_attn.cpp
    ${SOURCES_CU}
#    ${SOURCES_CPP}
#    flash_attn_with_bias_mask.cpp
  )

target_compile_options(flashattn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -O3 
     -w
     -Xcompiler="-fPIC"
     -Xcompiler="-O3"
     -Xcompiler="-std=c++17"
     -Xcompiler="-DVERSION_GE_1_1" 
     -Xcompiler="-DVERSION_GE_1_3" 
     -Xcompiler="-DDVERSION_GE_1_5" 
#     "SHELL:-gencode arch=compute_75,code=sm_75"
     "SHELL:-gencode arch=compute_80,code=sm_80"
     "SHELL:-gencode arch=compute_86,code=sm_86"
     -U__CUDA_NO_HALF_OPERATORS__ 
     -U__CUDA_NO_HALF_CONVERSIONS__ 
     -U__CUDA_NO_HALF2_OPERATORS__
     -U__CUDA_NO_BFLOAT16_CONVERSIONS__
     --expt-relaxed-constexpr 
     --expt-extended-lambda 
     --use_fast_math 
     -DVERSION_GE_1_1 
     -DVERSION_GE_1_3 
     -DVERSION_GE_1_5
     >)

target_compile_options(flashattn PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
     -O3
#     "-std=c++17"
     >)

#target_compile_features(flashattn PRIVATE cxx_std_17)

INSTALL(TARGETS flashattn
        LIBRARY DESTINATION "lib")

INSTALL(FILES flash_attn.h DESTINATION "include")
