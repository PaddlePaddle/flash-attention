cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flashattn LANGUAGES CXX CUDA)

find_package(Git QUIET REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

include_directories(
    src
    cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

#file(GLOB SOURCES_CU "src/*.cu")
#file(GLOB SOURCES_CPP "src/*.cpp")
set(SOURCES_CU
    src/fmha_fwd_hdim32.cu
    src/fmha_fwd_hdim64.cu
    src/fmha_fwd_hdim128.cu
    src/fmha_bwd_hdim32.cu
    src/fmha_bwd_hdim64.cu
    src/fmha_bwd_hdim128.cu
    src/fmha_fwd_with_mask_bias_hdim32.cu
    src/fmha_fwd_with_mask_bias_hdim64.cu
    src/fmha_fwd_with_mask_bias_hdim128.cu
    src/fmha_bwd_with_mask_bias_hdim32.cu
    src/fmha_bwd_with_mask_bias_hdim64.cu
    src/fmha_bwd_with_mask_bias_hdim128.cu
    src/utils.cu)
set(SOURCES_CPP src/cuda_utils.cpp)

#add_library(flashattn OBJECT
add_library(flashattn SHARED
    .
    ${SOURCES_CU}
    ${SOURCES_CPP}
    flash_attn.cpp
    flash_attn_with_bias_mask.cpp
  )

target_compile_options(flashattn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -O3 
     -w
     -Xcompiler="-fPIC"
     -Xcompiler="-O3"
     -Xcompiler="-DVERSION_GE_1_1" 
     -Xcompiler="-DVERSION_GE_1_3" 
     -Xcompiler="-DDVERSION_GE_1_5" 
     "SHELL:-gencode arch=compute_75,code=sm_75"
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

INSTALL(TARGETS flashattn
        LIBRARY DESTINATION "lib")

INSTALL(FILES flash_attn.h DESTINATION "include")
