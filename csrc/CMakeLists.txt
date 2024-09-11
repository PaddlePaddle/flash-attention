cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX CUDA)

find_package(Git QUIET REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

set(CUTLASS_3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

execute_process(
    COMMAND python flash_attn/src/generate_kernels.py -o flash_attn/src
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if(NOT result EQUAL 0)
    message(STATUS "Generating FA2 Python script error: ${error}")
    message(FATAL_ERROR "Generating FA2 Python script execution failed with exit code ${result}")
endif()

file(GLOB FA2_SOURCES_CU_SOURCES "flash_attn/src/*.cu")

message(STATUS "Auto generated CUDA source files: ${FA2_SOURCES_CU_SOURCES}")

set(FA2_REDUCE_ATTNSCORE_SOURCES_CU
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim32_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim32_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim64_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim64_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim96_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim96_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim128_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim128_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim160_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim160_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim192_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim192_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim224_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim224_bf16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim256_fp16_sm80.cu
    flash_attn/src/calc_reduced_attn_scores_dispatch/hdim256_bf16_sm80.cu
)

# merge FA2 cu sources
set(FA2_SOURCES_CU ${FA2_REDUCE_ATTNSCORE_SOURCES_CU} ${FA2_SOURCES_CU_SOURCES})

add_library(flashattn SHARED
    capi/flash_attn.cu
    ${FA2_SOURCES_CU}
  )
target_include_directories(flashattn PRIVATE
    flash_attn
    ${CUTLASS_3_DIR}/include)

set(FA1_SOURCES_CU
    flash_attn_with_bias_and_mask/flash_attn_with_bias_mask.cu
    flash_attn_with_bias_and_mask/src/cuda_utils.cu
    flash_attn_with_bias_and_mask/src/fmha_fwd_with_mask_bias_hdim32.cu
    flash_attn_with_bias_and_mask/src/fmha_fwd_with_mask_bias_hdim64.cu
    flash_attn_with_bias_and_mask/src/fmha_fwd_with_mask_bias_hdim128.cu
    flash_attn_with_bias_and_mask/src/fmha_bwd_with_mask_bias_hdim32.cu
    flash_attn_with_bias_and_mask/src/fmha_bwd_with_mask_bias_hdim64.cu
    flash_attn_with_bias_and_mask/src/fmha_bwd_with_mask_bias_hdim128.cu
    flash_attn_with_bias_and_mask/src/utils.cu)

add_library(flashattn_with_bias_mask STATIC
    flash_attn_with_bias_and_mask/
    ${FA1_SOURCES_CU}
  )

target_include_directories(flashattn_with_bias_mask PRIVATE
    flash_attn_with_bias_and_mask/src
    flash_attn_with_bias_and_mask/cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(flashattn_with_bias_mask INTERFACE
    flash_attn_with_bias_and_mask)

target_link_libraries(flashattn flashattn_with_bias_mask)
  
add_dependencies(flashattn flashattn_with_bias_mask)


if (NOT DEFINED NVCC_ARCH_BIN)
  message(FATAL_ERROR "NVCC_ARCH_BIN is not defined.")
endif()

if (NVCC_ARCH_BIN STREQUAL "")
  message(FATAL_ERROR "NVCC_ARCH_BIN is not set.")
endif()

STRING(REPLACE "-" ";" FA_NVCC_ARCH_BIN ${NVCC_ARCH_BIN})

set(FA_GENCODE_OPTION "SHELL:")
foreach(arch ${FA_NVCC_ARCH_BIN})
   if(${arch} GREATER_EQUAL 80)
     set(FA_GENCODE_OPTION "${FA_GENCODE_OPTION} -gencode arch=compute_${arch},code=sm_${arch}")
   endif()
endforeach()

target_compile_options(flashattn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -w
  -Xcompiler="-fPIC"
  -Xcompiler="-O3"
  -std=c++17
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
  "${FA_GENCODE_OPTION}"
  >)

target_compile_options(flashattn_with_bias_mask PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -w
  -Xcompiler="-fPIC"
  -Xcompiler="-O3"
  -std=c++17
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
  "${FA_GENCODE_OPTION}"
  >)

INSTALL(TARGETS flashattn
        LIBRARY DESTINATION "lib")

INSTALL(FILES capi/flash_attn.h DESTINATION "include")
