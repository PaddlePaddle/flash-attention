cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX CUDA)

find_package(Git REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

option(WITH_FLASHATTN_V3 "Enable compile with FA3" OFF)

if(NOT GIT_SUBMOD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to update Git submodules")
endif()

set(CUTLASS_3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)


find_package(PythonInterp REQUIRED)


execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" flash_attn/src/generate_kernels.py -o flash_attn/src
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Generating FA2 Python script execution failed with exit code ${result}: ${error}")
endif()

file(GLOB FA2_SOURCES_CU_SOURCES "flash_attn/src/*_sm80.cu")

message(STATUS "Auto generated CUDA source files: ${FA2_SOURCES_CU_SOURCES}")

set(FA2_REDUCE_ATTNSCORE_SOURCES_CU
    flash_attn/src/cuda_utils.cu
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
    ${FA1_SOURCES_CU}
  )

target_include_directories(flashattn PRIVATE
    flash_attn
    ${CUTLASS_3_DIR}/include)

target_include_directories(flashattn_with_bias_mask PRIVATE
    flash_attn_with_bias_and_mask/src
    flash_attn_with_bias_and_mask/cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(flashattn_with_bias_mask INTERFACE
    flash_attn_with_bias_and_mask)

target_link_libraries(flashattn flashattn_with_bias_mask)
  
add_dependencies(flashattn flashattn_with_bias_mask)


if(NOT DEFINED NVCC_ARCH_BIN OR NVCC_ARCH_BIN STREQUAL "")
    message(FATAL_ERROR "NVCC_ARCH_BIN is not defined or is empty.")
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

if(WITH_FLASHATTN_V3)
  execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" flash_attn_v3/generate_kernels.py -o flash_attn_v3
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error
  )
  file(GLOB FA3_SOURCES_CU_SOURCES "flash_attn_v3/*_sm90.cu")
  message(STATUS "Auto generated CUDA source files: ${FA3_SOURCES_CU_SOURCES}")
  add_library(flashattnv3 SHARED
      flash_attn_v3/cuda_utils.cu
      flash_attn_v3/flash_api.cu
      ${FA3_SOURCES_CU_SOURCES}
    )

  target_include_directories(flashattnv3 PRIVATE
      flash_attn_v3
      flash_attn_v3/cutlass/include
    )

  target_compile_options(flashattnv3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -w
    -Wno-abi
    -Xcompiler="-fPIC"
    -Xcompiler="-O3"
    -std=c++17
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_BFLOAT162_OPERATORS__
    -U__CUDA_NO_BFLOAT162_CONVERSIONS__
    -DCUTE_ARCH_MMA_SM90A_ENABLE
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    #--ptxas-options=-v
    #--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage
    -lineinfo
    -DCUTLASS_DEBUG_TRACE_LEVEL=0
    -DNDEBUG
    -gencode arch=compute_90a,code=sm_90a
    >)

  INSTALL(TARGETS flashattnv3
      LIBRARY DESTINATION "lib")

  INSTALL(FILES flash_attn_v3/flash_api.h DESTINATION "include" RENAME flashv3_api.h)
endif()
