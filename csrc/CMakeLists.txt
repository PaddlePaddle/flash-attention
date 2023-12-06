cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX CUDA)

find_package(Git QUIET REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
#cmake -DWITH_ADVANCED
if (WITH_ADVANCED)
  add_compile_definitions(PADDLE_WITH_ADVANCED)cu
endif()
add_definitions("-DFLASH_ATTN_WITH_TORCH=0")

set(CUTLASS_3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

set(FA2_SOURCES_CU
     flash_attn/src/cuda_utils.cu
     flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu
     flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu
     flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu
     flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu
     flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu
  )

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

add_custom_target(run_my_executable
    COMMAND ${CMAKE_COMMAND} -E env python ${CMAKE_SOURCE_DIR}/setup.py bdist_wheel
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS flashattn
    COMMENT "Running my_executable"
)

add_custom_target(default_target DEPENDS run_my_executable)

# set 'default_target'
set_property(DIRECTORY PROPERTY DEFAULT_TARGET default_target)

