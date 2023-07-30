set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --expt-relaxed-constexpr --expt-relaxed-constexpr --use_fast_math -t 8 \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

include_directories(
    src
    cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}    
    )

set(SOURCES_CU
    src/fmha_fwd_with_mask_bias_hdim32.cu
    src/fmha_fwd_with_mask_bias_hdim64.cu
    src/fmha_fwd_with_mask_bias_hdim128.cu
    src/fmha_bwd_with_mask_bias_hdim32.cu
    src/fmha_bwd_with_mask_bias_hdim64.cu
    src/fmha_bwd_with_mask_bias_hdim128.cu
    src/utils.cu)
set(SOURCES_CPP src/cuda_utils.cpp)

add_library(flashattn_with_bias_mask SHARED
    .
    ${SOURCES_CU}
    ${SOURCES_CPP}
    flash_attn_with_bias_mask.cpp
  )

INSTALL(TARGETS flashattn_with_bias_mask
        LIBRARY DESTINATION "lib")
