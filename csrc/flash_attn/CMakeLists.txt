set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --expt-relaxed-constexpr --expt-relaxed-constexpr --use_fast_math -t 8 \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

include_directories(${CUTLASS_DIR}/include)

set(SOURCES_CU
     flash_attn.cu
     src/cuda_utils.cu
     src/flash_fwd_hdim32_fp16_sm80.cu
     src/flash_fwd_hdim32_bf16_sm80.cu
     src/flash_fwd_hdim64_fp16_sm80.cu
     src/flash_fwd_hdim64_bf16_sm80.cu
     src/flash_fwd_hdim96_fp16_sm80.cu
     src/flash_fwd_hdim96_bf16_sm80.cu
     src/flash_fwd_hdim128_fp16_sm80.cu
     src/flash_fwd_hdim128_bf16_sm80.cu
     src/flash_fwd_hdim160_fp16_sm80.cu
     src/flash_fwd_hdim160_bf16_sm80.cu
     src/flash_fwd_hdim192_fp16_sm80.cu
     src/flash_fwd_hdim192_bf16_sm80.cu
     src/flash_fwd_hdim224_fp16_sm80.cu
     src/flash_fwd_hdim224_bf16_sm80.cu
     src/flash_fwd_hdim256_fp16_sm80.cu
     src/flash_fwd_hdim256_bf16_sm80.cu
     src/flash_bwd_hdim32_fp16_sm80.cu
     src/flash_bwd_hdim32_bf16_sm80.cu
     src/flash_bwd_hdim64_fp16_sm80.cu
     src/flash_bwd_hdim64_bf16_sm80.cu
     src/flash_bwd_hdim96_fp16_sm80.cu
     src/flash_bwd_hdim96_bf16_sm80.cu
     src/flash_bwd_hdim128_fp16_sm80.cu
     src/flash_bwd_hdim128_bf16_sm80.cu
     src/flash_bwd_hdim160_fp16_sm80.cu
     src/flash_bwd_hdim160_bf16_sm80.cu
     src/flash_bwd_hdim192_fp16_sm80.cu
     src/flash_bwd_hdim192_bf16_sm80.cu
     src/flash_bwd_hdim224_fp16_sm80.cu
     src/flash_bwd_hdim224_bf16_sm80.cu
     src/flash_bwd_hdim256_fp16_sm80.cu
     src/flash_bwd_hdim256_bf16_sm80.cu
  )
add_library(flashattn SHARED
    ${SOURCES_CU}
  )

target_link_libraries(flashattn flashattn_with_bias_mask)
