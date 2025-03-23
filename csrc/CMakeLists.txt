cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX CUDA)

find_package(Git REQUIRED)

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)

option(SKIP_BUILD_FA "Enable compile with FA3" OFF)
option(WITH_FLASHATTN_V3 "Enable compile with FA3" OFF)

if(NOT GIT_SUBMOD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to update Git submodules")
endif()

if(NOT SKIP_BUILD_FA)

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
        COMMAND "${PYTHON_EXECUTABLE}" flash_attn_v3/generate_kernels.py -o flash_attn_v3/instantiations
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    # Assume options or variables are defined elsewhere in the CMake project
    # You can define them for testing purposes like this:
    option(DISABLE_FP16 "Disable FP16" OFF)
    option(DISABLE_FP8 "Disable FP8" OFF)
    option(DISABLE_HDIM64 "Disable HDIM64" OFF)
    option(DISABLE_HDIM96 "Disable HDIM96" OFF)
    option(DISABLE_HDIM128 "Disable HDIM128" OFF)
    option(DISABLE_HDIM192 "Disable HDIM192" OFF)
    option(DISABLE_HDIM256 "Disable HDIM256" OFF)
    option(DISABLE_SPLIT "Disable Split" OFF)
    option(DISABLE_PAGEDKV "Disable PagedKV" OFF)
    option(DISABLE_SOFTCAP "Disable Softcap" OFF)
    option(DISABLE_PACKGQA "Disable PackGQA" OFF)
    option(DISABLE_BACKWARD "Disable Backward" OFF)
    option(DISABLE_SM8X "Disable SM8x" OFF)

    if(DISABLE_FP16)
      add_compile_definitions(FLASHATTENTION_DISABLE_FP16)
    endif()

    if(DISABLE_FP8)
      add_compile_definitions(FLASHATTENTION_DISABLE_FP8)
    endif()

    if(DISABLE_HDIM64)
      add_compile_definitions(FLASHATTENTION_DISABLE_HDIM64)
    endif()

    if(DISABLE_HDIM96)
      add_compile_definitions(FLASHATTENTION_DISABLE_HDIM96)
    endif()

    if(DISABLE_HDIM128)
      add_compile_definitions(FLASHATTENTION_DISABLE_HDIM128)
    endif()

    if(DISABLE_HDIM192)
      add_compile_definitions(FLASHATTENTION_DISABLE_HDIM192)
    endif()

    if(DISABLE_HDIM256)
      add_compile_definitions(FLASHATTENTION_DISABLE_HDIM256)
    endif()

    if(DISABLE_SPLIT)
      add_compile_definitions(FLASHATTENTION_DISABLE_SPLIT)
    endif()

    if(DISABLE_PAGEDKV)
      add_compile_definitions(FLASHATTENTION_DISABLE_PAGEDKV)
    endif()

    if(DISABLE_SOFTCAP)
      add_compile_definitions(FLASHATTENTION_DISABLE_SOFTCAP)
    endif()

    if(DISABLE_PACKGQA)
      add_compile_definitions(FLASHATTENTION_DISABLE_PACKGQA)
    endif()

    if(DISABLE_BACKWARD)
      add_compile_definitions(FLASHATTENTION_DISABLE_BACKWARD)
    endif()

    if(DISABLE_SM8X)
      add_compile_definitions(FLASHATTENTION_DISABLE_SM8X)
    endif()

    set(DTYPE_FWD_SM80 "bf16")
    if(NOT DISABLE_FP16)
        list(APPEND DTYPE_FWD_SM80 "fp16")
    endif()

    set(DTYPE_FWD_SM90 "bf16")
    if(NOT DISABLE_FP16)
        list(APPEND DTYPE_FWD_SM90 "fp16")
    endif()
    if(NOT DISABLE_FP8)
        list(APPEND DTYPE_FWD_SM90 "e4m3")
    endif()
    
    set(DTYPE_BWD "bf16")
    if(NOT DISABLE_FP16)
        list(APPEND DTYPE_BWD "fp16")
    endif()

    set(HEAD_DIMENSIONS_BWD)
    if(NOT DISABLE_HDIM64)
        list(APPEND HEAD_DIMENSIONS_BWD 64)
    endif()
    if(NOT DISABLE_HDIM96)
        list(APPEND HEAD_DIMENSIONS_BWD 96)
    endif()
    if(NOT DISABLE_HDIM128)
        list(APPEND HEAD_DIMENSIONS_BWD 128)
    endif()
    if(NOT DISABLE_HDIM192)
        list(APPEND HEAD_DIMENSIONS_BWD 192)
    endif()
    if(NOT DISABLE_HDIM256)
        list(APPEND HEAD_DIMENSIONS_BWD 256)
    endif()

    # set(HEAD_DIMENSIONS_FWD "all" "diff")
    set(HEAD_DIMENSIONS_FWD "all" "diff")
    set(HEAD_DIMENSIONS_FWD_SM80 ${HEAD_DIMENSIONS_BWD})
    
    set(SPLIT "__EMPTY__")
    if(NOT DISABLE_SPLIT)
        list(APPEND SPLIT "_split")
    endif()
    
    set(PAGEDKV "__EMPTY__")
    if(NOT DISABLE_PAGEDKV)
        list(APPEND PAGEDKV "_paged")
    endif()
    
    set(SOFTCAP "__EMPTY__")
    if(NOT DISABLE_SOFTCAP)
        list(APPEND SOFTCAP "_softcap")
    endif()
    
    set(SOFTCAP_ALL)
    if(DISABLE_SOFTCAP)
        set(SOFTCAP_ALL "__EMPTY__")
    else()
        set(SOFTCAP_ALL "_softcapall")
    endif()
    
    set(PACKGQA "__EMPTY__")
    if(NOT DISABLE_PACKGQA)
        list(APPEND PACKGQA "_packgqa")
    endif()

    set(sources_fwd_sm80)
    foreach(hdim ${HEAD_DIMENSIONS_FWD_SM80})
        foreach(dtype ${DTYPE_FWD_SM80})
            foreach(split ${SPLIT})
                foreach(paged ${PAGEDKV})
                    foreach(softcap ${SOFTCAP_ALL})
                        set(name "flash_attn_v3/instantiations/flash_fwd_hdim${hdim}_${dtype}${paged}${split}${softcap}_sm80.cu")
                        string(REPLACE "__EMPTY__" "" refine_name "${name}")
                        list(APPEND sources_fwd_sm80 "${refine_name}")
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    
    set(sources_fwd_sm90)
    foreach(hdim ${HEAD_DIMENSIONS_FWD})
        foreach(dtype ${DTYPE_FWD_SM90})
            foreach(split ${SPLIT})
                foreach(paged ${PAGEDKV})
                    foreach(softcap ${SOFTCAP})
                        foreach(packgqa ${PACKGQA})
                            if(packgqa STREQUAL "__EMPTY__" OR (paged STREQUAL "__EMPTY__" AND split STREQUAL "__EMPTY__"))
                                set(name "flash_attn_v3/instantiations/flash_fwd_hdim${hdim}_${dtype}${paged}${split}${softcap}${packgqa}_sm90.cu")
                                string(REPLACE "__EMPTY__" "" refine_name "${name}")
                                list(APPEND sources_fwd_sm90 "${refine_name}")
                            endif()
                        endforeach()
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()

    message(STATUS "wsm debug ${sources_fwd_sm90}")
    
    set(sources_bwd_sm80)
    foreach(hdim ${HEAD_DIMENSIONS_BWD})
        foreach(dtype ${DTYPE_BWD})
            foreach(softcap ${SOFTCAP})
                set(name "flash_attn_v3/instantiations/flash_bwd_hdim${hdim}_${dtype}${softcap}_sm80.cu")
                string(REPLACE "__EMPTY__" "" refine_name "${name}")
                list(APPEND sources_bwd_sm80 "${refine_name}")
            endforeach()
        endforeach()
    endforeach()
    
    set(sources_bwd_sm90)
    foreach(hdim ${HEAD_DIMENSIONS_BWD})
        foreach(dtype ${DTYPE_BWD})
            foreach(softcap ${SOFTCAP_ALL})
                set(name "flash_attn_v3/instantiations/flash_bwd_hdim${hdim}_${dtype}${softcap}_sm90.cu")
                string(REPLACE "__EMPTY__" "" refine_name "${name}")
                list(APPEND sources_bwd_sm90 "${refine_name}")
            endforeach()
        endforeach()
    endforeach()

    if(DISABLE_BACKWARD)
        set(sources_bwd_sm80 "")
        set(sources_bwd_sm90 "")
    endif()
    
    set(FA3_SOURCES_CU_SOURCES "flash_attn_v3/flash_api.cu")
    if(NOT DISABLE_SM8X)
        list(APPEND FA3_SOURCES_CU_SOURCES ${sources_fwd_sm80})
    endif()
    list(APPEND FA3_SOURCES_CU_SOURCES ${sources_fwd_sm90})
    if(NOT DISABLE_SM8X)
        list(APPEND FA3_SOURCES_CU_SOURCES ${sources_bwd_sm80})
    endif()
    list(APPEND FA3_SOURCES_CU_SOURCES ${sources_bwd_sm90})
    
    if(NOT DISABLE_SPLIT)
        list(APPEND FA3_SOURCES_CU_SOURCES "flash_attn_v3/flash_fwd_combine.cu")
    endif()
    
    list(APPEND FA3_SOURCES_CU_SOURCES "flash_attn_v3/flash_prepare_scheduler.cu")

    # file(GLOB FA3_SOURCES_CU_SOURCES "flash_attn_v3/instantiations/*_sm90.cu")
    # file(GLOB FA3_SOURCES_CU_SOURCES "flash_attn_v3/instantiations/flash_fwd_hdim*_sm90.cu")
    # file(GLOB FA3_SOURCES_CU_SOURCES "flash_attn_v3/instantiations/*_softcapall_sm90.cu")
    message(STATUS "Auto generated CUDA source files: ${FA3_SOURCES_CU_SOURCES}")
    add_library(flashattnv3 SHARED
        # flash_attn_v3/flash_api.cu
        # flash_attn_v3/flash_fwd_combine.cu
        # flash_attn_v3/flash_prepare_scheduler.cu
        ${FA3_SOURCES_CU_SOURCES}
      )
  
    target_include_directories(flashattnv3 PRIVATE
        flash_attn_v3
        flash_attn_v3/cutlass/include
      )
  
    target_compile_options(flashattnv3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      # -w
      # -Wno-abi
      -Xcompiler="-fPIC"
      -Xcompiler="-O3"
      -std=c++17
      --ftemplate-backtrace-limit=0
      --use_fast_math
      --resource-usage
      -lineinfo
      -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED # Necessary for the WGMMA shapes that we use
      -DCUTLASS_ENABLE_GDC_FOR_SM90 # For PDL
      -DCUTLASS_DEBUG_TRACE_LEVEL=0 # Can toggle for debugging
      -DNDEBUG # Important, otherwise performance is severely impacted
      -gencode arch=compute_90a,code=sm_90a

      # -U__CUDA_NO_HALF_OPERATORS__
      # -U__CUDA_NO_HALF_CONVERSIONS__
      # -U__CUDA_NO_BFLOAT16_OPERATORS__
      # -U__CUDA_NO_BFLOAT16_CONVERSIONS__
      # -U__CUDA_NO_BFLOAT162_OPERATORS__
      # -U__CUDA_NO_BFLOAT162_CONVERSIONS__
      --expt-relaxed-constexpr
      # --expt-extended-lambda
      >)
  
    INSTALL(TARGETS flashattnv3
        LIBRARY DESTINATION "lib")
  
    INSTALL(FILES flash_attn_v3/flash_api.h DESTINATION "include" RENAME flashv3_api.h)
  endif()

else()
  INSTALL(FILES capi/flash_attn.h DESTINATION "include")
  INSTALL(FILES flash_attn_v3/flash_api.h DESTINATION "include" RENAME flashv3_api.h)

endif()
#SKIP_BUILD_FA
