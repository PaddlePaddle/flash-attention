cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(flash-attention LANGUAGES CXX CUDA)

find_package(Git REQUIRED)

# 会触发所有submodule的下载
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

  if(WITH_FLASHATTN_V3) 
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" flashmask_attention_v3/generate_kernels.py -o flashmask_attention_v3/instantiations
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    option(DISABLE_FLASHMASK_V3_FP16 "Disable FP16 for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_FP8 "Disable FP8 for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_HDIM64 "Disable HDIM64 for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_HDIM96 "Disable HDIM96 for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_HDIM128 "Disable HDIM128 for flashmask_v3" OFF)
    option(DISABLE_FLASHMASK_V3_HDIM192 "Disable HDIM192 for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_HDIM256 "Disable HDIM256 for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_SPLIT "Disable Split for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_PAGEDKV "Disable PagedKV for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_SOFTCAP "Disable Softcap for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_PACKGQA "Disable PackGQA for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_BACKWARD "Disable Backward for flashmask_v3" ON)
    option(DISABLE_FLASHMASK_V3_SM8X "Disable SM8x for flashmask_v3" ON)

    if(DISABLE_FLASHMASK_V3_FP16)
      add_compile_definitions(FLASHMASK_V3_DISABLE_FP16)
    endif()

    if(DISABLE_FLASHMASK_V3_FP8)
      add_compile_definitions(FLASHMASK_V3_DISABLE_FP8)
    endif()

    if(DISABLE_FLASHMASK_V3_HDIM64)
      add_compile_definitions(FLASHMASK_V3_DISABLE_HDIM64)
    endif()

    if(DISABLE_FLASHMASK_V3_HDIM96)
      add_compile_definitions(FLASHMASK_V3_DISABLE_HDIM96)
    endif()

    if(DISABLE_FLASHMASK_V3_HDIM128)
      add_compile_definitions(FLASHMASK_V3_DISABLE_HDIM128)
    endif()

    if(DISABLE_FLASHMASK_V3_HDIM192)
      add_compile_definitions(FLASHMASK_V3_DISABLE_HDIM192)
    endif()

    if(DISABLE_FLASHMASK_V3_HDIM256)
      add_compile_definitions(FLASHMASK_V3_DISABLE_HDIM256)
    endif()

    if(DISABLE_FLASHMASK_V3_SPLIT)
      add_compile_definitions(FLASHMASK_V3_DISABLE_SPLIT)
    endif()

    if(DISABLE_FLASHMASK_V3_PAGEDKV)
      add_compile_definitions(FLASHMASK_V3_DISABLE_PAGEDKV)
    endif()

    if(DISABLE_FLASHMASK_V3_SOFTCAP)
      add_compile_definitions(FLASHMASK_V3_DISABLE_SOFTCAP)
    endif()

    if(DISABLE_FLASHMASK_V3_PACKGQA)
      add_compile_definitions(FLASHMASK_V3_DISABLE_PACKGQA)
    endif()

    if(DISABLE_FLASHMASK_V3_BACKWARD)
      add_compile_definitions(FLASHMASK_V3_DISABLE_BACKWARD)
    endif()

    if(DISABLE_FLASHMASK_V3_SM8X)
      add_compile_definitions(FLASHMASK_V3_DISABLE_SM8X)
    endif()

    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Generating flashmask_v3 Python script execution failed with exit code ${result}: ${error}")
    endif()

    # 以 flashmask_v3 为前缀的变量逻辑
    set(FLASHMASKV3_DTYPE_FWD_SM80 "bf16")
    if(NOT DISABLE_FLASHMASK_V3_FP16)
        list(APPEND FLASHMASKV3_DTYPE_FWD_SM80 "fp16")
    endif()

    set(FLASHMASKV3_DTYPE_FWD_SM90 "bf16")
    if(NOT DISABLE_FLASHMASK_V3_FP16)
        list(APPEND FLASHMASKV3_DTYPE_FWD_SM90 "fp16")
    endif()
    if(NOT DISABLE_FLASHMASK_V3_FP8)
        list(APPEND FLASHMASKV3_DTYPE_FWD_SM90 "e4m3")
    endif()

    set(FLASHMASKV3_DTYPE_BWD "bf16")
    if(NOT DISABLE_FLASHMASK_V3_FP16)
        list(APPEND FLASHMASKV3_DTYPE_BWD "fp16")
    endif()

    set(FLASHMASKV3_HEAD_DIMENSIONS_BWD)
    if(NOT DISABLE_FLASHMASK_V3_HDIM64)
        list(APPEND FLASHMASKV3_HEAD_DIMENSIONS_BWD 64)
    endif()
    if(NOT DISABLE_FLASHMASK_V3_HDIM96)
        list(APPEND FLASHMASKV3_HEAD_DIMENSIONS_BWD 96)
    endif()
    if(NOT DISABLE_FLASHMASK_V3_HDIM128)
        list(APPEND FLASHMASKV3_HEAD_DIMENSIONS_BWD 128)
    endif()
    if(NOT DISABLE_FLASHMASK_V3_HDIM192)
        list(APPEND FLASHMASKV3_HEAD_DIMENSIONS_BWD 192)
    endif()
    if(NOT DISABLE_FLASHMASK_V3_HDIM256)
        list(APPEND FLASHMASKV3_HEAD_DIMENSIONS_BWD 256)
    endif()

    # Disable diff, not support headdim != headdim_v
    set(FLASHMASKV3_HEAD_DIMENSIONS_FWD ${FLASHMASKV3_HEAD_DIMENSIONS_BWD})
    set(FLASHMASKV3_HEAD_DIMENSIONS_FWD_SM80 ${FLASHMASKV3_HEAD_DIMENSIONS_BWD})

    set(FLASHMASKV3_SPLIT "__EMPTY__")
    if(NOT DISABLE_FLASHMASK_V3_SPLIT)
        list(APPEND FLASHMASKV3_SPLIT "_split")
    endif()

    set(FLASHMASKV3_PAGEDKV "__EMPTY__")
    if(NOT DISABLE_FLASHMASK_V3_PAGEDKV)
        list(APPEND FLASHMASKV3_PAGEDKV "_paged")
    endif()

    set(FLASHMASKV3_SOFTCAP "__EMPTY__")
    if(NOT DISABLE_FLASHMASK_V3_SOFTCAP)
        list(APPEND FLASHMASKV3_SOFTCAP "_softcap")
    endif()

    set(FLASHMASKV3_SOFTCAP_ALL)
    if(DISABLE_FLASHMASK_V3_SOFTCAP)
        set(FLASHMASKV3_SOFTCAP_ALL "__EMPTY__")
    else()
        set(FLASHMASKV3_SOFTCAP_ALL "_softcapall")
    endif()

    set(FLASHMASKV3_PACKGQA "__EMPTY__")
    if(NOT DISABLE_FLASHMASK_V3_PACKGQA)
        list(APPEND FLASHMASKV3_PACKGQA "_packgqa")
    endif()

    set(flashmaskv3_sources_fwd_sm80)
    foreach(hdim ${FLASHMASKV3_HEAD_DIMENSIONS_FWD_SM80})
        foreach(dtype ${FLASHMASKV3_DTYPE_FWD_SM80})
            foreach(split ${FLASHMASKV3_SPLIT})
                foreach(paged ${FLASHMASKV3_PAGEDKV})
                    foreach(softcap ${FLASHMASKV3_SOFTCAP_ALL})
                        set(name "flashmask_attention_v3/instantiations/flash_fwd_hdim${hdim}_${dtype}${paged}${split}${softcap}_sm80.cu")
                        string(REPLACE "__EMPTY__" "" refine_name "${name}")
                        list(APPEND flashmaskv3_sources_fwd_sm80 "${refine_name}")
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    
    set(flashmaskv3_sources_fwd_sm90)
    foreach(hdim ${FLASHMASKV3_HEAD_DIMENSIONS_FWD})
        foreach(dtype ${FLASHMASKV3_DTYPE_FWD_SM90})
            foreach(split ${FLASHMASKV3_SPLIT})
                foreach(paged ${FLASHMASKV3_PAGEDKV})
                    foreach(softcap ${FLASHMASKV3_SOFTCAP})
                        foreach(packgqa ${FLASHMASKV3_PACKGQA})
                            if(packgqa STREQUAL "__EMPTY__" OR (paged STREQUAL "__EMPTY__" AND split STREQUAL "__EMPTY__"))
                                set(name "flashmask_attention_v3/instantiations/flash_fwd_hdim${hdim}_${dtype}${paged}${split}${softcap}${packgqa}_sm90.cu")
                                string(REPLACE "__EMPTY__" "" refine_name "${name}")
                                list(APPEND flashmaskv3_sources_fwd_sm90 "${refine_name}")
                            endif()
                        endforeach()
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()

    set(flashmaskv3_sources_bwd_sm80)
    foreach(hdim ${FLASHMASKV3_HEAD_DIMENSIONS_BWD})
        foreach(dtype ${FLASHMASKV3_DTYPE_BWD})
            foreach(softcap ${FLASHMASKV3_SOFTCAP})
                set(name "flashmask_attention_v3/instantiations/flash_bwd_hdim${hdim}_${dtype}${softcap}_sm80.cu")
                string(REPLACE "__EMPTY__" "" refine_name "${name}")
                list(APPEND flashmaskv3_sources_bwd_sm80 "${refine_name}")
            endforeach()
        endforeach()
    endforeach()

    set(flashmaskv3_sources_bwd_sm90)
    foreach(hdim ${FLASHMASKV3_HEAD_DIMENSIONS_BWD})
        foreach(dtype ${FLASHMASKV3_DTYPE_BWD})
            foreach(softcap ${FLASHMASKV3_SOFTCAP_ALL})
                foreach(causal IN ITEMS "" "_causal")
                    foreach(determ IN ITEMS "" "_determ")
                        set(name "flashmask_attention_v3/instantiations/flash_bwd_hdim${hdim}_${dtype}${causal}${determ}${softcap}_sm90.cu")
                        string(REPLACE "__EMPTY__" "" refine_name "${name}")
                        list(APPEND flashmaskv3_sources_bwd_sm90 "${refine_name}")
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()

    if(DISABLE_FLASHMASK_V3_BACKWARD)
        set(flashmaskv3_sources_bwd_sm80 "")
        set(flashmaskv3_sources_bwd_sm90 "")
    endif()


    set(FLASHMASKV3_SOURCES_CU_SOURCES "flashmask_attention_v3/flash_api.cu")
    if(NOT DISABLE_FLASHMASK_V3_SM8X)
        list(APPEND FLASHMASKV3_SOURCES_CU_SOURCES ${flashmaskv3_sources_fwd_sm80})
    endif()
    list(APPEND FLASHMASKV3_SOURCES_CU_SOURCES ${flashmaskv3_sources_fwd_sm90})
    if(NOT DISABLE_FLASHMASK_V3_SM8X)
        list(APPEND FLASHMASKV3_SOURCES_CU_SOURCES ${flashmaskv3_sources_bwd_sm80})
    endif()
    list(APPEND FLASHMASKV3_SOURCES_CU_SOURCES ${flashmaskv3_sources_bwd_sm90})

    if(NOT DISABLE_FLASHMASK_V3_SPLIT)
        list(APPEND FLASHMASKV3_SOURCES_CU_SOURCES "flashmask_attention_v3/flash_fwd_combine.cu")
    endif()

    list(APPEND FLASHMASKV3_SOURCES_CU_SOURCES "flashmask_attention_v3/flash_prepare_scheduler.cu")
    message(STATUS "Auto generated CUDA source files for flashmask_v3: ${FLASHMASKV3_SOURCES_CU_SOURCES}")

    # 3. 添加动态库
    add_library(flashmaskv3 SHARED
        ${FLASHMASKV3_SOURCES_CU_SOURCES}
    )

    target_include_directories(flashmaskv3 PRIVATE
        flashmask_attention_v3
        flashmask_attention_v3/cutlass/include
    )

    target_compile_options(flashmaskv3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
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
      --expt-relaxed-constexpr
      >)

    INSTALL(TARGETS flashmaskv3 LIBRARY DESTINATION "lib")

    INSTALL(FILES flashmask_attention_v3/flash_api.h DESTINATION "include" RENAME flashmaskv3_api.h)

  endif()

else()
  INSTALL(FILES flashmask_attention_v3/flash_api.h DESTINATION "include" RENAME flashmaskv3_api.h)

endif()
#SKIP_BUILD_FA


