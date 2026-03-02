cmake_minimum_required(VERSION 3.9)

find_package(CUDAToolkit REQUIRED)

# Make sure add_subdirectory is called later than include(distributed/cmake/nvshmem), so that nvshmem external proj is visible
if(NOT TARGET nvshmem)
    message(FATAL_ERROR "NVSHMEM target not found! Please ensure NVSHMEM is properly configured.")
endif()

set(DISTRIBUTED_SOURCES
    overlap_comm.cu
    sep_sr_buffer.cu
)

add_library(flashmaskv2_distributed STATIC
    ${DISTRIBUTED_SOURCES}
)

target_compile_options(flashmaskv2_distributed PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler="-fPIC"
    -Xcompiler="-O3"
    -std=c++17
    --use_fast_math
    -arch=sm_90
    -lineinfo
    -DNDEBUG
    --expt-relaxed-constexpr
    >)

target_include_directories(flashmaskv2_distributed PRIVATE
    ${NVSHMEM_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include          # include cutlass
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(flashmaskv2_distributed PRIVATE 
    CUDA::cudart 
    CUDA::cuda_driver 
    nvshmem
    ${NVSHMEM_BOOTSTRAP_UID_LIB}        # using MPI_Init, mpi bootstrap
)

if(TARGET extern_nvshmem)
    # There is no extern_nvshmem in non-standalone building
    add_dependencies(flashmaskv2_distributed extern_nvshmem)
endif()

set_target_properties(flashmaskv2_distributed PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)