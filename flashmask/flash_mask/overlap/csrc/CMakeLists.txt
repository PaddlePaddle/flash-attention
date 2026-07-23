# FM-4 Overlap bridge: standalone sub-CMake.
#
# This is invoked by flashmask/setup.py via subprocess (cmake configure + build),
# NOT by the parent FA-v2 build graph. It compiles three TUs into one shared lib:
#   - fm4_overlap_bridge.cu   (new: extern "C" surface)
#   - overlap_comm.cu         (reused from FM-3, unchanged)
#   - sep_sr_buffer.cu        (reused from FM-3, unchanged)
# and statically links the prebuilt NVSHMEM combined archive (libnvshmem.a) plus
# the UID bootstrap host .so. The recipe (rdc / device-link / symbol isolation /
# rpath) is copied verbatim from csrc/flashmask_v2/distributed/CMakeLists.txt and
# the symbol-isolation block at csrc/CMakeLists.txt:688-704.
#
# 3.18 is the floor: CUDA_ARCHITECTURES accepting arch strings like "90a"/"100a"
# landed in 3.18. The dev box has 3.31.
cmake_minimum_required(VERSION 3.18)
project(fm4_overlap LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Inputs (all passed by setup.py via -D...). No absolute paths baked in here:
# every path is either derived from CMAKE_CURRENT_SOURCE_DIR (relative to this
# file) or supplied by the caller. This keeps the tree relocatable across the
# A100 dev box and the H100/B200 build boxes.
# ---------------------------------------------------------------------------
set(FM4_OVERLAP_CUDA_ARCH "90a" CACHE STRING "target sm arch, e.g. 90a or 100a")
set(NVSHMEM_INSTALL_DIR "" CACHE PATH "nvshmem install dir (include/ + lib/)")
set(FM4_OVERLAP_CUTLASS_INC "" CACHE PATH "cutlass include dir (must contain cutlass/bfloat16.h)")

if(NOT NVSHMEM_INSTALL_DIR)
    message(FATAL_ERROR "NVSHMEM_INSTALL_DIR is required (e.g. Paddle's third_party/install/nvshmem). "
                        "Set it via -DNVSHMEM_INSTALL_DIR=... or the NVSHMEM_HOME env in setup.py.")
endif()
if(NOT FM4_OVERLAP_CUTLASS_INC)
    message(FATAL_ERROR "FM4_OVERLAP_CUTLASS_INC is required (dir containing cutlass/bfloat16.h).")
endif()

# Derive compute_ capability: gencode 'compute_' must drop the trailing 'a'
# (compute_90a is illegal; the correct pair is arch=compute_90,code=sm_90a).
string(REGEX REPLACE "a$" "" FM4_OVERLAP_CUDA_ARCH_NUM "${FM4_OVERLAP_CUDA_ARCH}")

# ---------------------------------------------------------------------------
# Source locations (relative to this CMakeLists.txt).
#   this file lives at: flashmask/flash_mask/overlap/csrc/CMakeLists.txt
#   FM-3 distributed dir is four levels up then into csrc/flashmask_v2/distributed:
#     overlap/csrc -> overlap -> flash_mask -> flashmask -> flash-attention
# ---------------------------------------------------------------------------
get_filename_component(DIST_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../csrc/flashmask_v2/distributed" ABSOLUTE)

# Fail loudly now (configure time) rather than with a cryptic nvcc error later.
foreach(_req
        "${DIST_DIR}/overlap_comm.cu"
        "${DIST_DIR}/sep_sr_buffer.cu"
        "${DIST_DIR}/overlap_comm.cuh")
    if(NOT EXISTS "${_req}")
        message(FATAL_ERROR "Required FM-3 source not found: ${_req}\n"
                            "Check the relative path from overlap/csrc, or that the "
                            "flash-attention tree is fully checked out.")
    endif()
endforeach()

if(NOT EXISTS "${FM4_OVERLAP_CUTLASS_INC}/cutlass/bfloat16.h")
    message(FATAL_ERROR "cutlass/bfloat16.h not found under FM4_OVERLAP_CUTLASS_INC="
                        "${FM4_OVERLAP_CUTLASS_INC}\n"
                        "If you rely on the FA4 submodule, run: "
                        "git submodule update --init --recursive")
endif()

if(NOT EXISTS "${NVSHMEM_INSTALL_DIR}/include/nvshmem.h")
    message(FATAL_ERROR "nvshmem.h not found under NVSHMEM_INSTALL_DIR=${NVSHMEM_INSTALL_DIR}")
endif()

# ---------------------------------------------------------------------------
# Self-built nvshmem IMPORTED target (copied from csrc/CMakeLists.txt:631-649).
# We do NOT depend on the parent FA-v2 build graph, so we cannot rely on its
# `nvshmem` target existing; we re-create it here. This also sidesteps the
# `if(NOT TARGET nvshmem) FATAL_ERROR` guard in distributed/CMakeLists.txt.
# ---------------------------------------------------------------------------
set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_INSTALL_DIR}/include")
set(NVSHMEM_LIB "${NVSHMEM_INSTALL_DIR}/lib/libnvshmem.a")
set(NVSHMEM_BOOTSTRAP_UID_LIB "${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_uid.so.3")

if(NOT EXISTS "${NVSHMEM_LIB}")
    message(FATAL_ERROR "libnvshmem.a not found at ${NVSHMEM_LIB}")
endif()
if(NOT EXISTS "${NVSHMEM_BOOTSTRAP_UID_LIB}")
    message(FATAL_ERROR "nvshmem_bootstrap_uid.so.3 not found at ${NVSHMEM_BOOTSTRAP_UID_LIB}")
endif()

add_library(nvshmem STATIC IMPORTED GLOBAL)
set_property(TARGET nvshmem PROPERTY IMPORTED_LOCATION ${NVSHMEM_LIB})

# ---------------------------------------------------------------------------
# The bridge shared library.
# ---------------------------------------------------------------------------
add_library(fm4_overlap SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/fm4_overlap_bridge.cu
    ${DIST_DIR}/overlap_comm.cu
    ${DIST_DIR}/sep_sr_buffer.cu
)

target_compile_options(fm4_overlap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-fPIC
    -Xcompiler=-O3
    -std=c++17
    --use_fast_math
    -gencode arch=compute_${FM4_OVERLAP_CUDA_ARCH_NUM},code=sm_${FM4_OVERLAP_CUDA_ARCH}
    -lineinfo
    -DNDEBUG
    --expt-relaxed-constexpr
    -DNVSHMEM_DISTRIBUTED_OVERLAP
>)

target_include_directories(fm4_overlap PRIVATE
    ${NVSHMEM_INCLUDE_DIR}
    ${FM4_OVERLAP_CUTLASS_INC}
    ${DIST_DIR}                       # overlap_comm.cuh, sr_buffer.cuh, ...
    ${CMAKE_CURRENT_SOURCE_DIR}       # fm4_overlap_bridge.cu's own dir
)

target_link_libraries(fm4_overlap PRIVATE
    CUDA::cudart
    nvshmem
    ${NVSHMEM_BOOTSTRAP_UID_LIB}
)

# Symbol isolation (copied from csrc/CMakeLists.txt:688-704). Export only the
# fm4_overlap_* C-ABI; localize everything else so the statically-linked NVSHMEM
# / cutlass internals do not interpose with libpaddle.so or DeepEP's NVSHMEM.
target_link_options(fm4_overlap PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/fm4_overlap.map
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

set_target_properties(fm4_overlap PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON     # = -rdc=true (NVSHMEM device symbols need it)
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    # = device-link, avoids runtime PTX JIT
    CUDA_ARCHITECTURES "${FM4_OVERLAP_CUDA_ARCH}"
    # $ORIGIN: the bridge .so sits next to nothing it needs except itself; the
    # NVSHMEM lib dir is needed at runtime for the bootstrap/transport host .so's
    # that libnvshmem.a dlopen's. Both are on the rpath.
    BUILD_RPATH "$ORIGIN;${NVSHMEM_INSTALL_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;${NVSHMEM_INSTALL_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH ON
)

