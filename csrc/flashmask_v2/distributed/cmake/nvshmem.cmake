# Standalone NVSHMEM building: the following code is adapted from
# Paddle nvshmem.cmake and will only be used when FlashMask is built
# outside of the scope of Paddle third_party, when we specify `WITH_DISTRIBUTED_OVERLAP` manually.

include(ExternalProject)

set(GDRCOPY_HOME
    $ENV{GDRCOPY_HOME}
    CACHE PATH "Path to GDRCOPY installation")
if(GDRCOPY_HOME)
  message(STATUS "GDRCOPY_HOME: ${GDRCOPY_HOME}")
else()
  message(
    WARNING
      "Setting GDRCOPY_HOME environment or cmake option maybe needed to specify your install path GDRCOPY."
  )
endif()

set(NVSHMEM_INSTALL_DIR ${CMAKE_BINARY_DIR}/install/nvshmem)
set(NVSHMEM_PREFIX_DIR ${CMAKE_BINARY_DIR}/nvshmem)
set(NVSHMEM_SOURCE_DIR ${NVSHMEM_PREFIX_DIR}/src/extern_nvshmem)
message(STATUS "NVSHMEM_INSTALL_DIR: ${NVSHMEM_INSTALL_DIR}")

set(NVSHMEM_INCLUDE_DIR
    "${NVSHMEM_INSTALL_DIR}/include"
    CACHE PATH "nvshmem include directory." FORCE)

include_directories(${NVSHMEM_INCLUDE_DIR})

set(NVSHMEM_TAR_NAME "nvshmem_src_3.2.5-1.txz")

if(NVSHMEM_SRC_TAR_PATH)
  set(NVSHMEM_DOWNLOAD_COMMAND
      rm -rf extern_nvshmem ${NVSHMEM_TAR_NAME} && cp ${NVSHMEM_SRC_TAR_PATH} .
      && tar xf ${NVSHMEM_TAR_NAME} && mv nvshmem_src extern_nvshmem)
else()
  set(NVSHMEM_URL
      "https://paddle-ci.gz.bcebos.com/${NVSHMEM_TAR_NAME}"
      CACHE STRING "" FORCE)
  set(NVSHMEM_DOWNLOAD_COMMAND
      rm -rf extern_nvshmem ${NVSHMEM_TAR_NAME} && wget --no-check-certificate
      -q ${NVSHMEM_URL} && tar xf ${NVSHMEM_TAR_NAME} && mv nvshmem_src
      extern_nvshmem)
endif()

# get current directory
get_filename_component(NVSHMEM_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

if(CUDA_VERSION VERSION_GREATER_EQUAL 13)
  set(NVSHMEM_PATCH_NAME "nvshmem_cuda13.patch")
else()
  set(NVSHMEM_PATCH_NAME "nvshmem.patch")
endif()

set(NVSHMEM_PATCH_PATH "${NVSHMEM_CMAKE_DIR}/${NVSHMEM_PATCH_NAME}")
set(NVSHMEM_PLUGIN_ISOLATION_PATCH "${NVSHMEM_CMAKE_DIR}/nvshmem_plugin_isolation.patch")

if(EXISTS "${NVSHMEM_PATCH_PATH}")
  message(STATUS "Found NVSHMEM patch: ${NVSHMEM_PATCH_PATH}")
  set(NVSHMEM_PATCH_COMMAND
      patch -p1 < "${NVSHMEM_PATCH_PATH}" || echo "Patch applied or already applied"
  )
else()
  message(STATUS "NVSHMEM patch not found at ${NVSHMEM_PATCH_PATH}, skipping patching")
  set(NVSHMEM_PATCH_COMMAND "")
endif()

if(EXISTS "${NVSHMEM_PLUGIN_ISOLATION_PATCH}")
  message(STATUS "Found NVSHMEM plugin isolation patch: ${NVSHMEM_PLUGIN_ISOLATION_PATCH}")
  if(NVSHMEM_PATCH_COMMAND)
    set(NVSHMEM_PATCH_COMMAND
        ${NVSHMEM_PATCH_COMMAND} && patch --forward -p1 < "${NVSHMEM_PLUGIN_ISOLATION_PATCH}" || echo "Plugin isolation patch applied or already applied"
    )
  else()
    set(NVSHMEM_PATCH_COMMAND
        patch --forward -p1 < "${NVSHMEM_PLUGIN_ISOLATION_PATCH}" || echo "Plugin isolation patch applied or already applied"
    )
  endif()
endif()

if(NOT NVSHMEM_PATCH_COMMAND)
  set(NVSHMEM_PATCH_COMMAND "")
endif()

set(NVSHMEM_LIB ${NVSHMEM_INSTALL_DIR}/lib/libnvshmem.a)
set(NVSHMEM_BOOTSTRAP_MPI_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_mpi.so.3)
set(NVSHMEM_BOOTSTRAP_UID_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_uid.so.3)
set(NVSHMEM_BOOTSTRAP_PMI_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_pmi.so.3)
set(NVSHMEM_BOOTSTRAP_PMI2_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_pmi2.so.3)
set(NVSHMEM_TRANSPORT_IBRC_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_transport_ibrc.so.3)
set(NVSHMEM_TRANSPORT_IBGDA_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_transport_ibgda.so.3)

# only compile nvshmem for sm90
set(CUDA_ARCHITECTURES "90")

ExternalProject_Add(
  extern_nvshmem
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  PREFIX ${NVSHMEM_PREFIX_DIR}
  SOURCE_DIR ${NVSHMEM_SOURCE_DIR}
  BUILD_IN_SOURCE FALSE                                 # do not create extern_nvshmem-build for install
  BUILD_ALWAYS FALSE
  DOWNLOAD_DIR ${NVSHMEM_PREFIX_DIR}/src
  DOWNLOAD_COMMAND ${NVSHMEM_DOWNLOAD_COMMAND}
  PATCH_COMMAND ${NVSHMEM_PATCH_COMMAND}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${NVSHMEM_INSTALL_DIR}
             -DGDRCOPY_HOME:PATH=${GDRCOPY_HOME}
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
             -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING=0
             -DNVSHMEM_SHMEM_SUPPORT=0
             -DNVSHMEM_UCX_SUPPORT=0
             -DNVSHMEM_USE_NCCL=0
             -DNVSHMEM_IBGDA_SUPPORT=1
             -DNVSHMEM_PMIX_SUPPORT=0
             -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0
             -DNVSHMEM_USE_GDRCOPY=1
             -DNVSHMEM_IBRC_SUPPORT=1
             -DNVSHMEM_BUILD_TESTS=0
             -DNVSHMEM_BUILD_EXAMPLES=0
             -DNVSHMEM_MPI_SUPPORT=1                # We need MPI (mpirun and within-code initialization)
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${NVSHMEM_INSTALL_DIR}
  BUILD_BYPRODUCTS ${NVSHMEM_LIB})

add_library(nvshmem STATIC IMPORTED GLOBAL)
set_property(TARGET nvshmem PROPERTY IMPORTED_LOCATION ${NVSHMEM_LIB})
add_dependencies(nvshmem extern_nvshmem)
