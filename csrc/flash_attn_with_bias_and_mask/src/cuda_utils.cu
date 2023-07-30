#include "cuda_utils.h"

#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>

static std::once_flag g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<cudaDeviceProp> g_device_props;

static int GetCurrentDeviceId() {
  int device_id;
  FMHA_CHECK_CUDA(cudaGetDevice(&device_id));
  return device_id;
}

static int GetCudaDeviceCount() {
  int count;
  FMHA_CHECK_CUDA(cudaGetDeviceCount(&count));
  return count;
}

cudaDeviceProp* GetDeviceProperties(int id) {
  std::call_once(g_device_props_size_init_flag, [&] {
    int gpu_num = 0;
    gpu_num = GetCudaDeviceCount();
    g_device_props_init_flags.resize(gpu_num);
    g_device_props.resize(gpu_num);
    for (int i = 0; i < gpu_num; ++i) {
      g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
    }
  });

  if (id == -1) {
    id = GetCurrentDeviceId();
  }

  std::call_once(*(g_device_props_init_flags[id]), [&] {
        FMHA_CHECK_CUDA(cudaGetDeviceProperties(&g_device_props[id], id));
  });

  return &g_device_props[id];
}

