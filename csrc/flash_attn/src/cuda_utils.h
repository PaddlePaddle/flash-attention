#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "fmha_utils.h"

int GetCurrentDeviceId();

static int GetCudaDeviceCount();

cudaDeviceProp* GetDeviceProperties(int id);
