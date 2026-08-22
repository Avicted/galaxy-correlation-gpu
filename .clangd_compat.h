// clangd compatibility header: stubs for HIP builtins not visible outside
// __HIP__ compilation mode. Only used by clangd intelliSense.

#pragma once

// These are normally declared in amd_hip_runtime.h inside
// #if __HIP_CLANG_ONLY__ (requires __HIP__), so clangd doesn't see them.
// Provide extern declarations so clangd can resolve references.
struct __hip_builtin_threadIdx_t {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
struct __hip_builtin_blockIdx_t {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
struct __hip_builtin_blockDim_t {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

extern const __attribute__((weak)) __hip_builtin_threadIdx_t threadIdx;
extern const __attribute__((weak)) __hip_builtin_blockIdx_t blockIdx;
extern const __attribute__((weak)) __hip_builtin_blockDim_t blockDim;

// atomicAdd and __syncthreads are declared inside __HIP_CLANG_ONLY__ guards
// in amd_device_functions.h / amd_hip_atomic.h. Stub them for clangd.
__attribute__((always_inline)) inline int atomicAdd(int* address, int val) { return *address += val; }
__attribute__((always_inline)) inline unsigned int atomicAdd(unsigned int* address, unsigned int val) { return *address += val; }
__attribute__((always_inline)) inline unsigned long long atomicAdd(unsigned long long* address, unsigned long long val) { return *address += val; }
__attribute__((always_inline)) inline float atomicAdd(float* address, float val) { float r = *address; *address += val; return r; }

__attribute__((always_inline)) inline void __syncthreads() {}

// hipLaunchKernelGGL is in amd_hip_runtime.h but guarded. Provide the macro.
#ifndef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernelName, ...)
#endif
