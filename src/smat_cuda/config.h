#ifndef __SMAT_CUDA_CONFIG_H__
#define __SMAT_CUDA_CONFIG_H__

#include <base/config.h>

#ifdef SMAT_CUDA_EXPORTS
#define SM_CUDA_EXPORT SM_DLLEXPORT
#else
#define SM_CUDA_EXPORT SM_DLLIMPORT
#endif

#define SM_DEVICE_INLINE __device__ __forceinline__
#define SM_DEVICE_HOST_INLINE __device__ __host__ __forceinline__

#endif // __SMAT_CUDA_CONFIG_H__
