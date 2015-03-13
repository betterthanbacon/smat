#ifndef __SM_CUDA_ERRORS_H__
#define __SM_CUDA_ERRORS_H__

#include <smat_cuda/config.h>
#include <base/assert.h>
#include <base/util.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

// ccu(XXX,...) = checked call to cudaXXX(...)
// ccb(XXX,...) = checked call to cublasXXX(...)
// ccr(XXX,...) = checked call to curandXXX(...)
// cce() = check cudaGetLastError and, if there is an error, report it
#define ccu(func,...) { cudaError_t    error  =   cuda##func(__VA_ARGS__); if (error != cudaSuccess)            SM_ERROR(format("AssertionError: CUDA error in %s: %s.",#func,cudaGetErrorString(error)).c_str()); }
#define ccb(func,...) { cublasStatus_t status = cublas##func(__VA_ARGS__); if (status != CUBLAS_STATUS_SUCCESS) SM_ERROR(format("AssertionError: CUBLAS error in %s: %s.",#func,get_cublas_err_str(status)).c_str()); }
#define ccr(func,...) { curandStatus_t status = curand##func(__VA_ARGS__); if (status != CURAND_STATUS_SUCCESS) SM_ERROR(format("AssertionError: CURAND failed in %s: %s.",#func,get_curand_err_str(status)).c_str()); }
#define cce()         { cudaError_t    error  = cudaGetLastError();        if (error != cudaSuccess)            SM_ERROR(format("AssertionError: CUDA error: %s.",cudaGetErrorString(error)).c_str()); }
#ifdef _DEBUG
#define cce_dbsync()  { cudaDeviceSynchronize(); cce(); }
#else
#define cce_dbsync()  { }
#endif

SM_NAMESPACE_BEGIN

SM_CUDA_EXPORT const char* get_cublas_err_str(cublasStatus_t s);
SM_CUDA_EXPORT const char* get_curand_err_str(curandStatus_t s);

SM_NAMESPACE_END

#endif // __SM_CUDA_ERRORS_H__
