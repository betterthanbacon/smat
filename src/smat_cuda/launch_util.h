#ifndef __SM_CUDA_LAUNCH_UTIL_H__
#define __SM_CUDA_LAUNCH_UTIL_H__

#include <smat_cuda/config.h>
#include <smat/dtypes.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
// Declare shorthand versions of the standard cuda thread-local constants
#define DECL_KERNEL_VARS          \
	const unsigned& tx  = threadIdx.x; ((void)tx); \
	const unsigned& ty  = threadIdx.y; ((void)ty); \
    const unsigned& bx  = blockIdx.x;  ((void)bx); \
    const unsigned& by  = blockIdx.y;  ((void)by); \
    const unsigned& bdx  = blockDim.x;  ((void)bdx); \
    const unsigned& bdy  = blockDim.y;  ((void)bdy); \
    const unsigned& gdx  = gridDim.x;  ((void)gdx); \
    const unsigned& gdy  = gridDim.y;  ((void)gdy);
#else
// These variable declarations are provided so that Visual Studio source editor doesn't try to highlight intellisense errors
#define DECL_KERNEL_VARS    \
	const unsigned tx  = 1; ((void)tx); \
	const unsigned ty  = 1; ((void)ty); \
    const unsigned bx  = 1; ((void)bx); \
    const unsigned by  = 1; ((void)by); \
    const unsigned bdx = 1; ((void)bdx); \
    const unsigned bdy = 1; ((void)bdy); \
    const unsigned gdx = 1; ((void)gdx); \
    const unsigned gdy = 1; ((void)gdy);
#endif

#ifdef __CUDACC__
SM_DEVICE_INLINE double atomicAdd(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		__double_as_longlong(val +__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

SM_NAMESPACE_BEGIN

// KERNEL CONFIGURATION MACROS
struct launchcfg {
	launchcfg(const dim3& gdim, const dim3& bdim, unsigned smem, cudaStream_t stream): gdim(gdim), bdim(bdim), smem(smem), stream(stream) { }
	dim3 gdim;
	dim3 bdim;
	unsigned smem;
	cudaStream_t stream;
};

#define SM_CUDA_LAUNCH(kernel,cfg) kernel<<<(cfg).gdim,(cfg).bdim,(cfg).smem,(cfg).stream>>>

SM_CUDA_EXPORT launchcfg make_elemwise_launchcfg(usize_t size); // Determines good launch config for basic linear kernels.

SM_NAMESPACE_END

#endif // __SM_CUDA_LAUNCH_UTIL_H__
