#include <smat/dtypes.h>
using namespace sm;

template <typename T>
__global__ void clamp_kernel(T* a, usize_t size, T lo, T hi)
{
	usize_t tx  = threadIdx.x, bx  = blockIdx.x;
	usize_t bdx = blockDim.x,  gdx = gridDim.x;

	#pragma unroll
	for (uindex_t i = bdx*bx+tx; i < size; i += bdx*gdx)
		a[i] = max(lo,min(hi,a[i]));
}

void launch_clamp(dim3 gdim, dim3 bdim, unsigned smem, cudaStream_t stream,
                  usize_t size, dtype_t dtype,
                  void* a, double lo, double hi)
{
	switch (dtype) {
	case f32: clamp_kernel<<<gdim,bdim,smem,stream>>>((float* )a,size,(float )lo,(float )hi); break;
	case f64: clamp_kernel<<<gdim,bdim,smem,stream>>>((double*)a,size,(double)lo,(double)hi); break;
	}
}


