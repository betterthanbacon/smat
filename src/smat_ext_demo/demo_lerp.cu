#include <smat/dtypes.h>
using namespace sm;

template <typename T>
__global__ void lerp_kernel(const T* a, const T* b, T* c, T alpha, usize_t size)
{
	usize_t tx  = threadIdx.x, bx  = blockIdx.x;
	usize_t bdx = blockDim.x,  gdx = gridDim.x;

	#pragma unroll
	for (uindex_t i = bdx*bx+tx; i < size; i += bdx*gdx)
		c[i] = (1-alpha)*a[i] + alpha*b[i];
}

void launch_lerp(dim3 gdim, dim3 bdim, unsigned smem, cudaStream_t stream,
                 usize_t size, dtype_t dtype,
                 const void* a, 
                 const void* b,
                       void* c,
                 double alpha)
{
	switch (dtype) {
	case f32: lerp_kernel<<<gdim,bdim,smem,stream>>>((const float* )a,(const float* )b,(float* )c,(float )alpha,size); break;
	case f64: lerp_kernel<<<gdim,bdim,smem,stream>>>((const double*)a,(const double*)b,(double*)c,(double)alpha,size); break;
	}
}
