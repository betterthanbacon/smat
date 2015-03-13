#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

const unsigned c_trans_tile_size = 16;

template <typename T> 
__global__ void kernel_trans(const T* src, T* dst, isize_t n, isize_t m) {
	DECL_KERNEL_VARS;
	unsigned i,j;
	__shared__ T tile[c_trans_tile_size][c_trans_tile_size+1];

	// Read the tile into shared memory.
	i = c_trans_tile_size*by + ty;
	j = c_trans_tile_size*bx + tx;
	if(i < n && j < m)
		tile[ty][tx] = src[m*i+j];

	__syncthreads();

	// Write the tile to global memory in transposed order
	i = c_trans_tile_size*bx + ty;
	j = c_trans_tile_size*by + tx;
	if(i < m && j < n)
		dst[n*i+j] = tile[tx][ty];
}

template <typename T>
struct execute_transpose_typed { // TODO: autotune this
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		if (src.size() > 0) {
			dim3 bdim(c_trans_tile_size,c_trans_tile_size);
			dim3 gdim(divup((unsigned)src.shape.x,c_trans_tile_size),
					  divup((unsigned)src.shape.y,c_trans_tile_size));
			kernel_trans<<<gdim,bdim,0,thread_cudactx().stream()>>>(src.get<const T*>(),dst.get<T*>(),src.shape.y,src.shape.x);
		}
	}
};

// Use NVIDIA BLAS extensions to do more highly-tuned transpose for float and double types.
// Use CUBLAS for float or double type.
template <>
struct execute_transpose_typed<float> {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		float alpha = 1, beta = 0;
		ccb(Sgeam,thread_cudactx().cublas(),CUBLAS_OP_T,CUBLAS_OP_T,(int)src.shape.y,(int)src.shape.x,
			&alpha,src.get<const float*>(),(int)src.shape.x,
			&beta ,src.get<const float*>(),(int)src.shape.x,
			dst.get<float*>(),(int)dst.shape.x)
	}
};

template <>
struct execute_transpose_typed<double> {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		double alpha = 1, beta = 0;
		ccb(Dgeam,thread_cudactx().cublas(),CUBLAS_OP_T,CUBLAS_OP_T,(int)src.shape.y,(int)src.shape.x,
			&alpha,src.get<const double*>(),(int)src.shape.x,
			&beta ,src.get<const double*>(),(int)src.shape.x,
			dst.get<double*>(),(int)dst.shape.x)
	}
};

void execute_transpose(opcode_t opcode, const argument& src, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_transpose_typed);
	specialization_table(src.dtype)(opcode,src,dst);
}

SM_NAMESPACE_END
