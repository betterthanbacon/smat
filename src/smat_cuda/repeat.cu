#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

template <unsigned bdx, unsigned bdy, int wdy, bool check_y, typename T>
__global__ void kernel_repeat_x(const T* __restrict__ src, T* __restrict__ dst, unsigned nx, unsigned ny, unsigned repx)
{
	unsigned bx = blockIdx.x;
	unsigned by = blockIdx.y;
	unsigned tx = threadIdx.x;
	unsigned ty = threadIdx.y;
	unsigned dst_x = bdx*bx+tx;
	unsigned src_x = dst_x/repx;
	if (dst_x >= nx*repx)
		return;
	#pragma unroll
	for (unsigned y = ty; y < wdy*bdy && (!check_y || wdy*bdy*by+y < ny); y += bdy)
		dst[(wdy*bdy*by+y)*nx*repx+dst_x] = src[(wdy*bdy*by+y)*nx+src_x];
}


template <typename T>
struct execute_repeat_typed {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		if (dst.size() == 0 || src.size() == 0)
			return;
		shape_t rep = dst.shape/src.shape;
		const unsigned bdx = 64; // tuned for relatively large matrices, on titan
		const unsigned bdy = 4;
		const unsigned wdy = 2;
		dim3 bdim(bdx,bdy);
		dim3 gdim(divup(dst.shape.x,bdx),divup(dst.shape.y,bdy*wdy));
		if (rep.y == 1 && rep.z == 1) {
			static bool s_was_cache_setup = false;
			if (!s_was_cache_setup) {
				s_was_cache_setup = true;
				cudaFuncSetCacheConfig(&kernel_repeat_x<bdx,bdy,wdy,true,T>,cudaFuncCachePreferL1);
			}
			kernel_repeat_x<bdx,bdy,wdy,true><<<gdim,bdim,0,thread_cudactx().stream()>>>(src.get<const T*>(),dst.get<T*>(),src.shape.x,src.shape.y,rep.x);
		} else {
			SM_UNIMPLEMENTED();
		}
	}
};

void execute_repeat(opcode_t opcode, const argument& src, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_repeat_typed);
	specialization_table(src.dtype)(opcode,src,dst);
}

SM_NAMESPACE_END
