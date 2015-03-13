#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

template <typename T>
__global__ void kernel_diff_y(const T* src, T* dst, usize_t m, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		dst[i] = src[i+m]-src[i];  // could be implemented by oc_sub operation on two views of arg, but this should be marginally faster.
}


template <typename T>
struct execute_diff_typed {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size == 0)
			return;
		if (opcode == oc_diff_x) {
			SM_UNIMPLEMENTED();
		} else if (opcode == oc_diff_y) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			kernel_diff_y<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(src.get<const T*>(),dst.get<T*>(),dst.shape.x,size);
		} else {
			SM_UNREACHABLE();
		}
	}
};

void execute_diff(opcode_t opcode, const argument& src, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_diff_typed);
	specialization_table(src.dtype)(opcode,src,dst);
}

SM_NAMESPACE_END
