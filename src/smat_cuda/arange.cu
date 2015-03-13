#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

template <typename T>
__global__ void kernel_arange(T start, T* dst, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		dst[i] = start + (T)i;
}

template <typename T>
struct execute_arange_typed { // TODO: autotune this
	static void execute(opcode_t opcode, const argument& start, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size > 0) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			kernel_arange<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(start.get<T>(),dst.get<T*>(),size);
		}
	}
};

void execute_arange(opcode_t opcode, const argument& start, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_N,execute_fn2,execute_arange_typed);
	specialization_table(dst.dtype)(opcode,start,dst);
}

SM_NAMESPACE_END
