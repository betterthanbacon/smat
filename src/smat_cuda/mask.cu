#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

template <typename T>
__global__ void kernel_apply_mask(T* A, const bool* M, unsigned size)
{
	DECL_KERNEL_VARS
	for (unsigned i = bdx*bx + tx; i < size; i += bdx*gdx)
		if (!M[i])
			A[i] = (T)0;
}


void execute_apply_mask(opcode_t opcode, const argument& A, const argument& M)
{
	unsigned size = (unsigned)A.size();
	launchcfg cfg = make_elemwise_launchcfg(size);
	if      (A.dtype == f32) kernel_apply_mask<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(A.get<float* >(),M.get<bool*>(),(unsigned)size);
	else if (A.dtype == f64) kernel_apply_mask<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(A.get<double*>(),M.get<bool*>(),(unsigned)size);
	else { SM_UNIMPLEMENTED(); }
}


SM_NAMESPACE_END
