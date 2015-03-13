#include <smat_cuda/reduce.cuh>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

// Determines good launch config for reduce operation.
launchcfg make_reduce_launchcfg(dtype_t dt, usize_t size)
{
	// TODO: add autotuning mechanism
	unsigned maxthread = 128;
	unsigned maxblock  = 2*8*thread_cudactx().deviceprop().multiProcessorCount;
	unsigned nthread   = (size >= maxthread*2) ? maxthread : divup_pow2((size+1)/2);
	unsigned nblock    = min(maxblock,(size + (nthread*2-1))/(nthread*2));
    unsigned smem = nthread*dtype_size(dt);

    // When there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds.
	if (nthread <= 32)
		smem *= 2;

	return launchcfg(nblock,nthread,smem,thread_cudactx().stream());
}

void execute_reduce(opcode_t opcode, const argument& src, const argument& dst)
{
	#define LAUNCH_CASE(typesets,f,matched) \
		if (opcode == oc_##f) { \
			DECL_SPECIALIZATION_TABLE(typesets,execute_fn2,execute_reduce_typed<reducer_##f>::matched); \
			specialization_table(src.dtype)(opcode,src,dst);  \
			return; \
		}

	LAUNCH_CASE(T_G,max,matched)
	LAUNCH_CASE(T_G,min,matched)
	LAUNCH_CASE(T_G,sum,promoted)
	LAUNCH_CASE(T_G,mean,asfloat)
	LAUNCH_CASE(T_G,nnz,asuindex)
	SM_UNIMPLEMENTED()
}

SM_NAMESPACE_END
