#include <smat_cuda/reduce_x.cuh>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

void execute_reduce(opcode_t opcode, const argument& src, const argument& dst);

void execute_reduce_x(opcode_t opcode, const argument& src, const argument& dst)
{
	if (src.shape.y == 1) {
		execute_reduce(opcode + (oc_max-oc_max_x),src,dst);
		return;
	}

	#define LAUNCH_CASE(typesets,f,matched) \
		if (opcode == oc_##f##_x) { \
			DECL_SPECIALIZATION_TABLE(typesets,execute_fn2,execute_reduce_x_typed<reducer_##f>::matched); \
			specialization_table(src.dtype)(opcode,src,dst);  \
			return; \
		}
	if (opcode == oc_max)
		opcode = opcode;
	LAUNCH_CASE(T_G,max,matched)
	LAUNCH_CASE(T_G,min,matched)
	LAUNCH_CASE(T_G,sum,promoted)
	LAUNCH_CASE(T_G,mean,asfloat)
	LAUNCH_CASE(T_G,nnz,asuindex)
	SM_UNIMPLEMENTED()
}

SM_NAMESPACE_END
