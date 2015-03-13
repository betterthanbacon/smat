#include <smat_cuda/reduce_y.cuh>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

void reduce_y_launch(opcode_t opcode, const argument& src, const argument& dst, 
					 unsigned bdx, unsigned wdx, unsigned wdy,
					 void (*reduce )(unsigned gdx, unsigned gdy, const void*, void*, unsigned nx, unsigned ny, unsigned n),
					 void (*partial)(unsigned gdx, unsigned gdy, const void*, void*, unsigned nx, unsigned ny, unsigned n))
{
	unsigned gdx = divup((unsigned)src.shape.x,bdx*wdx);
	unsigned gdy = divup((unsigned)src.shape.y,bdx*wdy);
	heap_alloc buffer;
	void* dst_ptr = dst.get<void*>();
	if (gdy > 1) {
		buffer = thread_cudactx().heap().alloc(src.shape.x*gdy*dtype_size(dst.dtype));
		dst_ptr = buffer.addr;
	}
	reduce(gdx,gdy,src.get<const void*>(),dst_ptr,src.shape.x,src.shape.y,gdy == 1 ? src.shape.y : 0);

	// There is more than one output block per column, so we cannot directly write to "dst".
	// Instead, allocate temporary memory to store the partial reductions.
	gdx = divup((unsigned)src.shape.x,reduce_y_partial_bdx);
	unsigned ny = gdy;
	while (ny > 1) {
		const void* src_ptr = dst_ptr; // read from the temporary buffer, possibly also writing to it if this is not the last iteration.
		gdy = divup(ny,reduce_y_partial_bdx);
		if (gdy == 1)
			dst_ptr = dst.get<void*>();  // we've come down to the last iteration, so pass it along to the final output destination
		partial(gdx,gdy,src_ptr,dst_ptr,src.shape.x,ny,gdy == 1 ? src.shape.y : 0);
		ny = gdy;
	}

	// If we had to allocate temporary memory for storing intermediate results, free it now
	if (buffer.addr)
		thread_cudactx().heap().free(buffer);
}

void execute_reduce(opcode_t opcode, const argument& src, const argument& dst);

void execute_reduce_y(opcode_t opcode, const argument& src, const argument& dst)
{
	if (src.shape.x == 1) {
		execute_reduce(opcode + (oc_max-oc_max_y),src,dst);
		return;
	}

	#define LAUNCH_CASE(typesets,f,matched) \
		if (opcode == oc_##f##_y) { \
			DECL_SPECIALIZATION_TABLE(typesets,execute_fn2,execute_reduce_y_typed<reducer_##f>::matched); \
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
