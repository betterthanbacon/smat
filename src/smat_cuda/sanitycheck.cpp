#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/instruction_db.h>
#include <smat/vm/heap.h>
#include <base/range.h>
#include <vector>

SM_NAMESPACE_BEGIN

using namespace std;

// This function should only be called if "sanity checks"
// are enabled.
// The starting address of the raw (padded) allocation should
// be stored as the value of 'arg', and the raw (padded) size of
// the allocation should be stored in arg.size.x.
// The first and last "heap.pitch()" bytes of the allocation
// will be filled with magic numbers and checked for integrity
// when opcode == oc_free.
void execute_alloc_free(opcode_t opcode, const argument& arg)
{
	SM_ASSERT(arg.vtype == vt_darray);
	
	auto stream = thread_cudactx().stream();
	const size_t padding_bytes = thread_cudactx().heap().pitch();
	const size_t padding_ints  = padding_bytes/sizeof(int);
	vector<int> magic(padding_ints,0xdeadbeef);

	char* ptr = arg.get<char*>();
	size_t alloc_size = arg.shape.x;
	if (opcode == oc_alloc) {
		// Write a magic number to the memory immediately before and after the allocation.
		// This obviously assumes padding was provided by alloc() ahead of time.
		ccu(StreamSynchronize,stream);
		cce();
		ccu(MemsetAsync,ptr,0xcc,alloc_size,stream);
		ccu(MemcpyAsync,ptr,                         &magic[0],padding_bytes,cudaMemcpyHostToDevice,stream);
		ccu(MemcpyAsync,ptr+alloc_size-padding_bytes,&magic[0],padding_bytes,cudaMemcpyHostToDevice,stream);
		ccu(StreamSynchronize,stream);
		cce();
	} else {
		SM_ASSERT(opcode == oc_free)
		// Read back the memory immediately before and after the allocation, 
		// and verify that it still contains the magic number.
		vector<int> readback(padding_ints,0xdeadb00b);
		auto check_readback = [&](const char* which) {
			for (auto i : range(padding_ints))
				if (readback[i] != magic[i]) {
					string msg = format("MemoryError: Device heap corruption %s allocated range; expected all 0xdeadbeef values but found:\n",which);
					for (auto j : range(padding_ints))
						msg += format("  0x%08x\n",readback[j]);
					SM_ERROR(msg.c_str());
				}
		};

		ccu(StreamSynchronize,stream);
		cce();
		ccu(MemcpyAsync,&readback[0],ptr,padding_bytes,cudaMemcpyDeviceToHost,stream);
		ccu(StreamSynchronize,stream);
		check_readback("before");
		
		ccu(MemcpyAsync,&readback[0],ptr+alloc_size-padding_bytes,padding_bytes,cudaMemcpyDeviceToHost,stream);
		ccu(StreamSynchronize,stream);
		check_readback("after");
		ccu(StreamSynchronize,stream);
		cce();

		// Finally, clobber the memory itself with garbage values, hoping 
		// to reliably trigger a bug if any code uses this memory while 
		// it's free.
		ccu(MemsetAsync,ptr,0xcc,alloc_size,stream);
	}
}

SM_NAMESPACE_END
