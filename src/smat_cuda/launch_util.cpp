#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

launchcfg make_elemwise_launchcfg(usize_t size)
{
	auto& dprop = thread_cudactx().deviceprop();
	unsigned  min_threads = dprop.warpSize;
	unsigned  max_threads = 256;
	unsigned  resident_blocks_per_multiprocessor = 8;
	unsigned  max_blocks = 4*resident_blocks_per_multiprocessor*dprop.multiProcessorCount;
	unsigned  block_size = max_threads;
	unsigned  grid_size  = max_blocks;

	if (size < min_threads) {
		// Array can be handled by the smallest block size.
		block_size = min_threads;
		grid_size = 1;
	} else if (size < max_blocks*min_threads) {
		// Array can be handled by several blocks of the smallest size.
		block_size = min_threads;
		grid_size = divup(size,block_size);
	} else if (size < max_blocks*max_threads) {
		// Array must be handled by max number of blocks, each of 
		// larger-than-minimal size, but still a multiple of warp size.
		// In this case, each thread within a block should handle 
		// multiple elements, looping until the entire grid has
		// processed 'size' unique elements.
		block_size = divup(divup(size,min_threads),max_blocks)*min_threads;
		grid_size = max_blocks;
	} else {
		// do nothing
	}

	return launchcfg(grid_size,block_size,0,thread_cudactx().stream());
}

SM_NAMESPACE_END
