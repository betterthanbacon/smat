#ifndef __SM_CUDA_REDUCE_Y_AUTOTUNE_H__
#define __SM_CUDA_REDUCE_Y_AUTOTUNE_H__

#include <smat_cuda/config.h>
#include <smat/vm/util/autotune.h>

SM_NAMESPACE_BEGIN

template <template <unsigned kernel, unsigned bdx, unsigned wdx, unsigned wdy> class launch>
class reduce_y_autotune_table: public autotune_table4<execute_fn2> {
public:
	reduce_y_autotune_table()
	{
		add_fn(&launch<0,128,1,1>::execute);
		add_fn(&launch<1,128,1,1>::execute);
		//add_fn(&launch<0,32,1,1>::execute);
		//insert(10,10,0);
		//insert(10,1000,0);
		//insert(1000,10,1);
		//insert(1000,1000,1);
	}
};

SM_NAMESPACE_END

#endif // __SM_CUDA_REDUCE_Y_AUTOTUNE_H__
