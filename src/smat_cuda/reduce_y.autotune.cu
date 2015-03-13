#include <smat_cuda/reduce_y.cuh>
#include <base/typelist.h>

SM_NAMESPACE_BEGIN

using namespace std;

// Just "sum of floats" to tune kernel launch parameters for all reducers and all dtypes.
typedef execute_reduce_y_typed<reducer_sum>::general<float,float> reduce_y_functor;

size_t reduce_y_setup_args(const autotune_query& query, argument* arg)
{
	isize_t nx = query.q0;
	isize_t ny = query.q1;
	arg[0].shape.x = nx;
	arg[0].shape.y = ny;
	arg[1].shape.x = nx;
	arg[1].shape.y = 1;
	return (ny-1)*nx; // flop count
}

void autotune_reduce_y()
{
	typedef make_typelist<     // <kernel,bdx,wdx,wdy>
		//make_intlist4<0, 32,1,1>::type,
		//make_intlist4<0, 64,1,1>::type,
		make_intlist4<0,128,1,1>::type,
		//make_intlist4<0, 32,2,1>::type,
		//make_intlist4<0, 64,2,1>::type,
		make_intlist4<0,128,4,1>::type,
		make_intlist4<0,128,4,4>::type,

		//make_intlist4<1, 32,1,1>::type,
		//make_intlist4<1, 64,1,1>::type,
		make_intlist4<1,128,1,1>::type,
		//make_intlist4<1, 32,1,2>::type,
		//make_intlist4<1, 64,1,2>::type,
		make_intlist4<1,128,1,4>::type
	>::type psets;

	autotune_queries queries;
	const size_t c_max_dim  = 1 << 20;   // nx and ny in range [0,max_dim-1]
	const size_t c_max_size = 1 << 28;   // largest nx*ny matrix to be considered
	for (size_t i = 2; i <= c_max_dim; i *= 2) {
		for (size_t j = 2; j <= c_max_dim; j *= 2) {
			if (i*j <= c_max_size)
				queries.push_back(autotune_query((index_t)i,(index_t)j));
		}
	}
//	queries.clear();
	//queries.push_back(autotune_query(10,1000000));
	//queries.push_back(autotune_query(1000000,10));

	autotuner tuner(oc_sum_y);
	tuner.sample<reduce_y_functor::launch,psets>(queries,reduce_y_setup_args);
	tuner.print_best();
	//tuner.save(__FILE__);
}

SM_NAMESPACE_END