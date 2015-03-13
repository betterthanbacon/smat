#ifndef __SM_SMAT_CUDA_REDUCERS_H__
#define __SM_SMAT_CUDA_REDUCERS_H__

#include <smat_cuda/launch_util.h>
#include <limits>
#include <cfloat>

SM_NAMESPACE_BEGIN

// Define dtype_limits<T>::min() and dtype_limits<T>::max() for 
// each type T that a kernel may be specialzed.
template <typename T> struct dtype_limits {};
#define DEF_DTYPE_LIMIT(dtype,_min,_max) \
	template <> struct dtype_limits<dtype>   { SM_DEVICE_INLINE static dtype min() { return _min; } SM_DEVICE_INLINE static dtype max() { return _max; } };
DEF_DTYPE_LIMIT(bool    ,    false,  true)
DEF_DTYPE_LIMIT(int8_t  , CHAR_MIN,  CHAR_MAX)
DEF_DTYPE_LIMIT(uint8_t ,        0, UCHAR_MAX)
DEF_DTYPE_LIMIT(int16_t , SHRT_MIN,  SHRT_MAX)
DEF_DTYPE_LIMIT(uint16_t,        0, USHRT_MAX)
DEF_DTYPE_LIMIT(int32_t ,  INT_MIN,   INT_MAX)
DEF_DTYPE_LIMIT(uint32_t,        0,  UINT_MAX)
DEF_DTYPE_LIMIT(int64_t ,LLONG_MIN, LLONG_MAX)
DEF_DTYPE_LIMIT(uint64_t,        0,ULLONG_MAX)
DEF_DTYPE_LIMIT(float   , -FLT_MAX,   FLT_MAX)
DEF_DTYPE_LIMIT(double  , -FLT_MAX,   DBL_MAX)

#define DEFINE_REDUCER(name,init,elem,part,fin) \
	template <typename _value_type, typename _result_type> \
	struct name {                              \
		typedef _value_type  value_type;                  \
		typedef _result_type result_type;                 \
		SM_DEVICE_INLINE name()             { result = init; } \
		SM_DEVICE_INLINE void element(value_type x)   { elem; }          \
		SM_DEVICE_INLINE void partial(result_type p)  { part; }          \
		SM_DEVICE_INLINE void finalize(usize_t size)  { fin; }           \
		result_type result;                                              \
	};

template <typename T> SM_DEVICE_INLINE T      reducer_element_max(T x, T y) { return y > x ? y : x; }
template <typename T> SM_DEVICE_INLINE T      reducer_element_min(T x, T y) { return y < x ? y : x; }
template <>           SM_DEVICE_INLINE float  reducer_element_max(float  x, float  y) { return ::fmaxf(x,y); }
template <>           SM_DEVICE_INLINE float  reducer_element_min(float  x, float  y) { return ::fminf(x,y); }
template <>           SM_DEVICE_INLINE double reducer_element_max(double x, double y) { return ::fmax(x,y); }
template <>           SM_DEVICE_INLINE double reducer_element_min(double x, double y) { return ::fmin(x,y); }

DEFINE_REDUCER(reducer_max,
			   dtype_limits<value_type>::min(),
			   result = reducer_element_max(result,x),
			   result = reducer_element_max(result,p),
			   )

DEFINE_REDUCER(reducer_min,
			   dtype_limits<value_type>::max(),
			   result = reducer_element_min(result,x),
			   result = reducer_element_min(result,p),
			   )

DEFINE_REDUCER(reducer_sum,
			   0,
			   result += x,
			   result += p,
			   )

DEFINE_REDUCER(reducer_mean,
			   0,
			   result += x,
			   result += p,
			   if (size) result /= size)

DEFINE_REDUCER(reducer_nnz,
			   0,
			   if (x) ++result,
			   result += p,
			   )


// For kernel_reduce, we want to call it with the normal reduce kernels, but then also for 
// subsequent "collect" passes in that algorithm we need to override the default behaviour 
// of element so that it becomes a partial operation.
template <typename reducer>
struct reducer_partial_results: public reducer { 
	SM_DEVICE_INLINE void element(typename reducer::result_type x) { reducer::partial(x); }
};

#if SM_WANT_UINT
	typedef uindex_t reduce_index_t;
#elif SM_WANT_INT
	typedef index_t reduce_index_t;
#else
	typedef float reduce_index_t;
#endif

SM_NAMESPACE_END

#endif // __SM_SMAT_CUDA_REDUCERS_H__
