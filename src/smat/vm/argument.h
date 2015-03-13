#ifndef __SM_ARGUMENT_H__
#define __SM_ARGUMENT_H__

#include <smat/dtypes.h>
#include <smat/shape.h>
#include <base/assert.h>

SM_NAMESPACE_BEGIN

// vtype_t
//   Specifies the value-type of an argument, i.e. the type of 
//   the information stored in the argument's "value" member.
//
enum vtype_t {
	vt_none,      // no information in this argument
	vt_harray,    // virtual  address of host-accessible array
	vt_darray,    // physical address of device-accessible array (e.g. pointer that can be passed to CUDA)
	vt_carray,    // constant scalar of some specific value
	vt_iarray,    // constant identity matrix
	vt_user       // an unknown user-extension type, not an array
};
SM_EXPORT const char* vtype2str(vtype_t vt);

// argument
//    An instruction argument.
//    (what smat was designed for)
//
struct SM_EXPORT argument {
	vtype_t  vtype;         // type of the argument's value (constant value? host address? device address?)
	dtype_t  dtype;         // if array vtype: the dtype of array elements (b8? i32? f32?)
	shape_t  shape;         // if array vtype: the shape of the array
	coord_t  strides;       // if array vtype: the strides to access the array (in number of elements)
	char     value[8];      // actual value stored in this argument (value of constant, or address of some memory)

	argument();
	
	template <typename T>
	argument(vtype_t vt, dtype_t dt, shape_t shape, coord_t strides, T value)
	: vtype(vt)
	, dtype(dt)
	, shape(shape)
	, strides(strides == coord_t() ? fullstride(shape) : strides)
	{
		set(value);
	}
	~argument();

	argument(const argument& src);
	argument& operator=(const argument& src);
#ifdef SM_CPP11
	argument(argument&& src);
	argument& operator=(argument&& src);
#endif

	SM_INLINE bool    empty() const { return shape.empty(); }
	SM_INLINE usize_t size()  const { return shape.size(); }
	template <typename T> SM_INLINE void set(T value) { *reinterpret_cast<T*>(this->value) = value; }
	template <typename T> SM_INLINE T    get() const  { return *reinterpret_cast<const T*>(&value[0]); }
};

SM_INLINE argument harray(void*  haddr, dtype_t dt, shape_t shape, coord_t strides = coord_t()) { return argument(vt_harray,dt,shape,strides,haddr); }
SM_INLINE argument darray(size_t daddr, dtype_t dt, shape_t shape, coord_t strides = coord_t()) { return argument(vt_darray,dt,shape,strides,daddr); }
template <typename T>
SM_INLINE argument carray(T value, shape_t shape = shape_t()) { return argument(vt_carray,ctype2dtype(T),shape,coord_t(),value); }
SM_INLINE argument iarray(dtype_t dt, shape_t shape)          { return argument(vt_iarray,dt,shape,coord_t(),0); }
SM_INLINE argument unused_arg() { return argument(); }
SM_EXPORT argument user_arg(void* user, void (*deleter)(void*));

////////////////////////////////////////////////////////////////////////

SM_NAMESPACE_END

#endif // __SM_ARGUMENT_H__
