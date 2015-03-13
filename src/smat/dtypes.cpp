#include <smat/dtypes.h>
#include <base/assert.h>
#include <type_traits>
#include <stdexcept>

SM_NAMESPACE_BEGIN

using namespace std;

static dtype_t  g_default_dtypef= dtype_t::f32;
static dtype_t  g_default_dtype = dtype_t::f32; // if you change this, be sure to change initialization of g_dtype_sizes[default_dtype] accordingly
static size_t   g_dtype_sizes[] = {
	sizeof(bool),
	sizeof(int8_t),
	sizeof(uint8_t),
	sizeof(int16_t),
	sizeof(uint16_t),
	sizeof(int32_t),
	sizeof(uint32_t),
	sizeof(int64_t),
	sizeof(uint64_t),
	sizeof(float),
	sizeof(double)
};

static const char* g_dtype_strings[] = {
	"b8",
	"i8",
	"u8",
	"i16",
	"u16",
	"i32",
	"u32",
	"i64",
	"u64",
	"f32",
	"f64"
};

SM_EXPORT const char* dtype2str(dtype_t dt)
{
	SM_ASSERT(dt != default_dtype);
	return g_dtype_strings[dt];
}

SM_EXPORT void set_default_dtype(dtype_t dt)
{
	SM_ASSERT(dt != default_dtype);
	g_default_dtype = dt;
	if (dt == f32 || dt == f64)
		g_default_dtypef = dt;
}

SM_EXPORT void set_default_dtypef(dtype_t dt)
{
	SM_ASSERT(dt != default_dtype);
	SM_ASSERT(dt == f32 || dt == f64);
	g_default_dtypef = dt;
}

SM_EXPORT dtype_t get_default_dtype()           { return g_default_dtype; }
SM_EXPORT dtype_t get_default_dtypef()          { return g_default_dtypef; }
SM_EXPORT dtype_t actual_dtype(dtype_t dt)      { return (dt == default_dtype) ? g_default_dtype : dt; }
SM_EXPORT int     dtype_size(dtype_t dt)        { return (int)g_dtype_sizes[actual_dtype(dt)]; }

// This conversion table is meant to roughly mimic numpy, except that
// here the conversion rules do not depend on the actual numerical
// values of the arguments.
// It implies, for example, that a binary operation on types (i64,f32)
// would result in a value of type f64, like numpy.
// However, in numpy if one evaluates Z = int8(X)*int16(y) for
// matrix X and scalar y, the dtype of Z is int8 if y <= 255
// and int16 otherwise.
const size_t c_num_dtype = sizeof(g_dtype_strings)/sizeof(g_dtype_strings[0]);
static const dtype_t g_common_types[c_num_dtype][c_num_dtype] = {
	{b8 ,i8 ,u8 ,i16,u16,i32,u32,i64,u64,f32,f64},  // (b8 ,j) -> ?
	{i8 ,i8 ,i8 ,i16,i16,i32,i32,i64,i64,f32,f64},  // (i8 ,j) -> ?
	{u8 ,i8 ,u8 ,i16,u16,i32,u32,i64,u64,f32,f64},  // (u8 ,j) -> ?
	{i16,i16,i16,i16,i16,i32,i32,i64,i64,f32,f64},  // (i16,j) -> ?
	{u16,i16,u16,i16,u16,i32,u32,i64,u64,f32,f64},  // (u16,j) -> ?
	{i32,i32,i32,i32,i32,i32,i32,i64,i64,f32,f64},  // (i32,j) -> ?
	{u32,i32,u32,i32,u32,i32,u32,i64,u64,f32,f64},  // (u32,j) -> ?
	{i64,i64,i64,i64,i64,i64,i64,i64,i64,f64,f64},  // (i64,j) -> ?
	{u64,i64,u64,i64,u64,i64,u64,i64,u64,f64,f64},  // (u64,j) -> ?
	{f32,f32,f32,f32,f32,f32,f32,f64,f64,f32,f64},  // (f32,j) -> ?
	{f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64}   // (f64,j) -> ?
};

SM_EXPORT dtype_t arithmetic_result_dtype(dtype_t dta, dtype_t dtb)
{
	return g_common_types[actual_dtype(dta)][actual_dtype(dtb)];
}

extern "C" {

SM_EXPORT void        api_set_default_dtype(dtype_t dt)   { SM_API_TRY; set_default_dtype(dt); SM_API_CATCH }
SM_EXPORT void        api_set_default_dtypef(dtype_t dt)  { SM_API_TRY; set_default_dtype(dt); SM_API_CATCH }
SM_EXPORT dtype_t     api_get_default_dtype()             { SM_API_TRY; return get_default_dtype(); SM_API_CATCH_AND_RETURN(f32) }
SM_EXPORT dtype_t     api_get_default_dtypef()            { SM_API_TRY; return get_default_dtypef(); SM_API_CATCH_AND_RETURN(f32) }
SM_EXPORT int         api_dtype_size(dtype_t dt)          { SM_API_TRY; return dtype_size(dt); SM_API_CATCH_AND_RETURN(0) }

}

SM_NAMESPACE_END
