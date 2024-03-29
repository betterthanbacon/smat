#include <smat_cuda/elemwise3.cuh>
#include <smat/vm/instruction_db.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

// Define some macros to make, for example, "x+y" be shorthand for "out(i) = arg0(i)+arg1(i))" 
#define EVAL_AS_XY(f) A x = a[i]; x = x; B y = b[j]; y = y; c[k] = f;
#define EVAL2(types,name,f1,f2)           types(name, EVAL_AS_XY(f1), EVAL_AS_XY(f2))
#define EVAL3(types,name,f1,f2,f3)        types(name, EVAL_AS_XY(f1), EVAL_AS_XY(f2), EVAL_AS_XY(f3))
#define EVAL4(types,name,f1,f2,f3,f4)     types(name, EVAL_AS_XY(f1), EVAL_AS_XY(f2), EVAL_AS_XY(f3), EVAL_AS_XY(f4))
#define EVAL5(types,name,f1,f2,f3,f4,f5)  types(name, EVAL_AS_XY(f1), EVAL_AS_XY(f2), EVAL_AS_XY(f3), EVAL_AS_XY(f4), EVAL_AS_XY(f5))

// TODO: faster ways to do integer POW on GPU
template <typename T> SM_DEVICE_INLINE  T powi(T x, T y) { return (T)::round(::pow((typename ctype2ftype(T))x,(typename ctype2ftype(T))y)); }

EVAL3(DEF_ZZZ, or,     x|y,   x|y,   x|y)
EVAL3(DEF_ZZZ, xor,    x^y,   x^y,   x^y)
EVAL3(DEF_ZZZ, and,    x&y,   x&y,   x&y)
EVAL5(DEF_GGG, mod,    false, x%y,   x%y, ::fmodf(x,y), ::fmod(x,y))

EVAL5(DEF_GGG, add,    x|y, x+y, x+y, x+y, x+y)
EVAL5(DEF_GGG, sub,    x^y, x-y, x-y, x-y, x-y)
EVAL5(DEF_GGG, mul,    x&y, x*y, x*y, x*y, x*y)
EVAL5(DEF_GGG, div,    x&y, x/y, x/y, x/y, x/y)
EVAL5(DEF_GGG, pow, x|(!y), _SM::powi(x,y),  _SM::powi(x,y),  ::__powf(x,y), ::pow(x,y))
EVAL5(DEF_GGG, maxe,   x|y, ::max(x,y), ::max(x,y), ::fmaxf(x,y), ::fmax(x,y))
EVAL5(DEF_GGG, mine,   x&y, ::min(x,y), ::min(x,y), ::fminf(x,y), ::fmin(x,y))

EVAL5(DEF_GGL, lor,    x||y,  (x!=0||y!=0)?1:0,  (x!=0||y!=0)?1:0,  (x!=0.0f||y!=0.0f)?1.0f:0.0f,  (x!=0.0||y!=0.0)?1.0:0.0)
EVAL5(DEF_GGL, land,   x&&y,  (x!=0&&y!=0)?1:0,  (x!=0&&y!=0)?1:0,  (x!=0.0f&&y!=0.0f)?1.0f:0.0f,  (x!=0.0&&y!=0.0)?1.0:0.0)
EVAL5(DEF_GGL, eq,     x==y,  x==y,  x==y, x==y, x==y)
EVAL5(DEF_GGL, ne,     x!=y,  x!=y,  x!=y, x!=y, x!=y)
EVAL5(DEF_GGL, lt,    !x&&y,  x< y,  x< y, x< y, x< y)
EVAL5(DEF_GGL, le, !(x&&!y),  x<=y,  x<=y, x<=y, x<=y)

void execute_elemwise3(opcode_t opcode, const argument& a, const argument& b, const argument& c)
{
	#define LAUNCH_CASE(typesets,matched,f,try_inplace)  \
		if (opcode == oc_##f) { \
			DECL_SPECIALIZATION_TABLE(typesets,execute_fn3,execute_elemwise3_typed<k_##f,try_inplace>::matched); \
			specialization_table(a.dtype)(opcode,a,b,c);  \
			return; \
		}
#if SM_WANT_INT || SM_WANT_UINT
	LAUNCH_CASE(T_Z,match,or,true)
	LAUNCH_CASE(T_Z,match,and,true)
	LAUNCH_CASE(T_Z,match,xor,true)
#endif
	LAUNCH_CASE(T_G,match,mod,true)
	LAUNCH_CASE(T_G,match,add,true)
	LAUNCH_CASE(T_G,match,sub,true)
	LAUNCH_CASE(T_G,match,mul,true)
	LAUNCH_CASE(T_G,match,div,true)
	LAUNCH_CASE(T_G,match,pow,true)
	LAUNCH_CASE(T_G,match,maxe,true)
	LAUNCH_CASE(T_G,match,mine,true)
	LAUNCH_CASE(T_G,logical,lor,false)
	LAUNCH_CASE(T_G,logical,land,false)
	LAUNCH_CASE(T_G,logical,eq,false)
	LAUNCH_CASE(T_G,logical,ne,false)
	LAUNCH_CASE(T_G,logical,lt,false)
	LAUNCH_CASE(T_G,logical,le,false)
	SM_UNIMPLEMENTED()
}

SM_NAMESPACE_END
