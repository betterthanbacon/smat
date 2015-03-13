#ifndef __SM_SPECIALIZATION_TYPELISTS_H__
#define __SM_SPECIALIZATION_TYPELISTS_H__

#include <smat/config.h>
#include <base/typelist.h>

SM_NAMESPACE_BEGIN

// Legend for the macros defined below
// L = logical (boolean)
// I = signed integral only
// U = unsigned integral only
// F = float only
// Z = integral
// S = signed numeric
// N = numeric
// G = generic (logical + numeric)
// P = generic types promoted to at least 32 bits (e.g. when doing summation over columns of int8, output is int32, like in numpy)
//
// So ZZL means "integral arg0, integral arg1, logical arg2", such as operator A >= B where A and B have integral types

/////////////////////////////////////////////////////////////////////////////////////////////

#define TYPES(...)   make_typelist<__VA_ARGS__>::type
#define CAT(...)     typelist_cat<__VA_ARGS__>::type
#define REP(T,n)     typelist_rep<T,n>::type
#define ZIP(...)     typelist_zip<__VA_ARGS__>::type
#define PROD(...)    typelist_prod<__VA_ARGS__>::type
#define WRAP(...)    typelist_wrap<__VA_ARGS__>::type
typedef TYPES(SM_BOOL_TYPES)                      _T_L;
typedef TYPES(SM_INT_TYPES)                       _T_I;
typedef TYPES(SM_UINT_TYPES)                      _T_U;
typedef TYPES(SM_FLOAT_TYPES)                     _T_F;
typedef CAT(_T_L,CAT(_T_I,_T_U))                  _T_Z;
typedef CAT(_T_I,_T_F)                            _T_S;
typedef CAT(_T_I,CAT(_T_U,_T_F))                  _T_N;
typedef CAT(_T_L,_T_N)                            _T_G;

////////////////////// DTYPE LISTS FOR UNARY OPERATIONS () //////////////////////////

typedef WRAP(_T_L) T_L;
typedef WRAP(_T_I) T_I;
typedef WRAP(_T_U) T_U;
typedef WRAP(_T_F) T_F;
typedef WRAP(_T_Z) T_Z;
typedef WRAP(_T_S) T_S;
typedef WRAP(_T_N) T_N;
typedef WRAP(_T_G) T_G;


////////////////////// DTYPE LISTS FOR UNARY OPERATIONS (ARG,OUT) //////////////////////////

typedef ZIP(_T_L,_T_L)  T_LL;
typedef ZIP(_T_I,_T_I)  T_II;
typedef ZIP(_T_U,_T_U)  T_UU;
typedef ZIP(_T_F,_T_F)  T_FF;
typedef ZIP(_T_Z,_T_Z)  T_ZZ;
typedef ZIP(_T_S,_T_S)  T_SS;
typedef ZIP(_T_N,_T_N)  T_NN;
typedef ZIP(_T_G,_T_G)  T_GG;
typedef PROD(_T_G,_T_G) T_GxG;

typedef PROD(_T_I,_T_L)  T_IL;
typedef PROD(_T_U,_T_L)  T_UL;
typedef PROD(_T_F,_T_L)  T_FL;
typedef PROD(_T_Z,_T_L)  T_ZL;
typedef PROD(_T_S,_T_L)  T_SL;
typedef PROD(_T_N,_T_L)  T_NL;
typedef PROD(_T_G,_T_L)  T_GL;

typedef ZIP(_T_L,TYPES(uint32_t))  T_LP;
typedef ZIP(_T_I,TYPES(int32_t,int32_t,int32_t,int64_t))  T_IP;
typedef ZIP(_T_U,TYPES(uint32_t,uint32_t,uint32_t,uint64_t))  T_UP;
typedef ZIP(_T_F,_T_F)              T_FP;
typedef CAT(T_LP,CAT(T_IP,T_UP))    T_ZP;  // promote integral to at least int32
typedef CAT(T_IP,T_FP)              T_SP;  // promote signed   to at least int32
typedef CAT(T_IP,CAT(T_UP,T_FP))    T_NP;  // promote numeric  to at least int32
typedef CAT(T_LP,T_NP)              T_GP;  // promote generic  to at least int32

typedef ZIP(_T_G,TYPES(SM_REPEAT(float,11)))  T_GPs; // promote generic to at least float32
typedef ZIP(_T_G,TYPES(SM_REPEAT(double,11))) T_GPd; // promote generic to at least float64
typedef CAT(T_GPs,T_GPd)                      T_GPF; // promote generic to at least default

SM_NAMESPACE_END

#endif // __SM_SPECIALIZATION_TYPELISTS_H__
