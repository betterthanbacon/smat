#ifndef __SM_ASSERT_H__
#define __SM_ASSERT_H__

#include <base/config.h>

#if defined(_WIN32)
#ifdef _WIN64
#define DL64
#include <intrin.h>
#ifndef SM_DEBUGBREAK
#define SM_DEBUGBREAK if (_SM::g_want_debug_break) __debugbreak()
#endif
#else
#define DL32
#ifndef SM_DEBUGBREAK
#define SM_DEBUGBREAK if (_SM::g_want_debug_break) __asm { int 3 }
#endif
#endif
#endif

#if defined(__GNUC__)
#if defined(__x86_64__) || defined(__ppc64__)
#define DL64
#else
#define DL32
#endif
#ifndef SM_DEBUGBREAK
#define SM_DEBUGBREAK  // not supported
#endif
#endif

SM_NAMESPACE_BEGIN
#if defined(_WIN32)
BASE_EXPORT extern      bool g_want_debug_break;
#endif
BASE_EXPORT             void assert_failed_print(const char* fmt, ...);
BASE_EXPORT SM_NORETURN void assert_failed(const char* fmt, ...);
SM_NAMESPACE_END

#define SM_ASSERT_FAILED(fmt,...) _SM::assert_failed_print(fmt,__VA_ARGS__); SM_DEBUGBREAK; _SM::assert_failed(fmt,__VA_ARGS__);

#define SM_ERROR(msg)          { SM_ASSERT_FAILED("%s\n\tin %s:%d",(const char*)msg,__FILE__,__LINE__); }
#define SM_ASSERT(expr)        { if (expr) { } else { SM_ASSERT_FAILED("AssertionError: ASSERT(%s) failed in %s:%d\n",#expr,__FILE__,__LINE__); } }
#define SM_ASSERTMSG(expr,msg) { if (expr) { } else { SM_ASSERT_FAILED("%s\n\nASSERT(%s) failed in %s:%d\n",(const char*)(msg),#expr,__FILE__,__LINE__); } }
#define SM_UNREACHABLE()       { SM_DEBUGBREAK; SM_ASSERT_FAILED("AssertionError: unreachable code in %s:%d\n",__FILE__,__LINE__); }
#define SM_UNIMPLEMENTED()     { SM_DEBUGBREAK; SM_ASSERT_FAILED("NotImplementedError: raised in %s:%d\n",__FILE__,__LINE__); }

#if defined(_DEBUG)
#ifndef SM_ENABLE_DBASSERT
#define SM_ENABLE_DBASSERT
#endif
#endif

#ifdef SM_ENABLE_DBASSERT
#define SM_DBSTATEMENT(expr) expr
#else
#define SM_DBSTATEMENT(expr)
#endif

#ifdef SM_ENABLE_DBASSERT
#define SM_DBASSERT(expr)      SM_ASSERT(expr)
#else
#define SM_DBASSERT(expr)      { }
#endif



#endif // __SM_ASSERT_H__
