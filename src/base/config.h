#ifndef __BASE_CONFIG_H__
#define __BASE_CONFIG_H__

#if defined _WIN32 || defined __CYGWIN__
#define SM_INLINE    __forceinline
#define SM_NOINLINE  __declspec(noinline)
#define SM_INTERFACE __declspec(novtable)
#define SM_NORETURN  __declspec(noreturn)
#define SM_DLLEXPORT __declspec(dllexport)
#define SM_DLLIMPORT __declspec(dllimport)
#define SM_THREADLOCAL __declspec(thread)
#elif defined __GNUC__
#define SM_INLINE    inline  //__attribute__ ((always_inline))
#define SM_NOINLINE  __attribute__ ((noinline))
#define SM_INTERFACE
#define SM_NORETURN  __attribute__ ((noreturn))
#define SM_DLLEXPORT __attribute__ ((visibility ("default")))
#define SM_DLLIMPORT 
#define SM_THREADLOCAL __thread
#else
#define SM_INLINE   inline
#define SM_NOINLINE
#define SM_INTERFACE
#define SM_NORETURN
#define SM_DLLEXPORT
#define SM_DLLIMPORT
#endif

#if defined (_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1700) || defined(__GXX_EXPERIMENTAL_CXX0X__)
#define SM_CPP11
#endif


#ifdef BASE_EXPORTS
#define BASE_EXPORT SM_DLLEXPORT
#define BASE_EXTERN_TEMPLATE
#else
#define BASE_EXPORT SM_DLLIMPORT
#define BASE_EXTERN_TEMPLATE extern
#endif

#define SM_NOCOPY(C) protected: C(const C&); C& operator=(const C&);
#define SM_COPYABLE(C) public: C(const C&); C& operator=(const C&);
#define SM_COPY_CTOR(C) C::C(const C& src)
#define SM_COPY_OPER(C) C& C::operator=(const C& src)
#ifdef SM_CPP11
#define SM_MOVEABLE(C) public: C(C&&); C& operator=(C&&);
#define SM_MOVE_CTOR(C) C::C(C&& src)
#define SM_MOVE_OPER(C) C& C::operator=(C&& src)
#else
#define SM_MOVEABLE(C)
#define SM_MOVE_CTOR(C)
#define SM_MOVE_OPER(C)
#endif

#ifndef SM_USE_NAMESPACE
#define SM_USE_NAMESPACE 1
#endif

#if SM_USE_NAMESPACE
#define _SM sm
#define SM_NAMESPACE_BEGIN namespace _SM {
#define SM_NAMESPACE_END   } // namespace _SM
#define USING_NAMESPACE_SM using namespace _SM
#else
#define _SM
#define SM_NAMESPACE_BEGIN
#define SM_NAMESPACE_END
#define USING_NAMESPACE_SM
#endif

#endif // __BASE_CONFIG_H__
