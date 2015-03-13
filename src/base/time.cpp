#include <base/time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define _WIN32_WINNT 0x0403
#include <windows.h>
#endif

SM_NAMESPACE_BEGIN

#ifdef _WIN32 
static ticks_t g_tickrate = 0;
extern "C" ticks_t ticks() // this is not thread safe
{
	ticks_t result = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&result);
	return result;
}
extern "C" ticks_t tickrate()
{
	if (g_tickrate == 0)
		QueryPerformanceFrequency((LARGE_INTEGER*)&g_tickrate);
	return g_tickrate;
}
#else
extern "C" ticks_t ticks()    { return ::clock(); }
extern "C" ticks_t tickrate() { return CLOCKS_PER_SEC; }
#endif

extern "C" double duration(ticks_t ticks)
{
	return (double)ticks / tickrate();
}

SM_NAMESPACE_END