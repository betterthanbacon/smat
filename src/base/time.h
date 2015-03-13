#ifndef __SM_TIME_H__
#define __SM_TIME_H__

#include <base/config.h>
#ifndef _WIN32
#include <ctime>
#endif

SM_NAMESPACE_BEGIN

#ifdef _WIN32
typedef long long ticks_t;
#else
typedef ::clock_t ticks_t;
#endif
extern "C" BASE_EXPORT ticks_t ticks();
extern "C" BASE_EXPORT ticks_t tickrate(); // ticks per second
extern "C" BASE_EXPORT double  duration(ticks_t ticks);

SM_NAMESPACE_END

#endif // __SM_TIME_H__
