#include <base/assert.h>
#include <base/util.h>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <stdexcept>

SM_NAMESPACE_BEGIN

using namespace std;

BASE_EXPORT bool g_want_debug_break = true;

BASE_EXPORT void assert_failed_print(const char* fmt, ...)
{
	printf("\n");
	va_list va;
	va_start(va,fmt);
	vprintf(fmt,va);
}

BASE_EXPORT SM_NORETURN void assert_failed(const char* fmt, ...)
{
	va_list va;
	va_start(va,fmt);
	char buffer[2048];
	vsnprintf(buffer,2048,fmt,va);
	throw runtime_error(buffer);
}

SM_NAMESPACE_END
