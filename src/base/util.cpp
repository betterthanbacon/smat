#include <base/util.h>
#include <base/assert.h>
#include <string>
#include <cstdarg>
#include <cstdio>
#include <cstring>

SM_NAMESPACE_BEGIN

using namespace std;

BASE_EXPORT string format(const char* fmt, ...)
{
	va_list va;
	va_start(va,fmt);
	char buffer[2048];
	vsnprintf(buffer,2048,fmt,va);
	return string(buffer);
}

void (*g_vprintf_fn)(const char* fmt, va_list va) = 0;

BASE_EXPORT void print(const char* fmt, ...)
{
	va_list va;
	va_start(va,fmt);
	if (g_vprintf_fn)
		g_vprintf_fn(fmt,va);  // print to user-provided output function (e.g. to python console)
	else
		vprintf(fmt,va);       // print to stdout
}

void (*g_check_interrupt_fn)() = 0;

BASE_EXPORT void check_interrupt()
{
	if (g_check_interrupt_fn)
		g_check_interrupt_fn();
}

BASE_EXPORT void set_print_fn(void (*fn)(const char* fmt, va_list va)) { g_vprintf_fn = fn; }
BASE_EXPORT void set_check_interrupt_fn(void (*fn)())  { g_check_interrupt_fn = fn; }

BASE_EXPORT vector<string> split(const string& s, const char* delims)
{
	if (!delims)
		delims = ", ";
	vector<string> out;
	size_t i = 0;
	for (size_t j = 0; j < s.size(); ++j) {
		if (strchr(delims,s[j]) != 0) {
			if (i < j)
				out.push_back(s.substr(i,j-i));
			i = j+1;
		}
	}
	if (i < s.size())
		out.push_back(s.substr(i));
	return out;
}

BASE_EXPORT unsigned gcd(unsigned a, unsigned b)
{
	if (a == b) return a;
	if (a > b)  return gcd(a-b,b);
	else        return gcd(a,b-a);
}

BASE_EXPORT unsigned lcm(unsigned a, unsigned b)
{
	return a*b / gcd(a,b);
}


SM_NAMESPACE_END
