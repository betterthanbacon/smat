#include <base/config.h>

SM_NAMESPACE_BEGIN

void autotune_reduce_y();

extern "C" SM_DLLEXPORT void register_ext()
{
	autotune_reduce_y();
}

SM_NAMESPACE_END
