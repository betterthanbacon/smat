#ifndef __SM_OS_H__
#define __SM_OS_H__

#include <base/config.h>
#include <cstddef>

SM_NAMESPACE_BEGIN

BASE_EXPORT size_t get_system_memory_avail();
BASE_EXPORT size_t get_system_memory_total();
BASE_EXPORT size_t get_process_memory_used();
BASE_EXPORT const char* get_last_os_error();

typedef size_t dllhandle_t;
BASE_EXPORT dllhandle_t load_dll(const char* name);
BASE_EXPORT void        unload_dll(dllhandle_t handle);
BASE_EXPORT void*       get_dll_proc(dllhandle_t dll, const char* procname);

BASE_EXPORT const char* user_home_dir();
BASE_EXPORT void mkdir(const char* dir);
BASE_EXPORT bool isdir(const char* dir);

SM_NAMESPACE_END

#endif // __SM_OS_H__
