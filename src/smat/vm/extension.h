#ifndef __SM_DLLS_H__
#define __SM_DLLS_H__

#include <smat/config.h>
#include <base/os.h>

SM_NAMESPACE_BEGIN

// load_extension,unload_extension:
//    Much like using load_dll and unload_dll, with two differences:
//
//       1) The search path automatically includes the smat "bin"
//          developer and install directories.
//
//       2) If the dll exports a "register_ext" symbol, it is 
//          called automatically.
//
SM_EXPORT dllhandle_t load_extension(const char* dllname);
SM_EXPORT void        unload_extension(dllhandle_t handle);

SM_NAMESPACE_END

#endif // __SM_DLLS_H__
