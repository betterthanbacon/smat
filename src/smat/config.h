#ifndef __SMAT_CONFIG_H__
#define __SMAT_CONFIG_H__

#include <base/config.h>
#include <exception>
#include <stdexcept>
#include <string>

const int cuda_uuid = 0x01;

#ifdef SMAT_EXPORTS
#define SM_EXPORT SM_DLLEXPORT
#define SM_EXTERN_TEMPLATE
#else
#define SM_EXPORT SM_DLLIMPORT
#define SM_EXTERN_TEMPLATE extern
#endif

#define SM_WANT_BOOL 1
#define SM_WANT_INT 1
#define SM_WANT_UINT 1
#define SM_WANT_DOUBLE 1

#ifndef SM_WANT_BOOL
#define SM_WANT_BOOL 1
#endif
#if SM_WANT_BOOL
#define SM_BOOL_TYPES  bool
#else
#define SM_BOOL_TYPES
#endif

#ifndef SM_WANT_INT
#define SM_WANT_INT 1
#endif
#if SM_WANT_INT
#define SM_INT_TYPES   int8_t, int16_t, int32_t, int64_t
#else
#define SM_INT_TYPES
#endif

#ifndef SM_WANT_UINT
#define SM_WANT_UINT 1
#endif
#if SM_WANT_UINT
#define SM_UINT_TYPES  uint8_t,uint16_t,uint32_t,uint64_t
#else
#define SM_UINT_TYPES 
#endif

#ifndef SM_WANT_DOUBLE
#define SM_WANT_DOUBLE 1
#endif
#if SM_WANT_DOUBLE
#define SM_FLOAT_TYPES float,double
#else
#define SM_FLOAT_TYPES float
#endif


#define SM_API_TRY    try {
#define SM_API_CATCH  } catch (const std::exception& e) {\
                         g_smat_last_error = e.what(); \
                         return; \
                      }
#define SM_API_CATCH_AND_RETURN(default_rval)\
                      } catch (const std::exception& e) {\
                         g_smat_last_error = e.what();\
                         return default_rval;\
                      }
extern "C" SM_EXPORT std::string g_smat_last_error;

#endif // __SMAT_CONFIG_H__
