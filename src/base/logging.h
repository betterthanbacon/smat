#ifndef __SM_LOGGING_H__
#define __SM_LOGGING_H__

#include <base/config.h>
#include <cstddef>

#ifndef SM_ENABLE_LOGGING
#define SM_ENABLE_LOGGING
#endif

#ifdef SM_ENABLE_LOGGING
#define SM_LOG(id,...) {if (_SM::get_log_policy(id) != lp_ignore) _SM::log_entry(id,__VA_ARGS__); }
#else
#define SM_LOG(id,...) 
#endif

SM_NAMESPACE_BEGIN

enum logging_policy_t {
	lp_ignore = 0,
	lp_record = 1 << 0,
	lp_write  = 1 << 1,
	lp_print  = 1 << 2
};

SM_INLINE logging_policy_t operator|(logging_policy_t a, logging_policy_t b) { return (logging_policy_t)((unsigned)a | (unsigned)(b)); }
SM_INLINE logging_policy_t operator&(logging_policy_t a, logging_policy_t b) { return (logging_policy_t)((unsigned)a & (unsigned)(b)); }

BASE_EXPORT void log_entry(const char* id, const char* fmt, ...);
BASE_EXPORT void set_log_policy(const char* id, logging_policy_t p);
BASE_EXPORT void set_log_capacity(size_t capacity);
BASE_EXPORT logging_policy_t get_log_policy(const char* id);

SM_NAMESPACE_END

#endif // __SM_LOGGING_H__
