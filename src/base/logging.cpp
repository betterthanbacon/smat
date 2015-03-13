#include <base/logging.h>
#include <base/util.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <list>

SM_NAMESPACE_BEGIN

using namespace std;

struct log_entry_t {
	log_entry_t() { id[0] = '\0'; msg[0] = '\0'; }
	void assign(const char* id, const char* fmt, va_list& va)
	{
		strncpy(this->id,id,16); this->id[15] = '\0';
		vsnprintf(msg,256,fmt,va); this->msg[192] = '\0';
	}
	void print() const
	{
		_SM::print("%s: %s\n",id,msg);
	}
	char id[16];
	char msg[256];
};

static vector<log_entry_t> g_log_entries(4096);
static size_t              g_log_pos = 0;
static unordered_map<string,logging_policy_t> g_policies;

BASE_EXPORT void log_entry(const char* id, const char* fmt, ...)
{
	logging_policy_t policy = get_log_policy(id);
	if (policy == lp_ignore)
		return;  // if not logging this kind of event, return immediately without formatting the message

	va_list va;
	va_start(va,fmt);

	if (policy & lp_record) {
		g_log_entries[g_log_pos].assign(id,fmt,va);
		if (policy & lp_print)
			g_log_entries[g_log_pos].print();
		g_log_pos++;
		if (g_log_pos >= g_log_entries.size())
			g_log_pos = 0;
	} else if (policy & lp_print) {
		log_entry_t entry;
		entry.assign(id,fmt,va);
		entry.print();
	}
}

BASE_EXPORT void set_log_policy(const char* id, logging_policy_t p)
{
	g_policies[id] = p;
}

BASE_EXPORT void set_log_capacity(size_t capacity)
{
	g_log_entries.clear();
	g_log_entries.resize(capacity);
	g_log_pos = 0;
}

BASE_EXPORT logging_policy_t get_log_policy(const char* id)
{
	auto it = g_policies.find(id);
	if (it == g_policies.end())
		return lp_ignore;
	return it->second;
}

SM_NAMESPACE_END
