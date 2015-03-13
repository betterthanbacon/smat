#include <smat/vm/argument.h>

SM_NAMESPACE_BEGIN

static const char* g_vtype_str[] = {
	"none",
	"harray",
	"darray",
	"carray",
	"iarray",
	"user",
};
const char* vtype2str(vtype_t vt) { return g_vtype_str[vt]; }

///////////////////////////////////////////////////////////////

typedef void (*user_deleter_t)(void*);

user_deleter_t& get_user_deleter(argument& op) { return *reinterpret_cast<user_deleter_t*>(&op.shape); }

argument::argument()
: vtype(vt_none)
, dtype(default_dtype)
, shape(0,0,0)
, strides(0,0,0)
{
	*(void**)&value[0] = 0;
}

argument::~argument()
{
	if (vtype == vt_user) {
		// If the argument holds a reference to a user type (i.e. not an array),
		// then check if the user also provided a deleter
		void* user_ptr = get<void*>();
		user_deleter_t user_deleter = get_user_deleter(*this);
		if (user_ptr && user_deleter)
			user_deleter(user_ptr);
	}
}

argument::argument(const argument& src)
: vtype(src.vtype)
, dtype(src.dtype)
, shape(src.shape)
, strides(src.strides)
{
	SM_ASSERT(src.vtype != vt_user)
	*(void**)&value[0] = *(void**)&src.value[0];
}

argument& argument::operator=(const argument& src)
{
	SM_ASSERT(src.vtype != vt_user)
	vtype = src.vtype;
	dtype = src.dtype;
	shape = src.shape;
	strides = src.strides;
	*(void**)&value[0] = *(void**)&src.value[0];
	return *this;
}

argument::argument(argument&& src)
: vtype(src.vtype)
, dtype(src.dtype)
, shape(src.shape)
, strides(src.strides)
{
	*(void**)&value[0] = *(void**)&src.value[0];
	*(void**)&src.value[0] = 0;
}

argument& argument::operator=(argument&& src)
{
	vtype = src.vtype;
	dtype = src.dtype;
	shape = src.shape;
	strides = src.strides;
	*(void**)&value[0] = *(void**)&src.value[0];
	*(void**)&src.value[0] = 0;
	return *this;
}

argument user_arg(void* user, void (*deleter)(void*))
{
	argument arg;
	arg.vtype = vt_user;
	arg.set(user);
	get_user_deleter(arg) = deleter;
	return std::move(arg);
}

SM_NAMESPACE_END
