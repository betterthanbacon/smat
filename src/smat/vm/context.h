#ifndef __SM_CONTEXT_H__
#define __SM_CONTEXT_H__

#include <smat/config.h>
#include <smat/vm/instruction.h>
#include <smat/vm/heap.h>

SM_NAMESPACE_BEGIN

class context;
class optionset;
class block_allocator;
class machine;
class instruction_list;
struct instruction;
struct instruction_info;

struct backend_info {
	int  uuid;           // unique identifier, can be hash of classname string for example.
	char name[32];       // "cuda", "mkl", etc.
	char version[32];
	char device[128];
};

SM_EXPORT void     set_backend(const char* backend_name);
SM_EXPORT void     set_backend(const char* backend_name, const optionset& opt);
SM_EXPORT void     reset_backend();
SM_EXPORT void     reset_backend(const optionset& opt);
SM_EXPORT void     destroy_backend(bool force=false);
SM_EXPORT context& thread_ctx(); // if thread_ctx is called before any context is created, a the default backend will be used (e.g. CUDA).

class SM_EXPORT context { SM_NOCOPY(context)
public:
	virtual ~context();

	const _SM::backend_info& backend_info() const;
	      _SM::machine& machine();
	const _SM::machine& machine() const;
	      _SM::heap&    heap();
	const _SM::heap&    heap() const;
	
	virtual void set_verbose(int verbose);
	virtual void set_randseed(size_t seed);
	virtual void set_sanitycheck(int level);
	virtual void set_max_queuesize(size_t size);
	virtual bool is_supported(dtype_t dt) const;
	virtual void set_options(const optionset& opt);
	virtual void sync();

	virtual heap_alloc alloc(shape_t s, dtype_t dtype);
	virtual heap_alloc alloc(size_t size);
	virtual void       free(heap_alloc& addr);
	virtual void       autotune();

	void emit(opcode_t opcode, argument arg0);
	void emit(opcode_t opcode, argument arg0, argument arg1);
	void emit(opcode_t opcode, argument arg0, argument arg1, argument arg2);
	void emit(opcode_t opcode, argument arg0, argument arg1, argument arg2, argument arg3);
	void emit(opcode_t opcode, argument arg0, argument arg1, argument arg2, argument arg3, argument arg4);

protected:
	context(const _SM::backend_info& info, _SM::machine* machine, _SM::heap* heap); // these instances all created by the backend's context sublass
	virtual void ensure_initialized() const;
#ifdef SM_CPP11
	void emit(instruction&& instr);
#endif
	void flush();
	void destroy_heap();

	_SM::backend_info mutable _backend_info;
	_SM::machine*     _machine;
	_SM::heap*        _heap;
	instruction_list* _queue;
	size_t  _queuesize;
	size_t  _max_queuesize;
	int     _verbose;
	int     _sanitycheck;
	int     _randseed;

	friend class machine;
};

SM_NAMESPACE_END

#endif // __SM_CONTEXT_H__
