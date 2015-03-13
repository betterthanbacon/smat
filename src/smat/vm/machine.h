#ifndef __SM_MACHINE_H__
#define __SM_MACHINE_H__

#include <smat/vm/instruction_db.h>
#include <smat/dtypes.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

class optionset;
class block_allocator;

class SM_EXPORT machine { SM_NOCOPY(machine)
public:
	machine();
	virtual ~machine();
	virtual void set_options(const optionset& opt);
	virtual void validate(instruction& instr, const instruction_info& info);
	virtual void execute(const instruction& instr, const instruction_info& info, const instruction_impl& impl);

protected:
	void validate_dtypes(instruction& instr, const instruction_info& info);
	void validate_strides(instruction& instr, const instruction_info& info);
	friend class context;
};

SM_NAMESPACE_END

#endif // __SM_MACHINE_H__
