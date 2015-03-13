#include <smat/vm/instruction.h>

SM_NAMESPACE_BEGIN

using namespace std;

///////////////////////////////////////////////////////////////

instruction::instruction(opcode_t opcode): opcode(opcode) { }
instruction::instruction(opcode_t opcode, argument&& arg0): opcode(opcode) { arg[0] = move(arg0); }
instruction::instruction(opcode_t opcode, argument&& arg0, argument&& arg1): opcode(opcode) { arg[0] = move(arg0); arg[1] = move(arg1); }
instruction::instruction(opcode_t opcode, argument&& arg0, argument&& arg1, argument&& arg2): opcode(opcode) { arg[0] = move(arg0); arg[1] = move(arg1); arg[2] = move(arg2); }
instruction::instruction(opcode_t opcode, argument&& arg0, argument&& arg1, argument&& arg2, argument&& arg3): opcode(opcode) { arg[0] = move(arg0); arg[1] = move(arg1); arg[2] = move(arg2); arg[3] = move(arg3); }
instruction::instruction(opcode_t opcode, argument&& arg0, argument&& arg1, argument&& arg2, argument&& arg3, argument&& arg4): opcode(opcode) { arg[0] = move(arg0); arg[1] = move(arg1); arg[2] = move(arg2); arg[3] = move(arg3); arg[4] = move(arg4); }
instruction::~instruction() { }

instruction::instruction(instruction&& src)
: opcode(src.opcode)
{
	for (int i = 0; i < max_arg; ++i)
		arg[i] = move(src.arg[i]);
}
///////////////////////////////////////////////////////////////

SM_NAMESPACE_END
