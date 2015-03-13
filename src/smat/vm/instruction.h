#ifndef __SM_INSTRUCTION_H__
#define __SM_INSTRUCTION_H__

#include <smat/vm/argument.h>
#include <string>
#include <list>

SM_NAMESPACE_BEGIN

typedef int opcode_t;

// instruction
//   Instruction specification.
//
struct SM_EXPORT instruction { SM_MOVEABLE(instruction) SM_NOCOPY(instruction)
public:
#ifdef SM_CPP11
	typedef argument&& operand_ref;
#else
	typedef argument& operand_ref;
#endif
	enum { max_arg = 5 };
	instruction(opcode_t opcode);
	instruction(opcode_t opcode, operand_ref arg0);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1, operand_ref arg2);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1, operand_ref arg2, operand_ref arg3);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1, operand_ref arg2, operand_ref arg3, operand_ref arg4);
	~instruction();

	opcode_t opcode;       // opcode for instruction
	argument arg[max_arg]; // operands for instruction
};

SM_NAMESPACE_END

#endif // __SM_INSTRUCTION_H__
