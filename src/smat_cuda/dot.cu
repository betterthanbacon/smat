#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

SM_INLINE cublasOperation_t cublas_opA(opcode_t opcode) { return (opcode == oc_dottn || opcode == oc_dottt) ? CUBLAS_OP_T : CUBLAS_OP_N; }
SM_INLINE cublasOperation_t cublas_opB(opcode_t opcode) { return (opcode == oc_dotnt || opcode == oc_dottt) ? CUBLAS_OP_T : CUBLAS_OP_N; }

template <typename T>
struct execute_dot_typed { };  // Only implemented for float types supported by CUBLAS.

// Use CUBLAS for float or double type.
template <>
struct execute_dot_typed<float> {
	static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c)
	{
		float alpha = 1, beta = 0;
		int n = (int)c.shape.y;
		int m = (int)c.shape.x;
		int k = (int)(cublas_opA(opcode) == CUBLAS_OP_T ? a.shape.y : a.shape.x);
		if (n > 0 && m > 0 && k > 0) {
			ccb(Sgemm,thread_cudactx().cublas(),
				cublas_opB(opcode),cublas_opA(opcode),
				m,n,k,&alpha,
				b.get<const float*>(),(int)b.shape.x,
				a.get<const float*>(),(int)a.shape.x,&beta,
				c.get<      float*>(),(int)c.shape.x);
		}
	}
};

template <>
struct execute_dot_typed<double> {
	static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c)
	{
		double alpha = 1, beta = 0;
		int n = (int)c.shape.y;
		int m = (int)c.shape.x;
		int k = (int)(cublas_opA(opcode) == CUBLAS_OP_T ? a.shape.y : a.shape.x);
		if (n > 0 && m > 0 && k > 0) {
			ccb(Dgemm,thread_cudactx().cublas(),
				cublas_opB(opcode),cublas_opA(opcode),
				m,n,k,&alpha,
				b.get<const double*>(),(int)b.shape.x,
				a.get<const double*>(),(int)a.shape.x,&beta,
				c.get<      double*>(),(int)c.shape.x);
		}
	}
};

void execute_dot(opcode_t opcode, const argument& a, const argument& b, const argument& c)
{
	DECL_SPECIALIZATION_TABLE(T_F,execute_fn3,execute_dot_typed);
	specialization_table(a.dtype)(opcode,a,b,c);
}

SM_NAMESPACE_END
