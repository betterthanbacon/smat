#ifndef __SM_CUDA_CONTEXT_H__
#define __SM_CUDA_CONTEXT_H__

#include <smat/vm/context.h>
#include <smat_cuda/config.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

SM_NAMESPACE_BEGIN

class SM_CUDA_EXPORT cuda_context: public context {
public:
	cuda_context();
	virtual ~cuda_context();

	virtual void set_device(int device);
	virtual void set_randseed(size_t seed);         // override
	virtual void set_options(const optionset& opt); // override
	virtual bool is_supported(dtype_t dt) const;    // override
	virtual void sync();      // override
	virtual void autotune();  // override

	int                   device() const;
	cudaStream_t          stream() const;
	cublasHandle_t        cublas() const;
	curandGenerator_t     curand() const;
	curandState*          curand_state() const;
	const cudaDeviceProp& deviceprop() const;

private:
	virtual void ensure_initialized() const; // override
	void set_curand_seed() const;

	int                       _device;
	bool                      _want_stream;
	mutable int               _curr_device;
	mutable cudaDeviceProp    _deviceprop;
	mutable cudaStream_t      _stream;
	mutable cublasHandle_t    _cublas;
	mutable curandGenerator_t _curand;
	mutable curandState*      _curand_state;
	friend class cuda_block_allocator;
};

SM_INLINE cuda_context& thread_cudactx() { return (cuda_context&)thread_ctx(); } // casts to cuda_context, for convenience

SM_NAMESPACE_END

#endif // __SM_CUDA_CONTEXT_H__
