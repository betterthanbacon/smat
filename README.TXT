smat -- GPU-accelerated library with C++ and Python interfaces
==============================================================

SYNOPSIS
--------
A numpy-like matrix library with CUDA and cuBLAS acceleration.


REQUIREMENTS
------------
CUDA 6.5+ SDK
   https://developer.nvidia.com/cuda-downloads


REQUIREMENTS FOR PYTHON INTERFACE
---------------------------------
Python 2.7+ (tested on 64-bit only)
   http://www.python.org/getit/

Numpy 1.7+ (preferably linked with MKL)
   http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy  (Win64)


BUILDING (WINDOWS)
------------------
You must have the CUDA 6.5+ runtime DLLs 

   cudart64_65.dll
   cublas64_65.dll
   ...etc...

somewhere in your PATH. On Windows you should furthermore add the
smat/build/release/bin to your PATH, so that smat_cuda.dll 
can be found and dynamically loaded.

Currently you need Visual C++ 2012 to compile smat from source by 
opening smat/vcproj/smat.sln and building. It should create

   smat/build/release/bin/base.dll
                          smat.dll
                          smat_cuda.dll
                          smat_ext_demo.dll


BUILDING (LINUX)
----------------
Once you've unzipped the source, run the makefile below.
By default it assumes cuda is installed in /usr/local/cuda
but this can be overridden with environment variable CUDADIR.
   
   cd smat/src
   make             ("make -j N" to compile with N CPUs)

This should generate four binaries:

   smat/build/release/bin/libbase.so  
                          libsmat.so
                          libsmat_cuda.so
                          libsmat_ext_demo.so


GETTING STARTED
---------------
First, run the unit tests

   cd smat/py
   python run_tests.py

If all the tests pass, you can see how fast you can train
a 784-600-400-10 RELU neural net on MNIST.

   python demo_mnist.py

Make sure you can import smat yourself:

   python
   >>> import smat
   >>> X = smat.eye(5)
   >>> X
   sarray([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)


SMAT's python interface is designed to resemble Numpy so,
to a certain extent, the Numpy documentation is helpful. 
For example:

    >>> import smat as sm
    >>> X = sm.arange(10,20).reshape((5,2))
    >>> X
    sarray([[ 10.,  11.],
            [ 12.,  13.],
            [ 14.,  15.],
            [ 16.,  17.],
            [ 18.,  19.]], dtype=float32)

    >>> sm.sum(X,axis=0)  # sum rows
    sarray([[ 70.,  75.]], dtype=float32)

    >>> sm.sum(X,axis=1)  # sum columns
    sarray([[ 21.],
            [ 25.],
            [ 29.],
            [ 33.],
            [ 37.]], dtype=float32)

    >>> Y = X[1:-1]   # Y is view to middle of X
    >>> Y[:] = 0.     # Set all elements of Y to zero
    >>> X             # Now X is modified.
    sarray([[ 10.,  11.],
            [  0.,   0.],
            [  0.,   0.],
            [  0.,   0.],
            [ 18.,  19.]], dtype=float32)

Note that sarrays may be stored in device memory, 
so to copy back to host one can do

    >>> A = X.asnumpy()   # ... or A = sm.as_numpy(X)
    >>> A
    array([[ 10.,  11.],  # Now this array is a regular
           [ 12.,  13.],  # Numpy array, in host memory.
           [ 14.,  15.],
           [ 16.,  17.],
           [ 18.,  19.]], dtype=float32)

All array-creation functions take an optional "dtype"
parameter:

    >>> I = sm.eye(3,dtype=sm.bool)
    >>> I
    sarray([[ True, False, False],
            [False,  True, False],
            [False, False,  True]], dtype=bool)

    >>> A = I.astype(sm.float32)     # Convert bool to float
    >>> A
    sarray([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]], dtype=float32)


Supported dtypes are: 
     bool,
     int8,  int16,  int32,  int64,
     uint8, uint16, uint32, uint64,
     float32, float64

There are some differences from Numpy, however:

  1. All sarrays have ndim=2, even vectors.
     Variable-dimension arrays will be supported later.

  2. Slicing is only supported along contiguous rows
     or contiguous columns. However, column-sliced
     arrays can only be copied -- they cannot be used
     as part of an arithmetic expression.

    >>> X = sm.eye(5)
    >>> X[2]            # OK: returns view of row  2
    >>> X[:2]           # OK: returns view of rows 0,1
    >>> X[:2,:]         # OK: returns view of rows 0,1
    >>> X[1:4,2:5]      # OK: returns 
    >>> X[1:4,2:5] = 1  # FAIL: column slicing not yet supported
    >>> X[2,-1]         # OK: returns last column value of second row
    >>> X[-2:]          # OK: returns view of rows 3,4
    >>> X[2,2]          # OK: returns 1x1 sarray, still on GPU
    sarray([[ 1.]])
    >>> float(X[2,2])   # OK: returns python float, copied to CPU memory
    1.0

  3. Global functions do not support "out" arguments:

    >>> add(X,Y,out=X)  # FAIL: invalid keyword 'out'
    >>> X[:] = X+Y      # OK (optimized to inplace, since + is element-wise)
    >>> X += Y          # OK (explicitly inplace)

     Why not? Because smat doesn't need it to be efficient.
     The reason is because the second line will be 
     transformed into the first *automatically* by
     the smat "virtual machine."
     It can do this because it executes asynchronously,
     and knows the temporary generated by X+Y 
     was not used anywhere else, so the temporary
     is elided, giving the equivalent of X += Y anyway.

  4. There is no smat.random submodule. Instead there are
     smat.rand(n,m) and smat.randn(n,m) functions. Note that,
     like Numpy, they are called as rand(n,m) and not rand((n,m)).



PERFORMANCE TIPS
----------------
2. NEVER EVER EVER perform operations that mix
   numpy ndarrays and smat sarrays. The numpy array
   will try to be 'smart' and iterate over each element
   of the smat array, resulting in INCORRECT and 
   INCREDIBLY SLOW code execution.

   For example:

    >>> X = np.eye(50)
    >>> Y = sm.eye(50)
    >>> X + Y                   # MEGA SUPER SLOW: X does horrifying/incorrect things to Y
    >>> Y + X                   # FAIL: Y reports a TypeError
    >>> X + Y.asnumpy()         # OK: compute sum on host
    >>> sm.asarray(X) + Y       # OK: compute sum on device

1. Do not try to be overly aggressive about eliminating
   temporaries from your code. Once SMAT's JIT optimizer is
   feature complete, it will do that for you to a large extent.
   For example, you can write:

    >>> A = dot(X,W) + b

   and it will by just as efficient as if it were written
   (using Numpy's "out=" notation)

    >>> A = empty((X.shape[0], W.shape[1]), dtype=X.dtype)
    >>> dot(X, W, out=A)
    >>> A += b

   See the explanation for lack of "out" keyword arguments.







