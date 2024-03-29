from testutil import *
import numpy as np
import smat        # want module name too
from smat import *
import cPickle as pickle
import os,os.path

####################################################
#  GLOBAL VARIABLES USED IN EACH TEST (as read-only)

n,m = 123,21
Z,_Z = None,None     # Z = numpy.zeros(n,m),    _Z = smat.zeros(n,m)
O,_O = None,None     # ones
I,_I = None,None     # identity
A,_A = None,None     # random matrix 1 is whatever
B,_B = None,None     # random matrix 2 is whatever
C,_C = None,None     # random matrix 3 has non-zero values (good as a denominator)
W,_W = None,None     # row vector taken from ravel'd C

def clear_test_matrices():
    global  Z, O, I, A, B, C, W
    global _Z,_O,_I,_A,_B,_C,_W
    Z, O, I, A, B, C, W = None, None, None, None, None, None, None 
    _Z,_O,_I,_A,_B,_C,_W= None, None, None, None, None, None, None

def alloc_test_matrices(dt):
    global n,m
    global  Z, O, I, A, B, C, W
    global _Z,_O,_I,_A,_B,_C,_W
    Z = np.zeros((n,m),dt);  _Z = zeros((n,m),dt);
    O = np.ones((n,m),dt);   _O = ones((n,m),dt);
    I = np.eye(n,dtype=dt);  _I = eye(n,dtype=dt);
    A = make_rand(n,m,dt);                           A = A.astype(dt)
    B = make_rand(n,m,dt); B = abs(B);               B = B.astype(dt)
    C = make_rand(n,m,dt); C = abs(C); C[C==0] += 1; C = C.astype(dt)
    W  =  C.ravel()[10:n+10].reshape((-1,1))
    _A = as_sarray(A)
    _B = as_sarray(B)
    _C = as_sarray(C)
    _W = as_sarray(W)
    assert_eq(_Z,Z)

#####################################################

def test_create(dt):
    # Machine-created matrices should match numpy versions
    assert_eq(_Z,Z)
    assert_eq(_O,O)
    assert_eq(_I,I)
    assert_eq(_A,A)
    assert_eq(_B,B)
    assert_eq(_C,C)
    assert_eq(_W,W)

    # Properties.
    assert _Z.size == Z.size
    assert _Z.ndim == Z.ndim
    assert _Z.shape== Z.shape
    assert _Z.dtype== Z.dtype
    assert _Z.nrow == Z.shape[0]
    assert _Z.ncol == Z.shape[1]

    # Create range-valued array.
    assert_eq(arange(5,67),np.arange(5,67))

    # Use _like creation functions.
    _X = empty_like(_Z)
    assert _X.shape == _Z.shape
    assert _X.dtype == _Z.dtype
    _X = empty_like(_Z,uint8)
    assert _X.shape == _Z.shape
    assert _X.dtype == uint8
    _X = zeros_like(_O)
    assert_eq(_X,_Z)
    _X = ones_like(_Z)
    assert_eq(_X,_O)


#####################################################

def test_copy(dt):
    # Upload and then download from machine
    assert_eq(as_sarray(A), A)
    assert_eq(_A.copy(),A)

    # Type casting.
    assert _A.astype(float32).dtype == float32
    if int32 in get_supported_dtypes():
        assert _A.astype(int32).dtype   == int32

#####################################################

ext_demo_dll = None

class c_clamp_args_t(Structure):  # This same structure is defined in cuda_ext_clamp.cu
    _fields_ = [("lo", c_double),
                ("hi", c_double)]
c_clamp_args_p = POINTER(c_clamp_args_t)

def load_extension_demo():
    global ext_demo_dll
    ext_demo_dll = load_extension("smat_ext_demo")
    ext_demo_dll.api_lerp.declare( c_smat_p, [c_smat_p,c_smat_p,c_double])  # C, [A,B,alpha]
    ext_demo_dll.api_clamp.declare(    None, [c_smat_p,c_clamp_args_p])     #    [A,(lo,hi)]

def unload_extension_demo():
    global ext_demo_dll
    unload_extension(ext_demo_dll)
    ext_demo_dll = None

def lerp(A,B,alpha):
    C_ptr = ext_demo_dll.api_lerp(A._ptr,B._ptr,c_double(alpha))
    return sarray(C_ptr)

def clamp(A,lo,hi):
    args = c_clamp_args_t(lo,hi)
    ext_demo_dll.api_clamp(A._ptr,byref(args))

def test_smat_extension(dt):
    load_extension_demo()
    
    # Function lerp(A,B) computes A*(1-B) and returns the result
    _X = lerp(_A,_B,0.25)
    X  = (1-0.25)*A + 0.25*B
    assert_close(_X,X)

    # Function clamp(A,lo,hi) computes A[:] = maximum(lo,minimum(hi,A)) inplace
    _X = _A.copy(); clamp(_X,-0.5,0.5)
    X  =  A.copy(); X = np.maximum(-0.5,np.minimum(0.5,X))
    assert_eq(_X,X)

    unload_extension_demo()

#####################################################

def test_random(dt):
    # Bernoulli random numbers
    _A1 = bernoulli((n,m),0.5,dt)
    _A2 = bernoulli((n,m),0.5,dt)
    _A3 = bernoulli((n,m),0.2,dt)
    assert_ne(_A1,_A2)  # pretty pathetic test of randomness, but whatever
    assert_ne(_A2,_A3)
    assert_any(_A1 == 0)
    assert_any(_A1 == 1)
    assert_all(logical_or(_A1 == 1,_A1 == 0))
    assert_all(nnz(_A1) > nnz(_A3)*1.1)

#####################################################

def test_random_int(dt):
    # Integral random numbers
    _A1 = rand(n,m,dt)
    _A2 = rand(n,m,dt)
    _A3 = rand(n,m,dt)
    assert_ne(_A1,_A2)  # pretty pathetic test of randomness, but whatever
    assert_ne(_A2,_A3)

    rand_seed(1234)
    _A4 = rand(n,m,dt)
    assert_ne(_A1,_A4)
    rand_seed(1234)
    _A5 = rand(n,m,dt)
    assert_eq(_A4,_A5)  # same random seed should give same random stream


#######################################################################

def test_random_float(dt):
    # Floating point random numbers
    _A1 = randn(n,m,dt)
    _A2 = randn(n,m,dt)
    _A3 = randn(n,m,dt)
    assert_ne(_A1,_A2)  # pretty pathetic test of randomness, but whatever
    assert_ne(_A2,_A3)

    rand_seed(1234)
    _A4 = randn(n,m,dt)
    assert_ne(_A1,_A4)
    rand_seed(1234)
    _A5 = randn(n,m,dt)
    assert_eq(_A4,_A5)  # same random seed should give same random stream

#######################################################################

def test_closeness(dt):
    A1 = np.require(make_rand(n,m,dt)*1e-5,dtype=float32)
    _A1 = asarray(A1)
    assert     allclose(A1,A1*(1+1e-6),rtol=1e-5,atol=0)
    assert not allclose(A1,A1*(1+1e-4),rtol=1e-5,atol=0)
    assert     allclose(A1,A1+1e-6,rtol=0,atol=1e-5)
    assert not allclose(A1,A1+1e-4,rtol=0,atol=1e-5)

#####################################################

def test_attributes():
    """Test setattr and getattr functions."""
    A = empty((5,5))
    A.setattr("foo",1)
    A.setattr("bar",10)
    assert A.foo == 1
    assert A.bar == 10
    del A.foo
    assert A.bar == 10
    del A.bar

#######################################################################

def test_serialize(dt):
    """
    Tests that an smat array of any type can be 
    serialized to disk, including its attributes.
    """
    A1 = rand(30,10,dtype=dt)
    X1 = rand(256,5,dtype=dt)
    X1.setattr("A",A1)
    fname = "smat_unittest_serialize.pkl"
    with open(fname,"wb") as file:
        pickle.dump(X1,file)
    with open(fname,"rb") as file:
        X2 = pickle.load(file)
    os.remove(fname)
    assert isinstance(X2,sarray)
    assert_eq(X1,X2)
    assert(X2.hasattr("A"))   # Make sure that attributes are also serialized
    A2 = X2.getattr("A")
    assert_eq(A1,A2)

#####################################################

def test_slicing(dt):
    # Row slicing.
    assert_eq(_A[0],      A[0])
    assert_eq(_A[0,:],    A[0,:])
    assert_eq(_A[11],     A[11])
    assert_eq(_A[11,:],   A[11,:])
    assert_eq(_A[-1],     A[-1])
    assert_eq(_A[-1,:],   A[-1,:])
    assert_eq(_A[:],      A[:])
    assert_eq(_A[:,:],    A[:,:])
    assert_eq(_A[:21],    A[:21])
    assert_eq(_A[:21,:],  A[:21,:])
    assert_eq(_A[-21:],   A[-21:])
    assert_eq(_A[-21:-16],A[-21:-16:])
    assert_eq(_A[-21:,:], A[-21:,:])
    assert_eq(_A[21:-21], A[21:-21:])
    assert_eq(_A[21:-21,:],A[21:-21,:])

    # Row slicing on a row vector
    _a,a = _A[3,:],A[3:4,:]
    assert_eq(_a, a)
    assert_eq(_a[0], a[0])

    # Column slicing.
    assert_eq(_A[:,0],      A[:,0:1])
    assert_eq(_A[:,1],      A[:,1:2])
    assert_eq(_A[:,:5],     A[:,:5])
    assert_eq(_A[:,-1],     A[:,-1:])
    assert_eq(_A[:,-5],     A[:,-5:-4])
    assert_eq(_A[:,-5:],    A[:,-5:])
    assert_eq(_A[:,-5:-1],    A[:,-5:-1])

    # Column slicing on a column vector
    _a,a = _A[:,3],A[:,3:4]
    assert_eq(_a, a)
    assert_eq(_a[:,0], a[:,0:1])

    # Row + Column slicing.
    assert_eq(_A[5,5],      A[5,5])
    assert_eq(_A[:5,5],     A[:5,5:6])
    assert_eq(_A[2:5,5],    A[2:5,5:6])
    assert_eq(_A[2:5,5:7],  A[2:5,5:7])
    assert_eq(_A[-6:,-10:], A[-6:,-10:])

    # Row-sliced assignments.
    _X,X = _A.copy(),A.copy(); _X[:]   ,X[:]    = 789     ,789;     assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[:]   ,X[:]    = _B[:]   ,B[:];    assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[0]   ,X[0]    = _B[0]   ,B[0];    assert_eq(_X,X) # Broadcast copy.
    _X,X = _A.copy(),A.copy(); _X[:]   ,X[:]    = _B[0]   ,B[0];    assert_eq(_X,X) # Broadcast copy.
    _X,X = _A.copy(),A.copy(); _X[:]   ,X[:]    = _W      ,W;       assert_eq(_X,X) # Broadcast copy.
    _X,X = _A.copy(),A.copy(); _X[-1]  ,X[-1]   = _B[-1]  ,B[-1];   assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[:11] ,X[:11]  = _B[:11] ,B[:11];  assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[-11:],X[-11:] = _B[-11:],B[-11:]; assert_eq(_X,X)

    # Col-sliced assignments.
    # _X,X = _A.copy(),A.copy(); _X[:,0]   ,X[:,0] = 789     ,789;     assert_eq(_X,X) # Assigning const to column strided array not implemented
    _X,X = _A.copy(),A.copy(); _X[:,0]   ,X[:,0]   = _B[:,0]   ,B[:,0];    assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[:,1]   ,X[:,1]   = _B[:,1]   ,B[:,1];    assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[:,:10] ,X[:,:10] = _B[:,:10] ,B[:,:10];  assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[:,:10] ,X[:,:10] = _B[:,:10] ,B[:,:10];  assert_eq(_X,X)
    _X,X = _A.copy(),A.copy(); _X[:,10:-10] ,X[:,10:-10] = _B[:,10:-10] ,B[:,10:-10];  assert_eq(_X,X)

    # Row+Col-sliced assignments.
    _X,X = _A.copy(),A.copy(); _X[5:10,7:13] ,X[5:10,7:13] = _B[5:10,7:13] ,B[5:10,7:13];  assert_eq(_X,X)

#####################################################

def test_reshape(dt):
    _X,X = _A[:45,:].copy(),A[:45,:].copy()
    assert_eq(_X.reshape((7,135)),X.reshape((7,135)))
    assert_eq(_X.reshape((-1,7)),X.reshape((-1,7)))
    _Y,Y = _X[:9],X[:9]; _Y[:,:],Y[:,:] = 1,1
    assert_eq(_Y,Y)
    assert_eq(_X,X)
    assert_eq(_X.reshape((7,135)),X.reshape((7,135)))
    assert_eq(_X.reshape((135,-1)),X.reshape((135,-1)))

#####################################################

def test_transpose(dt):
    assert_eq(transpose(_I),I)
    assert_eq(transpose(_A.reshape((-1,1))),np.transpose(A.reshape((-1,1))))
    assert_eq(transpose(_A.reshape((1,-1))),np.transpose(A.reshape((1,-1))))
    assert_eq(transpose(_A.reshape((3,-1))),np.transpose(A.reshape((3,-1))))
    assert_eq(transpose(_A.reshape((-1,3))),np.transpose(A.reshape((-1,3))))
    assert_eq(transpose(_A),np.transpose(A))
    assert_eq(transpose(_B),np.transpose(B))
    assert_eq(_A.T,         np.transpose(A))
    assert_eq(_B.T,         np.transpose(B))

#######################################################################

def test_dot(dt):
    assert_eq(dot(_I,_I), I)
    assert_close(dot(_A.reshape((1,-1)),_B.reshape((-1,1))), np.dot(A.reshape((1,-1)),B.reshape((-1,1))))
    assert_close(dot(_A,_B.T) ,np.dot(A,B.T))
    assert_close(dot_nt(_A,_B),np.dot(A,B.T))
    assert_close(dot(_A.T,_B) ,np.dot(A.T,B))
    assert_close(dot_tn(_A,_B),np.dot(A.T,B))

#######################################################################

def test_bitwise(dt):
    # Bitwise and logical (NOT,AND,OR,XOR).
    assert_eq(~_A, ~A)
    assert_eq(_A |  0, A | 0)
    assert_eq( 1 | _B, 1 | B)
    assert_eq(_A | _B, A | B)
    assert_eq(_A ^  0, A ^ 0)
    assert_eq( 1 ^ _B, 1 ^ B)
    assert_eq(_A ^ _B, A ^ B)
    assert_eq(_A &  0, A & 0)
    assert_eq( 1 & _B, 1 & B)
    assert_eq(_A & _B, A & B)

#######################################################################

def test_logical(dt):
    # Logical operations (as opposed to bitwise)
    assert_eq(logical_not(_A),  np.logical_not(A))
    assert_eq(logical_or(_A,_B),  np.logical_or(A,B))
    assert_eq(logical_and(_A,_B),  np.logical_and(A,B))
    
#######################################################################

def test_modulo(dt):
    _X,X = _A,A
    _Y,Y = _C,C
    if dt in dtypes_sint:
        _X,X = abs(_X),abs(X)   # cuda modulo for signed types differs from numpy, 
        _Y,Y = abs(_Y),abs(Y)   # so don't compare that case
    assert_eq(_X %  7, (X % np.asarray(7,dtype=dt)).astype(dt))
    assert_eq( 7 % _Y, (np.asarray(7,dtype=dt) % Y).astype(dt))
    assert_eq(_X % _Y, (X % Y).astype(dt))

#######################################################################

def test_naninf(dt):
    _X = _A.copy(); _X[3] = np.nan; _X[5] = np.inf
    X  =  A.copy();  X[3] = np.nan;  X[5] = np.inf
    assert_eq(isnan(_X), np.isnan(X))
    assert_eq(isinf(_X), np.isinf(X))
    assert_eq(isinf(_A/0),np.ones(A.shape,dtype=bool))
    assert_eq(isnan(0*_A/0),np.ones(A.shape,dtype=bool))

#######################################################################

def test_math_float(dt):
    Amin = A.min()
    Amax = A.max()
    A2  = (2*( A-Amin)/(Amax-Amin)-1)*.999
    _A2 = (2*(_A-Amin)/(Amax-Amin)-1)*.999
    assert_eq(clip(_A,0,1),np.clip(A,0,1))
    assert_eq(abs(_O),     np.abs(O))
    assert_eq(abs(_A),     np.abs(A))
    assert_eq(square(_A),  np.square(A))
    assert_eq(round(_A),   np.round(A))
    assert_eq(floor(_A),   np.floor(A))
    assert_eq(ceil(_A),    np.ceil(A))
    assert_close(sin(_A),  np.sin(A))
    assert_close(cos(_A),  np.cos(A))
    assert_close(tan(_A),  np.tan(A))
    assert_close(arcsin(_A2), np.arcsin(A2))
    assert_close(arccos(_A2), np.arccos(A2))
    assert_close(arctan(_A2), np.arctan(A2))
    assert_close(sinh(_A),  np.sinh(A))
    assert_close(cosh(_A),  np.cosh(A))
    assert_close(tanh(_A),  np.tanh(A))
    assert_close(arcsinh(_A2), np.arcsinh(A2))
    assert_close(arccosh(1+abs(_A2)), np.arccosh(1+np.abs(A2)))
    assert_close(arctanh(_A2), np.arctanh(A2))
    assert_close(exp(_C),  np.exp(C))
    assert_close(exp2(_C), np.exp2(C))
    assert_close(log(_C),  np.log(C))
    assert_close(log2(_C), np.log2(C))
    assert_close(logistic(_A), 1 / (1 + np.exp(-A)))

    # Handle sign and sqrt separately...
    if dt == bool:
        assert_eq(sign(_O), np.sign(np.asarray(O,dtype=uint8)))  # numpy doesn't support sign on type bool
        assert_eq(sign(_A), np.sign(np.asarray(A,dtype=uint8))) 
    else:
        assert_eq(sign(_O), np.sign(O))
        assert_eq(sign(_I), np.sign(I))
        if dt in (int8,int16,int32,int64,float32,float64):
            assert_eq(sign(-_I), np.sign(-I))
        assert_eq(sign(_A), np.sign(A))
        assert_eq(signbit(_O), np.signbit(O,out=np.empty(O.shape,dtype=dt)))
        assert_eq(signbit(_I), np.signbit(I,out=np.empty(I.shape,dtype=dt)))
        if dt in (int8,int16,int32,int64,float32,float64):
            assert_eq(signbit(-_I), np.signbit(-I,out=np.empty(I.shape,dtype=dt)))
        assert_eq(signbit(_A), np.signbit(A,out=np.empty(A.shape,dtype=dt)))
    if dt in dtypes_float:
        assert_close(sqrt(abs(_A)),np.sqrt(np.abs(A)))  # numpy converts integer types to float16/float32/float64, and we don't want that.

#######################################################################

def test_reduce(dt):
    X = np.asarray([[12.5],[1]])
    _X = as_sarray(X)
    assert_eq(sum(_X,axis=1),np.sum(X,axis=1).reshape((-1,1)))


    # Operations that reduce in one or more dimensions.
    reducers = [(max,np.max,assert_eq),
                (min,np.min,assert_eq),
                (sum,np.sum,assert_close),
                (mean,np.mean,assert_close),
                (nnz,np.nnz,assert_eq),
                (any,np.any,assert_eq),
                (all,np.all,assert_eq),
                ]
    shapes = [_A.shape,(-1,1),(3,-1),(-1,3),(-1,7),(1,-1),(7,-1)]

    for shape in shapes:
        for sreduce,nreduce,check in reducers:
            _X = _A.reshape(shape).copy(); _X.ravel()[5:100] = 0;
            X  =  A.reshape(shape).copy();  X.ravel()[5:100] = 0; 
            assert_eq(_X,X)
            check(sreduce(_X,axis=1), nreduce(X,axis=1).reshape((-1,1)))  # reshape because we don't want to follow numpy's convention of turning all reduces into dimension-1 vector
            check(sreduce(_X,axis=0), nreduce(X,axis=0).reshape((1,-1)))
            check(sreduce(_X),        nreduce(X))

#######################################################################

def test_trace(dt):
    #assert_eq(trace(_I), np.trace(I))            # not yet implemented
    pass

#######################################################################

def test_diff(dt):
    for axis in (0,1):
        if axis == 1: continue # TODO: axis=1 not yet implemented
        for n in range(5):
            assert_eq(diff(_A,n,axis=axis), np.diff(A,n,axis=axis))

#######################################################################

def test_repeat(dt):
    for n in range(5): assert_eq(repeat(_A,n,axis=1), np.repeat(A,n,axis=1))
    for n in range(5): assert_eq(repeat(_A,n), np.repeat(A,n).reshape((-1,1)))
    # TODO: axis=0 not yet implemented

#######################################################################

def test_tile(dt):
    for n in range(5): assert_eq(tile(_A,n,axis=1), np.tile(A,(1,n)))
    for n in range(5): assert_eq(tile(_A,n), np.tile(A.reshape((-1,1)),n).reshape((-1,1)))
    # TODO: axis=0 not yet implemented

#######################################################################

def test_arithmetic(dt):
    # Arithmetic operators (+,-,*,/)
    _X,X = _A,A
    _Y,Y = _B,B
    _D,D = _C,C
    if dt in dtypes_sint:
        _Y,Y = abs(_Y),abs(Y)  # cuda/numpy differ on how signed integer types
        _D,D = abs(_D),abs(D)  # are rounded under division, so skip that comparison
    assert_eq(_X+_Y, X+Y)
    assert_eq(_X+_Y[5,:], X+Y[5,:])     # test broadcast of row vector
    assert_eq(_X[0,:]+_Y, X[0,:]+Y)     # test broadcast of row vector
    assert_eq(_X+_W, X+W)              # test broadcast of col vector
    assert_eq(_X+3 , np.asarray(X+3,dtype=dt))
    assert_eq(3+_X , np.asarray(3+X,dtype=dt))
    assert_eq(_X-_Y, X-Y)
    assert_eq(_X-_Y[5,:], X-Y[5,:])
    assert_eq(_X[0,:]-_Y, X[0,:]-Y)
    assert_eq(_X-_W, X-W)
    assert_eq(_X-3 , X-np.asarray(3,dtype=dt))
    assert_eq(3-_X , np.asarray(3,dtype=dt)-X)
    assert_eq(_X*_Y, X*Y)
    assert_eq(_X*_Y[5,:], X*Y[5,:])
    assert_eq(_X[0,:]*_Y, X[0,:]*Y)
    assert_eq(_X*_W, X*W)
    assert_eq(_X*3 , X*np.asarray(3,dtype=dt))
    assert_eq(3*_X , np.asarray(3,dtype=dt)*X)
    assert_close(_Y/_D[5,:], Y/D[5,:])
    assert_close(_Y[0,:]/_D, Y[0,:]/D)
    assert_close(_Y/_W, Y/W)
    assert_close(_Y/_D, np.asarray(Y/D,dtype=dt))
    assert_close(_Y/3 , np.asarray(Y/np.asarray(3,dtype=dt),dtype=dt))
    assert_close(3/_D , np.asarray(np.asarray(3,dtype=dt)/D,dtype=dt))

    if dt != bool:
        _X = _A.copy(); X = A.copy(); _X +=  2; X += 2;  assert_eq(_X,X)
        _X = _A.copy(); X = A.copy(); _X += _C; X += C;  assert_eq(_X,X)
        _X = _A.copy(); X = A.copy(); _X -=  2; X -= 2;  assert_eq(_X,X)
        _X = _A.copy(); X = A.copy(); _X -= _C; X -= C;  assert_eq(_X,X)
    _X = _A.copy(); X = A.copy(); _X *=  2; X *= 2;  assert_eq(_X,X)
    _X = _A.copy(); X = A.copy(); _X *= _C; X *= C;  assert_eq(_X,X)
    _X = _A.copy(); X = A.copy(); _X *=  0; X *= 0;  assert_eq(_X,X)
    _X = _A.copy(); X = A.copy(); _X *=  1; X *= 1;  assert_eq(_X,X)
    _X = _A.copy(); X = A.copy(); _X /=  1; X /= 1;  assert_eq(_X,X)

#######################################################################

def test_elemwise_minmax(dt):
    # Elementwise minimum/maximum
    assert_eq(maximum(_A, 9),np.maximum(A,np.asarray(9,dtype=dt)).astype(dt))
    assert_eq(maximum( 9,_B),np.maximum(np.asarray(9,dtype=dt),B).astype(dt))
    assert_eq(maximum(_A,_B),np.maximum(A,B))
    assert_eq(minimum(_A, 9),np.minimum(A,np.asarray(9,dtype=dt)).astype(dt))
    assert_eq(minimum( 9,_B),np.minimum(np.asarray(9,dtype=dt),B).astype(dt))
    assert_eq(minimum(_A,_B),np.minimum(A,B))

#######################################################################

def test_pow(dt):
    if dt in [int64,uint64]: # Currently not work well with int64 and compute capability 1.2 (no doubles)
        return
    # Power (**). 
    _X,X = abs(_A),np.abs(A); 
    _Y,Y = (_I[:21,:].reshape((-1,21))+1.2).astype(dt),(I[:21,:].reshape((-1,21))+1.2).astype(dt)
    assert_close(_X**_Y, X**Y)
    assert_close(_X**_Y[0,:], X**Y[0,:])  # broadcast
    assert_close(_X**2.1 , (X**np.asarray(2.1,dtype=dt)).astype(dt))
    assert_close(7**_Y , np.asarray(7**Y,dtype=dt))

#######################################################################

def test_softmax(dt):
    assert_close(softmax(_A,axis=0),numpy_softmax(A,axis=0))
    assert_close(softmax(_A,axis=1),numpy_softmax(A,axis=1))

#######################################################################

def test_apply_mask(dt):
    for _ in range(5):
        # smat version
        _X = _A.copy()
        _M = bernoulli(_A.shape, 0.8, dtype=np.bool)
        _X[5:7] = np.nan
        _M[5:7] = False
        apply_mask(_X, _M)

        # numpy version
        X = A.copy()
        X[5:7] = 0
        X *= _M.asnumpy()
        X[np.where(X==-0.)] = 0

        # compare
        assert_eq(_X, X)

#######################################################################

def test_memory_manager():
    reset_backend()
    #reset_backend(verbose=1,log=["heap"])  # for debugging, if there's a problem

    size = 10*1024*1024    # 10 million element chunks
    m = 1024
    n = size/m
    
    status0 = get_heap_status()
    Y = ones((n,m),dtype=float32)
    status1 = get_heap_status()
    Y = None
    status2 = get_heap_status()
    Y = ones((n,m),dtype=float32)
    status3 = get_heap_status()
    Y = None
    status4 = get_heap_status()
    Y = ones((n,3*m//4),dtype=float32)
    status5 = get_heap_status()
    Y = None
    status6 = get_heap_status()
    assert status1.device_used      >= status0.device_used + n*m  # use >= n*m instead of == n*m because sanity checks/alignment constraints might allocate a few extra bytes
    assert status1.device_committed >= status0.device_committed
    assert status2.device_used      == status0.device_used
    assert status2.device_committed == status1.device_committed
    assert status3.device_used      == status1.device_used    
    assert status3.device_committed == status1.device_committed
    assert status4.device_used      == status0.device_used    
    assert status4.device_committed == status1.device_committed
    assert status5.device_used      <  status1.device_used     # allocated smaller array, but should use same block
    assert status5.device_committed == status1.device_committed
    assert status6.device_used      == status0.device_used    
    assert status6.device_committed == status1.device_committed

    for i in range(2):   # try to alloc and free all memory, several times
        # Each trial allocates (and continues to reference)
        # enough matrix data to nearly fill the available device memory,
        # then syncs with the machine.
        mem = get_heap_status()
        X = []
        Y = ones((n,m),dtype=float32)
        elem_to_alloc = int(mem.device_avail*0.9)/4
        chunks_to_alloc = elem_to_alloc/size-2
        for j in range(chunks_to_alloc):
            X.append(ones((n,m),dtype=float32))
            Y = Y + X[-1]
        sync()
        X = None
        Y = None
        sync()

    reset_backend()

#######################################################################

def run_unittest(test,dtypes=None):
    print rpad("%s..." % test.__name__.partition("_")[2],19),
    if dtypes == None:
        test()
    else:
        supported = get_supported_dtypes()
        for dt in [bool, int8, int16, int32, int64,
                        uint8,uint16,uint32,uint64,float32,float64]:
            if not dt in supported:
                continue
            print ("%3s" % dtype_short_name[dt] if dt in dtypes else "   "),
            if dt in dtypes:
                alloc_test_matrices(dt)
                test(dt)
                clear_test_matrices()
    print

#######################################################################

def unittest():
    print '\n---------------------- UNIT TESTS -------------------------\n'
    np.random.seed(42)
    set_backend_options(randseed=42,verbose=0,sanitycheck=False)
    
    run_unittest(test_memory_manager)
    run_unittest(test_create        ,dtypes_generic)
    run_unittest(test_copy          ,dtypes_generic)
    run_unittest(test_random        ,dtypes_generic)
    run_unittest(test_random_int    ,dtypes_integral)
    run_unittest(test_random_float  ,dtypes_float)
    run_unittest(test_smat_extension,dtypes_float)
    run_unittest(test_closeness     ,dtypes_float)
    run_unittest(test_attributes)
    run_unittest(test_serialize     ,dtypes_generic)
    run_unittest(test_slicing       ,dtypes_generic)
    run_unittest(test_reshape       ,dtypes_generic)
    run_unittest(test_transpose     ,dtypes_generic)
    run_unittest(test_dot           ,dtypes_float)
    run_unittest(test_bitwise       ,dtypes_integral)
    run_unittest(test_logical       ,dtypes_integral)
    run_unittest(test_modulo        ,dtypes_integral)
    run_unittest(test_naninf        ,dtypes_float)
    run_unittest(test_math_float    ,dtypes_float)
    run_unittest(test_reduce        ,dtypes_generic)
    run_unittest(test_trace         ,dtypes_generic)
    run_unittest(test_diff          ,dtypes_generic)
    run_unittest(test_repeat        ,dtypes_generic)
    run_unittest(test_tile          ,dtypes_generic)
    run_unittest(test_arithmetic    ,dtypes_generic)
    run_unittest(test_elemwise_minmax,dtypes_generic)
    run_unittest(test_pow           ,dtypes_generic)
    run_unittest(test_softmax       ,dtypes_float)
    run_unittest(test_apply_mask    ,dtypes_float)
    #run_unittest(test_repmul_iadd   ,dtypes_float)

