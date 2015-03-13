from smat_dll import *
from ctypes import *
from exceptions import *
import numpy as np
import string,atexit,__builtin__
from copy import copy,deepcopy
import cPickle as pickle
import util
import gc

bool   = np.bool
int8   = np.int8
uint8  = np.uint8
int16  = np.int16
uint16 = np.uint16
int32  = np.int32
uint32 = np.uint32
int64  = np.int64
uint64 = np.uint64
float32= np.float32
float64= np.float64
index   = int32   # should be same as c_index_t
uindex  = uint32  # should be same as c_uindex_t

tic = util.tic
toc = util.toc

_int2dtype = util._int2dtype
_dtype2int = util._dtype2int
_arg2dtype = util._arg2dtype

def int2dtype(i):
    return _int2dtype[i]

def dtype2int(dt):
    if type(dt) == type(None):return -1
    return _dtype2int[arg2dtype(dt)]

def dtype2str(dtype):
    dtype = _int2dtype(dll.actual_dtype(_dtype2int[dtype]))
    return dtype.__name__

def arg2dtype(s):
    # Can't just test s == None due to some weird bug where "np.dtype('float64') == None" returns true
    if type(s) == type(None):   return _int2dtype[dll.api_get_default_dtype()]
    return _arg2dtype[s]

_integral_types = (__builtin__.chr,__builtin__.int,__builtin__.long,int8,int16,int32,int64)
_py_max = __builtin__.max
_py_min = __builtin__.min
_py_sum = __builtin__.sum
_py_all = __builtin__.all
_py_any = __builtin__.any

_smat_exiting = False
_span = c_slice_t(0,c_slice_end)

_axis2int = { None: -1, 0:1, 1:0 }  # Only x and y axes supported right now

##############################################################

class sarray(object):
    '''
    A numpy-like wrapper around a streaming-mode matrix (smat) object.
    '''
    def __init__(self,ptr):
        self._ptr = ptr   # Just wrap a pointer to the C++ smat instance.
        self._attr = None # List of "extra" attributes that have been attached to this smat instance, if any

    def __del__(self):
        if dll != None and not _smat_exiting:  # dll may have already been unloaded if exiting application
            dll.api_delete(self._ptr)

    @property
    def size(self): return dll.api_size(self._ptr)

    @property
    def shape(self):
        s = c_shape_t(0,0,0)
        dll.api_shape(self._ptr,byref(s))
        return (s.y,s.x)

    @property
    def nrow(self): return dll.api_nrow(self._ptr) # number of rows

    @property
    def ncol(self): return dll.api_ncol(self._ptr) # number of cols

    @property
    def ndim(self): return 2   # TODO: support variable dimension arrays

    @property
    def dtype(self): return np.dtype(_int2dtype[dll.api_dtype(self._ptr)])

    @property
    def T(self): return transpose(self)

    def setattr(self,name,val):
        if self._attr is None:
            self._attr = set()
        self._attr.add(name)
        self.__setattr__(name,val)

    def getattr(self,name,default=None):
        return getattr(self,name) if self.hasattr(name) else default
    
    def hasattr(self,name):
        return self._attr is not None and name in self._attr
    
    def clearattr(self):
        if self._attr:
            for attr in copy(self._attr):
                delattr(self,attr)

    def copyattr(self,A,deep=False):
        if A is self:
            return
        self.clearattr()
        if A._attr:
            for attr in A._attr:
                val = A.getattr(attr)
                if deep:
                    val = deepcopy(val)
                self.setattr(attr,val)

    def copy(self): return deepcopy(self)

    def __delattr__(self,name):
        object.__delattr__(self,name)
        if self._attr is not None:
            self._attr.remove(name)

    # These provided for the standard library copy() and deepcopy() functions
    def __deepcopy__(self,memo):
        A = empty(self.shape,self.dtype)  # deep copy, including all data and extra attributes
        A[:] = self
        sync()
        if self._attr is not None:
            for attr in self._attr:
                A.setattr(attr,deepcopy(getattr(self,attr),memo))
        return A 

    def __copy__(self):  # shallow copy, where things like shape can be modified separately from the original, but shares the same underlying data array
        A = self[:]
        sync()
        if self._attr is not None:
            for attr in self._attr:
                A.setattr(attr,getattr(self,attr))
        return A

    def astype(self,dtype,copy=False): 
        return as_sarray(self,dtype,copy)

    def asnumpy(self,async=False,out=None):
        if   out == None:                    out = np.empty(self.shape,dtype=self.dtype,order='C')
        elif not isinstance(out,np.ndarray): raise ValueError("Keyword argument 'out' must be a numpy array.\n")
        elif out.shape != A.shape:           raise ValueError("Keyword argument 'out' must have matching dimensions.\n")
        if out.ndim > 1:
            rstride = out.strides[0]
            cstride = out.strides[1]
        else:
            cstride = out.strides[0]
            rstride = cstride*A.size
        # Looks good, so issue the copy operation.
        dll.api_copy_to(self._ptr,out.ctypes.data_as(c_void_p),rstride,cstride)
        if not async:
            sync()  # Wait for writes to finish.
        return out

    def isscalar(self):
        return self.size == 1
    
    # Do NOT provide len() and iteration, because it gives numpy a way to 
    # silently cause terrible performance problems when mixed with sarrays.
    
    def __len__(self):
        #raise NotImplementedError("An sarray does not support __len__, to prevent accidental operations mixing numpy ndarrays.")
        return self.nrow if self.nrow > 1 else max(self.nrow,self.ncol)
    
    def __iter__(self): 
        #raise NotImplementedError("An sarray does not support __iter__, to prevent accidental operations mixing numpy ndarrays.")
        for row in xrange(self.shape[0]):  # Iterate over rows
            yield self[row]

    ############################### PICKLING ##############################

    def __getstate__(self):
        data = self.asnumpy()
        attrdict = {name : pickle.dumps(self.getattr(name)) for name in self._attr} if self._attr is not None else None
        return (data,attrdict)

    def __setstate__(self,state):
        data,attrdict = state
        self._ptr = dll.api_empty(_make_shape_p(data.shape),dtype2int(data.dtype))
        dll.api_copy_from(self._ptr,data.ctypes.data_as(c_void_p),data.strides[0],data.strides[1]) # Copy from data
        self._attr = None
        if attrdict:
            for name,value in attrdict.items():
                self.setattr(name,pickle.loads(value))
    
    ############################### RESHAPING ##############################

    def reshape(self,shape):
        return sarray(dll.api_reshape(self._ptr,_make_shape_p(shape)))

    def ravel(self):
        return sarray(dll.api_reshape(self._ptr,_make_shape_p((-1,1))))

    ############################### SLICING ################################

    def __getitem__(self,i):
        ti = type(i)
        if ti != tuple:
            # One slice dimension
            if _is_full_slice(i):
                return self
            rows = _make_gettable_slice_p(i)
            cols = _span
        else:
            # Two slice dimensions
            if len(i) != 2:
                raise IndexError("Too many indices.\n")
            rows = _make_gettable_slice_p(i[0])
            cols = _make_gettable_slice_p(i[1]) if i[1] != slice(None) else _make_gettable_slice_p(slice(0,self.ncol))

        if type(rows) == c_slice_t and type(cols) == c_slice_t:
            # Contiguous row indices, contiguous col indices
            return sarray(dll.api_slice(self._ptr,byref(rows),byref(cols)))
        elif type(rows) == c_slice_t and type(cols) == np.ndarray:
            # Contiguous row indices, list of individual col indices
            raise NotImplementedError("List-based slicing not implemented")
        else:
            raise NotImplementedError("Gettable slicing only supports slice-, integer-, or list-based indexing.")


    def __setitem__(self,i,val):
        ti = type(i)
        if ti != tuple:
            # One slice dimension
            if _is_full_slice(i):
                self.assign(val)
                return
            rows = _make_settable_slice_p(i)
            cols = _span
        else:
            # Two slice dimensions
            if len(i) != 2:
                raise IndexError("Too many indices.\n")
            rows = _make_settable_slice_p(i[0])
            cols = _make_settable_slice_p(i[1]) if i[1] != slice(None) else _make_settable_slice_p(slice(0,self.ncol))
        
        if type(rows) == c_slice_t and type(cols) == c_slice_t:
            # Contiguous row indices, contiguous col indices
            view = sarray(dll.api_slice(self._ptr,byref(rows),byref(cols)))
            view.assign(val)  # use .assign to avoid recursion
        else:
            raise NotImplementedError("Settable slicing only supports slice- or integer-based indexing.")

    def assign(self,val):
        val = as_sarray(val)
        dll.api_assign(self._ptr,val._ptr)
        return self

    ######################## COMPARISON OPERATIONS #########################

    def __eq__(self,other): 
        if isscalar(other):       other = _scalar2smat(other)
        if type(other) == sarray: return sarray(dll.api_eq(self._ptr,other._ptr))
        return False

    def __ne__(self,other): return self.__eq__(other).__invert__()
    def __lt__(self,other): other = _scalar2smat(other); return sarray(dll.api_lt(self._ptr,other._ptr))
    def __le__(self,other): other = _scalar2smat(other); return sarray(dll.api_le(self._ptr,other._ptr))
    def __gt__(self,other): other = _scalar2smat(other); return sarray(dll.api_gt(self._ptr,other._ptr))
    def __ge__(self,other): other = _scalar2smat(other); return sarray(dll.api_ge(self._ptr,other._ptr))

    ######################## LOGICAL/BITWISE OPERATIONS #########################

    def __or__(self,other):  other = _scalar2smat(other); return sarray(dll.api_or (self._ptr,other._ptr))
    def __xor__(self,other): other = _scalar2smat(other); return sarray(dll.api_xor(self._ptr,other._ptr))
    def __and__(self,other): other = _scalar2smat(other); return sarray(dll.api_and(self._ptr,other._ptr))

    def __ror__(self,other):  other = _scalar2smat(other); return sarray(dll.api_or (other._ptr,self._ptr))
    def __rxor__(self,other): other = _scalar2smat(other); return sarray(dll.api_xor(other._ptr,self._ptr))
    def __rand__(self,other): other = _scalar2smat(other); return sarray(dll.api_and(other._ptr,self._ptr))

    def __ior__(self,other):  other = _scalar2smat(other); dll.api_ior (self._ptr,other._ptr); return self
    def __ixor__(self,other): other = _scalar2smat(other); dll.api_ixor(self._ptr,other._ptr); return self
    def __iand__(self,other): other = _scalar2smat(other); dll.api_iand(self._ptr,other._ptr); return self

    ########################## UNARY OPERATORS ##########################

    def __neg__(self):   return sarray(dll.api_neg(self._ptr))
    def __abs__(self):   return sarray(dll.api_abs(self._ptr))
    def __invert__(self):return sarray(dll.api_not(self._ptr))
    def __nonzero__(self):
        if self.size != 1:
            raise ValueError("Truth value of matrix is ambiguous; use all() or any().")
        return self.asnumpy().__nonzero__()  # must pull back from device
    def __int__(self):   return int(self.asnumpy())
    def __long__(self):  return long(self.asnumpy())
    def __float__(self): return float(self.asnumpy())

    ########################## ARITHMETIC OPERATORS #########################

    def __add__(self,other): other = _scalar2smat(other); return sarray(dll.api_add(self._ptr,other._ptr))
    def __sub__(self,other): other = _scalar2smat(other); return sarray(dll.api_sub(self._ptr,other._ptr))
    def __mul__(self,other): other = _scalar2smat(other); return sarray(dll.api_mul(self._ptr,other._ptr))
    def __div__(self,other): other = _scalar2smat(other); return sarray(dll.api_div(self._ptr,other._ptr))
    def __mod__(self,other): other = _scalar2smat(other); return sarray(dll.api_mod(self._ptr,other._ptr))
    def __pow__(self,other): other = _scalar2smat(other); return sarray(dll.api_pow(self._ptr,other._ptr))

    def __radd__(self,other): other = _scalar2smat(other); return sarray(dll.api_add(other._ptr,self._ptr))
    def __rsub__(self,other): other = _scalar2smat(other); return sarray(dll.api_sub(other._ptr,self._ptr))
    def __rmul__(self,other): other = _scalar2smat(other); return sarray(dll.api_mul(other._ptr,self._ptr))
    def __rdiv__(self,other): other = _scalar2smat(other); return sarray(dll.api_div(other._ptr,self._ptr))
    def __rmod__(self,other): other = _scalar2smat(other); return sarray(dll.api_mod(other._ptr,self._ptr))
    def __rpow__(self,other): other = _scalar2smat(other); return sarray(dll.api_pow(other._ptr,self._ptr))

    def __iadd__(self,other): other = _scalar2smat(other); dll.api_iadd(self._ptr,other._ptr); return self
    def __isub__(self,other): other = _scalar2smat(other); dll.api_isub(self._ptr,other._ptr); return self
    def __imul__(self,other): other = _scalar2smat(other); dll.api_imul(self._ptr,other._ptr); return self
    def __idiv__(self,other): other = _scalar2smat(other); dll.api_idiv(self._ptr,other._ptr); return self
    def __imod__(self,other): other = _scalar2smat(other); dll.api_imod(self._ptr,other._ptr); return self
    def __ipow__(self,other): other = _scalar2smat(other); dll.api_ipow(self._ptr,other._ptr); return self

    ########################## REDUCE OPERATIONS ##########################

    def max(self,axis=None):  return sarray(dll.api_max( self._ptr,_axis2int[axis]))  #.reshape((1,-1))   # mimick numpy's conversion to 1d row vector ? Naaaah, it's too annoying; pretend keep_dim is on by default
    def min(self,axis=None):  return sarray(dll.api_min( self._ptr,_axis2int[axis]))
    def sum(self,axis=None):  return sarray(dll.api_sum( self._ptr,_axis2int[axis]))
    def mean(self,axis=None): return sarray(dll.api_mean(self._ptr,_axis2int[axis]))
    def nnz(self,axis=None):  return sarray(dll.api_nnz( self._ptr,_axis2int[axis]))
    def any(self,axis=None):  return sarray(dll.api_any( self._ptr,_axis2int[axis]))
    def all(self,axis=None):  return sarray(dll.api_all( self._ptr,_axis2int[axis]))

    ########################## REPEAT OPERATORS ##########################
    
    def _rep_op(self,n,axis,op):
        if axis not in (None,0,1): raise ValueError("Axis must be None, 0 or 1.")
        A = self
        if isinstance(n,(tuple,list)):
            if axis is not None: raise ValueError("Axis must be None if n is a tuple")
            if len(n) == 1:
                n = (n[0],1) if axis == 0 else (1,n[0])
        else:
            if axis is None:
                A = self.ravel()  # emulate numpy flattening on axis=None
            n = (n,1) if axis == 0 else (1,n)
        B = sarray(op(A._ptr,_make_shape_p(n)))
        return B if axis is not None else B.reshape((-1,1))
        
    def repeat(self,n,axis=None): return self._rep_op(n,axis,dll.api_repeat)
    def tile(self,n,axis=None):   return self._rep_op(n,axis,dll.api_tile)

    ########################## OTHER OPERATORS ##########################

    def __repr__(self):
        max_device_rows = 512 if self.shape[1] > 16 else 2048
        if True or self.shape[0] <= max_device_rows:
            A = self.asnumpy()
        else:
            # If this is a huge matrix, only copy the start and end of the matrix to the host,
            # so that printing is faster, and so that interactive debuggers like Visual Studio
            # are faster (otherwise have to wait for huge memory transfers at each breakpoint,
            # to update the variable values in Visual Studio's Locals window).
            # For now, just handle the case when there are many rows.
            A = np.empty((max_device_rows,)+self.shape[1:],self.dtype)
            A[:max_device_rows/2] = self[:max_device_rows/2].asnumpy()
            A[max_device_rows/2:] = self[-max_device_rows/2:].asnumpy()
        txt = A.__repr__().replace('array(', 'sarray(').replace(' [','  [')
        if txt.find("dtype=") == -1:
            txt = txt[:-1] + (",dtype=%s)" % A.dtype)
        return txt
    
    def __str__(self):  return self.asnumpy().__str__()

##############################################################

def empty(shape,dtype=None): return sarray(dll.api_empty(_make_shape_p(shape),dtype2int(dtype)))
def zeros(shape,dtype=None): return sarray(dll.api_zeros(_make_shape_p(shape),dtype2int(dtype)))
def ones (shape,dtype=None): return sarray(dll.api_ones (_make_shape_p(shape),dtype2int(dtype)))
def empty_like(A,dtype=None):return sarray(dll.api_empty_like(A._ptr,dtype2int(dtype)))
def zeros_like(A,dtype=None):return sarray(dll.api_zeros_like(A._ptr,dtype2int(dtype)))
def ones_like (A,dtype=None):return sarray(dll.api_ones_like (A._ptr,dtype2int(dtype)))

def eye  (n,m=None,k=0,dtype=None): 
    if _dtype2int.has_key(m): dtype = m; m = None
    if m != None and n != m: raise NotImplementedError("Non-square identity matrices not supported.\n")
    if k != 0: raise NotImplementedError("Off-diagonal identity matrices not supported.\n")
    return sarray(dll.api_eye(n,dtype2int(dtype)))

def identity(n,dtype=None): 
    return sarray(dll.api_eye(n,dtype2int(dtype)))

def arange(*args,**kwargs):
    if len(args) == 0: raise ValueError("Not enough arguments.\n")
    if len(args) == 1: start = 0;       stop = args[0]
    if len(args) == 2: start = args[0]; stop = args[1]
    return sarray(dll.api_arange(start,stop,dtype2int(kwargs.get("dtype",None))))

def rand(n,m=1,dtype=None):        return sarray(dll.api_rand(     _make_shape_p((int(n),int(m))),dtype2int(dtype)))
def randn(n,m=1,dtype=None):       return sarray(dll.api_randn(    _make_shape_p((int(n),int(m))),dtype2int(dtype)))
def bernoulli(shape,p,dtype=None): return sarray(dll.api_bernoulli(_make_shape_p(shape),p,dtype2int(dtype)))
def rand_seed(seed):           dll.api_set_rand_seed(seed)
def sync():  dll.api_sync()

###############################################################

# Maps python/numpy type to corresponding smat scalar constructor.
_smat_const_lookup = {
    __builtin__.bool:   (dll.api_const_b8, c_bool),
                bool:   (dll.api_const_b8, c_bool),
    __builtin__.chr:    (dll.api_const_i8, c_byte),
                int8:   (dll.api_const_i8, c_byte),
                uint8:  (dll.api_const_u8, c_ubyte),
                int16:  (dll.api_const_i16,c_short),
                uint16: (dll.api_const_u16,c_ushort),
    __builtin__.int:    (dll.api_const_i32,c_int),
                int32:  (dll.api_const_i32,c_int), 
                uint32: (dll.api_const_u32,c_uint),
    __builtin__.long:   (dll.api_const_i64,c_longlong),
                int64:  (dll.api_const_i64,c_longlong),
                uint64: (dll.api_const_u64,c_ulonglong),
                float32:(dll.api_const_f32,c_float),
                float64:(dll.api_const_f64,c_double),
    __builtin__.float:  (dll.api_const_f64,c_double)
    }

def as_sarray(A,dtype=None,copy=False,force=False):
    if isinstance(A,sarray):
        # Type SARRAY (smat)
        if dtype is None or A.dtype == dtype:
            return A if not copy else A.copy()  # Return a reference to A, or a direct copy
        B = empty(A.shape,dtype)
        B[:] = A       # Return type-converted copy of A
        return B
    if isinstance(A,list):
        if dtype is None:
            if type(A[0]) == float or (isinstance(A[0],(list,tuple)) and type(A[0][0]) == float):
                dtype = get_default_dtypef() # Convert to "float32" or "float64" depending on current default for floats
        A = np.asarray(A,dtype=dtype)        # Let numpy do the dirty work of rearranging the data, and fall through to the next if statement.
    if isinstance(A,np.ndarray):
        # Type NDARRAY (numpy)
        if dtype != None and dtype != A.dtype:
            A = np.require(A,dtype=dtype)  # Implicit conversion first, since simultaneous copy-and-convert is not supported by smat.
        if A.ndim > 2:
            raise NotImplementedError("Only 1- or 2-D sarrays are supported.")
        if not A.flags['C_CONTIGUOUS']:
            if not force:
                raise TypeError("Expected C-contiguous ndarray, but received F-contiguous; use force=True to allow automatic conversion.")
            A = np.require(A,requirements=["C_CONTIGUOUS"])
        if A.ndim > 1:
            rstride = A.strides[0]
            cstride = A.strides[1]
        else:
            cstride = A.strides[0]
            rstride = cstride*A.size
        B = empty(A.shape,A.dtype)
        dll.api_copy_from(B._ptr,A.ctypes.data_as(c_void_p),rstride,cstride) # Return copy of A
        return B
    if np.isscalar(A):
        # Type SCALAR; convert to a 1x1 sarray of the appropriate type
        func,ctype = _smat_const_lookup[type(A) if dtype is None else arg2dtype(dtype)]
        b = sarray(func(ctype(A)))               # Return scalar wrapped in an smat
        return b
    raise TypeError("Unrecognized type '%s'.\n" % str(type(A)))

asarray = as_sarray
array = as_sarray

def index_array(A):  return as_sarray(A,dtype=index)
def uindex_array(A): return as_sarray(A,dtype=uindex)

def asnumpy(A,async=False,out=None):
    try:
        if isinstance(A,list):   return list(as_numpy(item) for item in A)
        if isinstance(A,tuple):  return tuple(as_numpy(item) for item in A)
        if isinstance(A,sarray): return A.asnumpy(async,out)
        if out != None: raise ValueError("Keyword argument 'out' only supported when input is of type sarray.")
        # If not an SARRAY, pass it along to the regular numpy asarray() function
        return np.asarray(A) if A is not None else None
    except MemoryError as mem:
        print ("OUT OF MEMORY in asnumpy() with A=%s (%d bytes)" % (str(A.shape),A.size))
        raise

as_numpy = asnumpy

def as_numpy_array(A): return as_numpy(A)  # gnumpy calls it as_numpy_array 

def isarray(x): return type(x) == sarray

def isscalar(x):
    if type(x) == sarray:  return x.isscalar()
    if type(x) == str: return False  # for some reason np.isscalar returns true for strings
    return np.isscalar(x)

def sign(A):      return sarray(dll.api_sign(A._ptr))       if isinstance(A,sarray) else np.sign(A)
def signbit(A):   return sarray(dll.api_signbit(A._ptr))    if isinstance(A,sarray) else np.signbit(A,out=np.empty(A.shape,A.dtype)) # force numpy to use input dtype instead of bool
def sqrt(A):      return sarray(dll.api_sqrt(A._ptr))       if isinstance(A,sarray) else np.sqrt(A)
def square(A):    return sarray(dll.api_square(A._ptr))     if isinstance(A,sarray) else np.square(A)
def sin(A):       return sarray(dll.api_sin(A._ptr))        if isinstance(A,sarray) else np.sin(A)
def cos(A):       return sarray(dll.api_cos(A._ptr))        if isinstance(A,sarray) else np.cos(A)
def tan(A):       return sarray(dll.api_tan(A._ptr))        if isinstance(A,sarray) else np.tan(A)
def arcsin(A):    return sarray(dll.api_arcsin(A._ptr))     if isinstance(A,sarray) else np.arcsin(A)
def arccos(A):    return sarray(dll.api_arccos(A._ptr))     if isinstance(A,sarray) else np.arccos(A)
def arctan(A):    return sarray(dll.api_arctan(A._ptr))     if isinstance(A,sarray) else np.arctan(A)
def sinh(A):      return sarray(dll.api_sinh(A._ptr))       if isinstance(A,sarray) else np.sinh(A)
def cosh(A):      return sarray(dll.api_cosh(A._ptr))       if isinstance(A,sarray) else np.cosh(A)
def tanh(A):      return sarray(dll.api_tanh(A._ptr))       if isinstance(A,sarray) else np.tanh(A)
def arcsinh(A):   return sarray(dll.api_arcsinh(A._ptr))    if isinstance(A,sarray) else np.arcsinh(A)
def arccosh(A):   return sarray(dll.api_arccosh(A._ptr))    if isinstance(A,sarray) else np.arccosh(A)
def arctanh(A):   return sarray(dll.api_arctanh(A._ptr))    if isinstance(A,sarray) else np.arctanh(A)
def exp(A):       return sarray(dll.api_exp(A._ptr))        if isinstance(A,sarray) else np.exp(A)
def exp2(A):      return sarray(dll.api_exp2(A._ptr))       if isinstance(A,sarray) else np.exp2(A)
def log(A):       return sarray(dll.api_log(A._ptr))        if isinstance(A,sarray) else np.log(A)
def log2(A):      return sarray(dll.api_log2(A._ptr))       if isinstance(A,sarray) else np.log2(A)
def logistic(A):  return sarray(dll.api_logistic(A._ptr))   if isinstance(A,sarray) else 1/(1+np.exp(-A))
def round(A):     return sarray(dll.api_round(A._ptr))      if isinstance(A,sarray) else np.round(A)
def floor(A):     return sarray(dll.api_floor(A._ptr))      if isinstance(A,sarray) else np.floor(A)
def ceil(A):      return sarray(dll.api_ceil(A._ptr))       if isinstance(A,sarray) else np.ceil(A)
def clip(A,lo=0.,hi=1.):return sarray(dll.api_clip(A._ptr,lo,hi)) if isinstance(A,sarray) else np.clip(A,lo,hi)
def isinf(A):     return sarray(dll.api_isinf(A._ptr))      if isinstance(A,sarray) else np.isinf(A)
def isnan(A):     return sarray(dll.api_isnan(A._ptr))      if isinstance(A,sarray) else np.isnan(A)
def transpose(A): return sarray(dll.api_trans(A._ptr))      if isinstance(A,sarray) else np.transpose(A)
def dot(A,B):     return sarray(dll.api_dot(A._ptr,B._ptr)) if isinstance(A,sarray) and isinstance(B,sarray) else np.dot(A,B)
def dot_tn(A,B):  return sarray(dll.api_dot_tn(A._ptr,B._ptr)) if isinstance(A,sarray) and isinstance(B,sarray) else np.dot(A.T,B)
def dot_nt(A,B):  return sarray(dll.api_dot_nt(A._ptr,B._ptr)) if isinstance(A,sarray) and isinstance(B,sarray) else np.dot(A,B.T)
def dot_tt(A,B):  return sarray(dll.api_dot_tt(A._ptr,B._ptr)) if isinstance(A,sarray) and isinstance(B,sarray) else np.dot(A.T,B.T)

def _binary_elemwise(sop,nop,A,B,*args):
    if type(A) == sarray and np.isscalar(B): B = as_sarray(B,dtype=A.dtype)
    if type(B) == sarray and np.isscalar(A): A = as_sarray(A,dtype=B.dtype)
    if type(A) == sarray and type(B) == sarray: return sarray(sop(A._ptr,B._ptr))
    if nop is not None: return nop(A,B,*args)
    raise RuntimeException("Both arguments should be of type sarray.")

def maximum(A,B): return _binary_elemwise(dll.api_maximum,np.maximum,A,B)
def minimum(A,B): return _binary_elemwise(dll.api_minimum,np.minimum,A,B)

def isclose(A,B,rtol=None,atol=None):
    if rtol == None: rtol = _default_rtol(A.dtype)
    if atol == None: atol = _default_atol(A.dtype)
    return _binary_elemwise(dll.api_isclose,None,A,B,rtol,atol)

def allclose(A,B,rtol=None,atol=None):
    if rtol == None: rtol = _default_rtol(A.dtype)
    if atol == None: atol = _default_atol(A.dtype)
    return _binary_elemwise(dll.api_allclose,np.allclose,A,B,rtol,atol)

def _reduce_op(A,axis,sop,nop,pyop):
    if isinstance(A,sarray):     return sop(A,axis)
    if isinstance(A,np.ndarray): return nop(A,axis)
    if pyop == None:             raise TypeError("Invalid type for reduce operation.")
    if isinstance(A,list) and axis==None: return pyop(A)
    return pyop(A,axis) # A is first item, axis is second item (e.g. call __builtin__.min(A,axis))

def max(A,axis=None):  return _reduce_op(A,axis,sarray.max,np.ndarray.max,_py_max)
def min(A,axis=None):  return _reduce_op(A,axis,sarray.min,np.ndarray.min,_py_min)
def sum(A,axis=None):  return _reduce_op(A,axis,sarray.sum,np.ndarray.sum,_py_sum)
def mean(A,axis=None): return _reduce_op(A,axis,sarray.mean,np.ndarray.mean,None)
def nnz(A,axis=None):  return A.nnz(axis)  if isinstance(A,sarray) else (np.count_nonzero(A) if axis == None else np.sum(A!=0,axis))
def all(A,axis=None):  return _reduce_op(A,axis,sarray.all,np.ndarray.all,_py_all)
def any(A,axis=None):  return _reduce_op(A,axis,sarray.any,np.ndarray.any,_py_any)
def count_nonzero(A):  return A.nnz() if isinstance(A,sarray) else np.count_nonzero(A)

def repeat(A,n,axis=None):
    if isinstance(A,sarray):
        return A.repeat(n,axis)
    return np.repeat(A,n,axis)

def tile(A,n,axis=None):
    if isinstance(A,sarray):
        return A.tile(n,axis)
    assert axis is None
    return np.tile(A,n)

def diff(A,n=1,axis=1):
    if not isinstance(A,sarray): return np.diff(A,n,axis)
    if n <= 0:  return A
    B = diff(A,n-1,axis)
    return sarray(dll.api_diff(B._ptr,_axis2int[axis]))

def softmax(A,axis=1): return sarray(dll.api_softmax(A._ptr,_axis2int[axis]))
def apply_mask(A,mask): dll.api_apply_mask(A._ptr,mask._ptr)

def logical_not(A):     return sarray(dll.api_lnot(A._ptr))        if isinstance(A,sarray) else np.logical_not(A)
def logical_or(A,B):    return sarray(dll.api_lor(A._ptr,B._ptr))  if isinstance(A,sarray) and isinstance(B,sarray) else np.logical_or(A,B)
def logical_and(A,B):   return sarray(dll.api_land(A._ptr,B._ptr)) if isinstance(A,sarray) and isinstance(B,sarray) else np.logical_and(A,B)

###############################################################
# These extra global functions are provided so that there's an
# easy, named function available for all smat operations.

def eq(A,B): return A == B
def ne(A,B): return A != B
def lt(A,B): return A <  B
def le(A,B): return A <= B
def gt(A,B): return A >  B
def ge(A,B): return A >= B
def _or(A,B):  return A | B
def _xor(A,B): return A ^ B
def _and(A,B): return A & B
def _abs(A):   return abs(A)
def invert(A):     return ~A
def reciprocal(A): return 1./A
def negative(A):   return -A
def add(A,B):      return A+B
def subtract(A,B): return A-B
def multiply(A,B): return A*B
def divide(A,B):   return A/B
def mod(A,B):      return A%B
def power(A,B):    return A**B
def max_x(A):  return A.max(axis=1)
def max_y(A):  return A.max(axis=0)
def min_x(A):  return A.min(axis=1)
def min_y(A):  return A.min(axis=0)
def sum_x(A):  return A.sum(axis=1)
def sum_y(A):  return A.sum(axis=0)
def mean_x(A): return A.mean(axis=1)
def mean_y(A): return A.mean(axis=0)
def nnz_x(A):  return A.nnz(axis=1)
def nnz_y(A):  return A.nnz(axis=0)
def any_x(A):  return A.any(axis=1)
def any_y(A):  return A.any(axis=0)
def all_x(A):  return A.all(axis=1)
def all_y(A):  return A.all(axis=0)
def diff_x(A): return A.diff(axis=1)
def diff_y(A): return A.diff(axis=0)
def repeat_x(A,n): return A.repeat(n,axis=1)
def repeat_y(A,n): return A.repeat(n,axis=0)
def tile_x(A,n):   return A.tile(n,axis=1)
def tile_y(A,n):   return A.tile(n,axis=0)
def softmax_x(A):      return softmax(A,axis=1)
def softmax_y(A):      return softmax(A,axis=0)

###############################################################

def _as_tuple(x):      return x if type(x) != tuple else (x,)
def _is_full_slice(x): return type(x) == slice and x.start == None and x.stop == None
def _scalar2smat(x):
    if type(x) == sarray:  return x
    if not np.isscalar(x): raise TypeError("Type %s not directly supported in this operation.\n" % str(type(x)))
    func,ctype = _smat_const_lookup[type(x)]
    b = sarray(func(ctype(x)))                    # Return scalar wrapped in an smat
    return b

def _make_settable_slice_p(x):
    tx = type(x)
    if tx == slice:
        if not x.step in (None, 1):
            raise NotImplementedError("Settable slicing is only supported for contiguous ranges.\n")
        return c_slice_t(x.start or 0L, x.stop if x.stop != None else c_slice_end)
    if tx in _integral_types:
        return c_slice_t(x,x+1)
    if tx == sarray:
        if x.dtype == bool: raise NotImplementedError("Logical slicing not yet implemented.\n")
        else:               raise NotImplementedError("Settable list-based slicing not yet implemented.\n")
    raise NotImplementedError("Settable index must be integral or contiguous slice.\n")

def _make_gettable_slice_p(x):
    tx = type(x)
    if tx == slice:
        if not x.step in (None, 1):
            return np.arange(x.start,x.stop,x.step,dtype=index)
        return c_slice_t(x.start or 0L, x.stop if x.stop != None else c_slice_end)
    if tx in _integral_types:
        return c_slice_t(x,x+1)
    if tx == list or tx == tuple:
        x = np.asarray(x)
        tx = np.ndarray
    if tx == np.ndarray:
        x = as_sarray(x)
        tx = sarray
    if tx == sarray:
        if x.dtype == bool: raise NotImplementedError("Logical slicing not yet implemented.\n")
        if x.ndim != 1:     raise NotImplementedError("List-based slicing must use 1-dimensional vector.")
        return x
    raise NotImplementedError("Gettable index must be integral, slice, or list.\n")


def _make_shape_p(shape):
    if isinstance(shape,int): return byref(c_shape_t(shape,1,1))
    if not isinstance(shape,tuple) or not len(shape) in [1,2]:
        raise ValueError("Shape must be a tuple of length 1 or 2.\n")
    if len(shape) == 1: return byref(c_shape_t(shape[0],1,1))
    return byref(c_shape_t(shape[1],shape[0],1))

def _kwargs2argv(kwargs):
    as_str = lambda v: str(val) if not isinstance(val,list) else string.join([str(v) for v in val],",")
    args = [key + '=' + as_str(val) for key,val in kwargs.items()]
    argv = (c_char_p * len(args))()  # convert list into ctype array of char*
    argv[:] = args                   # make each char* item point to the corresponding string in 'args'
    return argv

###############################################################

def set_backend(name,**kwargs):
    gc.collect()
    argv = _kwargs2argv(kwargs)
    return dll.api_set_backend(c_char_p(name),len(argv),argv)

def set_backend_options(**kwargs):
    gc.collect()
    argv = _kwargs2argv(kwargs)
    return dll.api_set_backend_options(len(argv),argv)

def reset_backend(**kwargs):
    gc.collect()
    argv = _kwargs2argv(kwargs)
    return dll.api_reset_backend(len(argv),argv)


def get_backend_name():     return str(dll.api_get_backend_info().name)
def get_supported_dtypes():   return [_int2dtype[dt] for dt in _dtype2int.values() if dll.api_is_dtype_supported(dt) == True]
def set_default_dtype(dt):  dll.api_set_default_dtype(dtype2int(dt))
def set_default_dtypef(dt): dll.api_set_default_dtypef(dtype2int(dt))
def get_default_dtype():    return int2dtype(dll.api_get_default_dtype())
def get_default_dtypef():   return int2dtype(dll.api_get_default_dtypef())
def get_dtype_size(dt):     return int(dll.api_dtype_size(dtype2int(dt)))

def get_backend_info():
    info = c_backend_info()
    dll.api_get_backend_info(byref(info))
    return backend_info(info)

class backend_info(object):
    def __init__(self,info):
        self.uuid    = int(info.uuid)
        self.name    = str(info.name)
        self.version = str(info.version)
        self.device  = str(info.device)

    def __repr__(self):
        return "%s (v%s) using %s\n" % (self.name,self.version,self.device)

def get_heap_status():
    info = c_heap_status()
    dll.api_get_heap_status(byref(info))
    return heap_status(info)

class heap_status(object):
    def __init__(self,info):
        self.host_total   = long(info.host_total)
        self.host_avail   = long(info.host_avail)
        self.host_used    = long(info.host_used)
        self.device_total = long(info.device_total)
        self.device_avail = long(info.device_avail)
        self.device_used  = long(info.device_used)
        self.device_committed = long(info.device_committed)

    def __repr__(self):
        string = ''
        for name in ['host_total','host_avail','host_used','device_total','device_avail','device_used','device_committed']:
            string += '%s: %s\n' % (name,util.format_bytecount(self.__dict__[name],fmt="2.2cM"))
        return string

def load_extension(dllname,search_dirs=None):
    sync()
    handle = dll.api_load_extension(dllname)
    ext_dll = CDLL(dllname,handle=handle)
    ext_dll = safe_dll(ext_dll,dll.api_get_last_error,dll.api_clear_last_error)
    return ext_dll

def unload_extension(ext_dll):
    sync()
    dll.api_unload_extension(ext_dll._dll._handle)
    del ext_dll._dll

def autotune():
    dll.api_autotune_backend()

def destroy_backend(force=False):
    """
    Destroys the backend, including any device resources associated with the current thread.
    If there are outstanding handles to memory alloations (e.g. an sarray instance still
    holding on to memory used by the backend) then the call will fail; use force=True to override,
    though the program may later crash due to those objects holding invalid pointers.
    """
    gc.collect()
    dll.api_destroy_backend(force)

