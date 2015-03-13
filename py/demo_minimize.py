import numpy
import numpy.random
import smat
import smat.util
import argparse
import scipy.optimize

parser = argparse.ArgumentParser(description="Train a 784-1000-1000-10 neural net on MNIST and print out the error rates.")
parser.add_argument("-d","--device",type=int,default=None,help="The device to use, e.g. CUDA device.")
parser.add_argument("-m","--method",type=str,default="L-BFGS-B",help="The optimization algorithm to use. Valid values are COBYLA and L-BFGS-B.")
args = parser.parse_args()

if args.device is not None:
    smat.set_backend_options(device=args.device)

print "Using device",smat.get_backend_info().device
print "Using method",args.method,"with float64"

# Load some sample bio data. Specifically this is a subset of the 
# RNAcompete protein binding affinities from Ray et al., Nature, 2013.
y = numpy.load('data/rnac/rnac_subset.npz')['y']
n,m = y.shape

def objective_function(x,y,lib):
    # The test objective function below happens to be that corresponding to
    # "Variance Stabilization" (Huber et al., Bioinformatics, 2002).
    # The specific objective is not important.
    # The point is that the parameters can be sent to the GPU, 
    # evaluated, pulled back, and STILL be much faster than CPU.
    
    # Shorthand for some functions that we're getting from lib=smat/numpy
    asarray,arcsinh,sqrt,mean,log,sum = lib.asarray,lib.arcsinh,lib.sqrt,lib.mean,lib.log,lib.sum

    # Push coefficients to GPU and get separate views to 'a' and 'b'
    a,b = asarray(x).reshape((2,-1))

    # Calculate h(y) and h'(y); see Huber et al., equation (6)
    y = a+y*b
    h      = arcsinh(y)
    hprime = b/sqrt(y**2+1)

    # Calculate negative log-likelihood of current variance distribution; see Huber et al., equation (13)
    hmean = mean(h,axis=1).reshape((-1,1))
    term1 = log(sum((h-hmean)**2))
    term2 = sum(log(hprime))
    variance_nll = (.5*n*m)*term1 - term2

    # Pull final objective value back from GPU
    return float(variance_nll)


def run_minimize(y,method,lib):
    print "\nOptimizing with %s..." % lib.__name__

    # Push y to GPU ahead of time, in the case of smat
    y = lib.asarray(y)

    # Set up initial parameter vector x=[a;b]
    a = numpy.zeros((1,m))
    b = numpy.ones((1,m))
    x = numpy.vstack([a,b]).ravel()

    # Set up bounds for vector a (unbounded) and vector b (positive)
    bounds = [(None,None) for i in range(m)] + [(1e-5,None) for i in range(m)]

    # Call scipy to do the optimization
    if   method == "COBYLA":   maxiter = 1000
    elif method == "L-BFGS-B": maxiter = 5
    else: quit("Unsupported \"method\".")
    time = 0
    print "   iter   0: objective = %.1f at start" % (objective_function(x,y,lib))
    for t in range(5):
        smat.util.tic()
        x = scipy.optimize.minimize(objective_function,x,args=(y,lib),bounds=bounds,method=method,options={'maxiter':maxiter},tol=1e-20).x
        time += smat.util.toc()
        print "   iter %3d: objective = %.1f, time elapsed = %.1fs" % ((t+1)*maxiter,objective_function(x,y,lib),time)

run_minimize(y,args.method,smat)
run_minimize(y,args.method,numpy)
