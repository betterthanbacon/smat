import argparse
import smat
import smat.tests

smat.set_backend_options(device=0)
'''
for dt in (smat.float32,smat.float64):
    maxsize = 20*1024*1024/4
    for logn in range(2,20):
        for logm in range(2,20):
            for logk in range(2,20):
                n = 2**logn
                m = 2**logm
                k = 2**logk
                if n*k > maxsize or k*m > maxsize or m*n > maxsize:
                    continue
                A = smat.ones((n,k),dtype=dt)
                B = smat.ones((k,m),dtype=dt)
                smat.dot(A,B)
                #smat.dot_nt(A,B.T)
                #smat.dot_tn(A.T,B)
                #smat.dot_tt(A.T,B.T)
                print n,m,k

quit()
'''
parser = argparse.ArgumentParser(description="Run the smat unit tests and/or performance tests.")
parser.add_argument("-p","--perf",action="store_true",default=False,help="Run performance tests instead of unit tests.")
parser.add_argument("-d","--device",type=int,default=None,help="The device to use, e.g. CUDA device.")
parser.add_argument("-b","--backend",type=str,default=None,help="The backend to use. Currently only \"cuda\" is supported.")
args = parser.parse_args()

if args.backend is not None:
    smat.set_backend(args.backend)

if args.device is not None:
    smat.set_backend_options(device=args.device)

print smat.get_backend_info()
print smat.get_heap_status()

if args.perf:
    smat.tests.perftest()
else:
    smat.tests.unittest()
