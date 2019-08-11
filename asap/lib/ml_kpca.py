import numpy as np
import scipy.linalg as salg

from .ml_kernel_operations import *

"""
tools for doing kernal PCA on environmental similarity
e.g.
eva = np.genfromtxt(prefix+".k",skip_header=1)
dis = kerneltodis(eva)
np.savetxt(prefix+(".dmat"), dis, fmt='%.18e')
proj = kpca(eva,6)
"""

def kpca(kernel, ndim=2):
    
    print(" - Centering the data ")
    k = fixcenter(kernel)

    print("  And now we build a projection ")
    eval, evec = salg.eigh(k ,eigvals=(len(k)-ndim,len(k)-1) )
    eval=np.flipud(eval); evec=np.fliplr(evec)

    pvec = evec.copy()
    for i in range(ndim):
        pvec[:,i] *= 1./np.sqrt(eval[i])
    print("Done, super quick. ")
    return np.dot(k, pvec)

def ooskpca(sqrk,rectk,ndim=2):
    print("Centering the data ")
    k = fixcenter(sqrk)
    print("  And now we build a projection ")
    eval, evec = salg.eigh(k ,eigvals=(len(k)-ndim,len(k)-1) )
    eval=np.flipud(eval); evec=np.fliplr(evec)
    pvec = evec.copy()
    for i in range(ndim):
        pvec[:,i] *= 1./np.sqrt(eval[i])
    print("Done, super quick. ")
    recc = fixcenter(rectk)
    return np.dot(recc, pvec)

