import numpy as np
import scipy.linalg as salg

"""
tools for doing kernal PCA on environmental similarity
e.g.
eva = np.genfromtxt(prefix+".k",skip_header=1)
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

def fixcenter(kernel):
    cernel = kernel.copy()
    cols=np.mean(cernel,axis=0);
    #print "numcol ", cols.shape
    rows=np.mean(cernel,axis=1);
    #print "numrows", rows.shape
    mean=np.mean(cols);
    for i in range(len(rows)): 
        cernel[i,:]-=cols
    for j in range(len(cols)):
        cernel[:,j]-=rows
    cernel += mean
    return cernel

