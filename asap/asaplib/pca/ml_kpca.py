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

def ooskpca(insqrk,inrectk,ndim=2):
    ### CORRECTLY CENTERS THE TEST POINTS 
    ### CENTERS THE OOS WITH LM MEANS
    print("Centering the data ")
    sqrk = insqrk.copy()
    rectk = inrectk.copy()
    k = skenter(sqrk)
    m = len(rectk)
    n = len(sqrk)
    recc = rectk - np.dot(np.ones((m,n)),sqrk)*1./n - np.dot(rectk,np.ones((n,n)))*1./n + 1./n**2 * np.dot(np.ones((m,n)),sqrk).dot(np.ones((n,n)))

    print("  And now we build a projection ")
    evalo,evec = salg.eigh(k ,eigvals=(len(k)-ndim,len(k)-1) )
    evalo=np.flipud(evalo); evec=np.fliplr(evec)
    pvec = evec.copy()
    
    for i in xrange(ndim):
        pvec[:,i] *= 1./np.sqrt(evalo[i])
    print("Done, super quick. ")
    
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

