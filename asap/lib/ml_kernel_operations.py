import numpy as np
import scipy.linalg as salg
import copy

def normalizekernel(kernel):
    # first normalize the kernel matrix
    nkernel = copy.deepcopy(kernel)
    size = len(kernel)
    for i in range(size):
        nkernel[i,:] /= np.sqrt(kernel[i,i])
        nkernel[:,i] /= np.sqrt(kernel[i,i])
        nkernel[i,i] = 1.0
    return nkernel.clip(max=1)

def kerneltodis(kernel):
    # there can be many transformations between the k-matrix and the distance matrix
    # Here we use d_ij = sqrt(2-2*k_ij) 
    # (k_ij is a normalized symetric kernel)
    nk = normalizekernel(kernel)
    size = len(kernel)
    dis = np.zeros((size,size),dtype=np.float64)
    for i in range(size):
        for j in range(i-1):
            dis[i,j] = dis[j,i] = np.sqrt(2.-2.*nk[i,j])
    
    return dis.clip(min=0)


def kerneltodis_linear(kernel):
    # there can be many transformations between the k-matrix and the distance matrix
    # Here we use d_ij = 1-k_ij
    # (k_ij is a normalized symetric kernel)
    nk = normalizekernel(kernel)
    dis = 1.-nk
    return dis.clip(min=0)

def kerneltorho(kernel, delta):
    # we compute the "density" of the data from kernel matrix
    # delta is the charecteristic spread in similarity

    rho = np.zeros(len(kernel))
    allrhofromdis = np.exp((np.asmatrix(kernel)-1.0)*delta)

    for i in range(len(allrhofromdis)):
        rho[i] = np.sum(allrhofromdis[i])

    return rho

def distorho(dis, delta):
    # we compute the "density" of the data from distance matrix
    # the distance matrix can be computed such as
    # dis = kerneltodis(kernel)
    # delta is the charecteristic distance
    rho = np.zeros(len(dis))
    allrhofromdis = np.exp(dis*(-1./delta))

    for i in range(len(allrhofromdis)):
        rho[i] += np.sum(allrhofromdis[i])

    return rho


def estimate_rho_laio(dcut,dist):
    ### Using a Gaussian Kernel, like in Laio's
    ### Clustering by fast search and find of density peaks
    rho = np.zeros(len(dist))
    for i in range(len(dist)):
        for j in range(len(dist)):
            if dist[i,j]<dcut : rho[i]+= np.exp(-(dist[i,j]/dcut)*(dist[i,j]/dcut))
    return rho

def estimate_delta_laio(rho,dist):
    ### Clustering by fast search and find of density peaks
    # https://science.sciencemag.org/content/sci/344/6191/1492.full.pdf
    delta = (rho*0.0).copy()
    nneigh = np.ones(len(delta),dtype='int')
    for i in range(len(rho)):
        js = np.where(rho>rho[i])[0]
        if len(js)==0:
            delta[i] = np.max(dist[i,:])
            nneigh[i] = i
        else:
            delta[i] = np.min(dist[i,js])
            nneigh[i] = js[np.argmin(dist[i,js])]
    return delta,nneigh

### Just to test the centering 
#from sklearn.preprocessing import KernelCenterer

#def skenter(kernel):
#    return KernelCenterer().fit_transform(kernel)

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
