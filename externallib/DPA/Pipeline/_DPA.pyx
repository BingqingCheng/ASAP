# file '_DPA.pyx'. 
# Non-parametric Density Peak clustering: 
# Automatic topography of high-dimensional data sets 
#
# Author: Maria d'Errico <mariaderr@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
cimport numpy as c_np
import copy


def get_centers(N, indices, k_hat, g):
    cdef int i, j, c
    cdef list centers = []
    # Criterion 1 from Heuristic 1
    for i in range(0, N):
        putative_center = True
        for j in indices[i][1:k_hat[i]+1]:
            if g[j]>g[i]:
                putative_center = False
                break
        if putative_center:
            centers.append(i)
    # Criterion 2 from Heuristic 1
    for c in centers:
        for i in range(0,N):
            if g[c]<g[i] and c in indices[i][1:k_hat[i]+1]:
                centers.remove(c)
                break
    return centers

def initial_assignment(g, N, indices, centers):
    cdef long c, i, el, k
    cdef c_np.ndarray[long, ndim=1] ig_sort = np.argsort([-x for x in g])

    # Assign points to clusters
    #--------------------------
    # Assign all the points that are not centers to the same cluster as the nearest point with higher g. 
    # This assignation is performed in order of decreasing g
    cdef c_np.ndarray[long, ndim=1] clu_labels = np.zeros(N,dtype=int)-1
    for c in centers:
        clu_labels[c] = centers.index(c)
    for i in range(0,N):
        el = ig_sort[i]
        k = 0
        while (clu_labels[el]==-1):
            k=k+1
            clu_labels[el] = clu_labels[indices[el][k]] # the point with higher g is already assigned by construction
    return clu_labels


def get_borders( N, k_hat, indices, clu_labels, Nclus, g, densities, err_densities, Z, centers):
    cdef dict border_dict = {}
    cdef dict g_saddle = {}
    cdef int i, k, j, c, cp, m_c, M_c

    # Criterion 1 from Heuristic 2:
    # point i belonging to c is at the border if its closest point j belonging to c′ is 
    #within a distance k_hat[i] 
    for i in range(0,N):
        for k in range(0,k_hat[i]):
            j = indices[i][k+1]
            if clu_labels[j]!=clu_labels[i]:
                if (i, clu_labels[i]) not in border_dict.keys():
                    border_dict[(i, clu_labels[i])] = [-1]*Nclus
                    border_dict[(i, clu_labels[i])][clu_labels[j]] = j
                    break
                elif border_dict[(i, clu_labels[i])][clu_labels[j]]==-1:
                    border_dict[(i, clu_labels[i])][clu_labels[j]] = j
                    break
                else:
                    break

    # Criterion 2 from Heuristic 2:
    # check if i is the closest point to j among those belonging to c.
    for i,c in border_dict.keys():
        for cp in range(Nclus):
            j = border_dict[(i,c)][cp]
            if j!=-1:
                if (j,cp) in border_dict.keys() and border_dict[(j,cp)][c] == i:
                    m_c = min(c,cp)
                    M_c = max(c,cp)
                    if (m_c, M_c) not in g_saddle.keys() or g[i] > g_saddle[(m_c,M_c)][1]:
                        g_saddle[(m_c,M_c)] = (i, g[i])

    # Fill in the border density matrix
    cdef c_np.ndarray[double, ndim=2] Rho_bord = np.zeros((Nclus,Nclus),dtype=float)
    cdef c_np.ndarray[double, ndim=2] Rho_bord_err = np.zeros((Nclus,Nclus),dtype=float)
    for c,cp in g_saddle.keys():
        i = g_saddle[(c,cp)][0]
        Rho_bord[c][cp] = densities[i]
        Rho_bord[cp][c] = densities[i]
        Rho_bord_err[c][cp] = err_densities[i]
        Rho_bord_err[cp][c] = err_densities[i]
    for c in range(0,Nclus):
        Rho_bord[c][c] = -1
        Rho_bord_err[c][c] = 0

    # Merging
    cdef int check = 1
    cdef float rho_bord_max
    cdef float a1, a2, e1, e2
    cdef int cmax, cmin
    cdef dict M = {}
    cdef dict D = {}
    cdef list S
    while(check == 1):
        check=0
        rho_bord_max = 0
        # check if any pairs of clusters has to be merged
        for c,cp in g_saddle.keys():
            a1 = densities[centers[c]]-Rho_bord[c][cp]
            a2 = densities[centers[cp]]-Rho_bord[c][cp]
            e1 = Z*(err_densities[centers[c]]+Rho_bord_err[c][cp])
            e2 = Z*(err_densities[centers[cp]]+Rho_bord_err[c][cp])
            if a1<e1 or a2<e2:
                check = 1
                # Select the pair (imax, jmax) with the highest border density
                # imax will correspond to the peak with highest density
                if Rho_bord[c][cp]> rho_bord_max:
                    rho_bord_max = Rho_bord[c][cp]
                    cmax = c
                    cmin = cp
                    if densities[centers[c]]<densities[centers[cp]]:
                         cmax = cp
                         cmin = c
        if check:
            # Store the clusters labels to merge in assignment
            M[cmin]=cmax
            # Update topography
            Rho_bord[cmax][cmin] = -1
            Rho_bord[cmin][cmax] = -1
            Rho_bord_err[cmax][cmin] = 0
            Rho_bord_err[cmin][cmax] = 0
            del g_saddle[(min(cmax,cmin), max(cmax,cmin))]
            S = list(g_saddle.keys()) 
            for c,cp in S:
                if c==cmin and cp!=cmax:
                    # The cluster cmax inherits the clusters neighboring to cmin if with higher border densities
                    if Rho_bord[cmax][cp] < Rho_bord[cmin][cp]:
                        Rho_bord[cmax][cp] = Rho_bord[cmin][cp]
                        Rho_bord[cp][cmax] = Rho_bord[cmin][cp]
                        Rho_bord_err[cmax][cp] = Rho_bord_err[cmin][cp]
                        Rho_bord_err[cp][cmax] = Rho_bord_err[cmin][cp]
                        g_saddle[(min(cmax,cp), max(cmax,cp))] = g_saddle[(cmin, cp)]
                    else:
                        pass
                    # Delete border information between cmin and cp
                    del g_saddle[(c,cp)]
                    Rho_bord[c][cp]=-1
                    Rho_bord[cp][c]=-1
                    Rho_bord_err[cp][c]=0
                    Rho_bord_err[c][cp]=0
                elif c!=cmax and cp==cmin:
                    if Rho_bord[cmax][c] < Rho_bord[cmin][c]:
                        Rho_bord[cmax][c] = Rho_bord[cmin][c]
                        Rho_bord[c][cmax] = Rho_bord[cmin][c]
                        Rho_bord_err[cmax][c] = Rho_bord_err[cmin][c]
                        Rho_bord_err[c][cmax] = Rho_bord_err[cmin][c]
                        g_saddle[(min(cmax,c), max(cmax,c))] = g_saddle[(c, cmin)]
                    else:
                        pass
                    # Delete border information between cmin and c
                    del g_saddle[(c,cp)]
                    Rho_bord[c][cp]=-1
                    Rho_bord[cp][c]=-1
                    Rho_bord_err[cp][c]=0
                    Rho_bord_err[c][cp]=0
                else:
                    pass

    # Update clustering lables for merged clusters    
    for c in M.keys():
        for i in range(N):
            if clu_labels[i]==c:
                clu_labels[i]=M[c]

    # Rename the labels of the final clusters
    # Clusters labels go from 0 to Nclus_m-1
    Nclus_m = 0
    for i in set(clu_labels):
        D[i] = Nclus_m
        Nclus_m +=1 
    for i in range(N):
        clu_labels[i] = D[clu_labels[i]]

    # Update topography
    cdef c_np.ndarray[double, ndim=1] min_rho_bord = np.zeros(Nclus_m)
    cdef c_np.ndarray[double, ndim=2] Rho_bord_m = np.zeros((Nclus_m,Nclus_m),dtype=float)
    cdef c_np.ndarray[double, ndim=2] Rho_bord_err_m = np.zeros((Nclus_m,Nclus_m),dtype=float)
    for c,cp in g_saddle.keys():
        i = g_saddle[(c,cp)][0]
        c = D[c]
        cp = D[cp]
        if densities[i]>min_rho_bord[c]:
            min_rho_bord[c] = densities[i]
        else:
            pass
        if densities[i]> min_rho_bord[cp]:
           min_rho_bord[cp] = densities[i]
        else:
            pass
        Rho_bord_m[c][cp] = densities[i]
        Rho_bord_m[cp][c] = densities[i]
        Rho_bord_err_m[c][cp] = err_densities[i]
        Rho_bord_err_m[cp][c] = err_densities[i]
    for c in range(0,Nclus_m): 
        Rho_bord_m[c][c] = -1
        Rho_bord_err_m[c][c] = 0

    # Halos
    cdef c_np.ndarray[long, ndim=1] clu_halos = copy.deepcopy(clu_labels)
    clu_halos = find_halos(min_rho_bord, clu_halos, densities)

    return Rho_bord_m, Rho_bord_err_m, clu_labels, clu_halos, Nclus_m

def find_halos(min_rho_bord, clu_halos, densities):
    cdef int i
    for i in range(len(densities)):
        if densities[i]<min_rho_bord[clu_halos[i]] and min_rho_bord[clu_halos[i]]>0:
            clu_halos[i]=-1
    return clu_halos


