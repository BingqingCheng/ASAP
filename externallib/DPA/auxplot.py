import numpy as np
from Pipeline import DPA
import hdbscan
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from examples import example
import scipy as sp
from sklearn import manifold
from scipy import cluster
from matplotlib.collections import LineCollection
from math import sqrt
from sklearn.metrics.cluster import normalized_mutual_info_score
def plot_w_noise(ax,data,labels,noise_id):
    x=data[:,0]
    y=data[:,1]
    xp=x[labels==noise_id]
    yp=y[labels==noise_id]
    ax.scatter(xp,yp,s=15.,alpha=0.9, c='black',linewidths=0.0)
    xp=x[labels!=noise_id]
    yp=y[labels!=noise_id]
    cp=labels[labels!=noise_id]
    ax.scatter(xp,yp,s=15.,alpha=0.9, c=cp,linewidths=0.0,cmap='tab10')
    return ax
def plot_no_noise(ax,data,labels):
    x=data[:,0]
    y=data[:,1]
    xp=x
    yp=y
    cp=labels
    ax.scatter(xp,yp,s=15.,alpha=0.9, c=cp,linewidths=0.0,cmap='tab10')
    return ax
def plot_colored(ax,data,color):
    ax.scatter(data[:,0],data[:,1],s=15.,alpha=0.9, c=color,linewidths=0.0)
    return ax
def plot_contour_interpolated(ax,data,color):
    ax.tricontour(data[:,0],data[:,1],color,levels=10, linewidths=0.5, colors='k')
    ax.tricontourf(data[:,0],data[:,1],color,levels=250,alpha=0.9)
    return ax
def plots_topography(dpa,ax_dendrogram,ax_project):
    Nclus_m=np.max(dpa.labels_)+1
    cmap = plt.get_cmap('tab10', Nclus_m)
    # Convert from border densities to distances
    nd=int((Nclus_m*Nclus_m-Nclus_m)/2)
    Dis = np.empty(nd,dtype=float)
    nl=0
    Fmax=max(dpa.densities)
    Rho_bord= np.zeros((Nclus_m,Nclus_m),dtype=float)
    for row in dpa.topography_:
        Rho_bord[row[0]][row[1]]=row[2]
        Rho_bord[row[1]][row[0]]=row[2]
        Dis[nl]=Fmax-row[2]
        nl=nl+1
    # dendrogram representation
    DD=sp.cluster.hierarchy.single(Dis)
    dn=sp.cluster.hierarchy.dendrogram(DD,color_threshold=0, above_threshold_color='k',ax=ax_dendrogram)
    xlbls = ax_dendrogram.get_xmajorticklabels()
    dorder=[]
    for lbl in xlbls:
        dorder.append(int(lbl._text))
        lbl.set_color(cmap(int(lbl._text)))
        lbl.set_weight('bold')
# 2D projection representation of the topography
    pop=np.zeros((Nclus_m),dtype=int)
    for i in range (len(dpa.labels_)):
        pop[dpa.labels_[i]]=pop[dpa.labels_[i]]+1
    d_dis = np.zeros((Nclus_m,Nclus_m),dtype=float)
    model = manifold.MDS(n_components=2,n_jobs=10,dissimilarity='precomputed')
    for i in range(Nclus_m):
        for j in range(Nclus_m):
            d_dis[i][j]=Fmax-Rho_bord[i][j]
    for i in range(Nclus_m):
        d_dis[i][i]=0.
    out = model.fit_transform(d_dis)
    ax_project.yaxis.set_major_locator(plt.NullLocator())
    ax_project.xaxis.set_major_locator(plt.NullLocator())
    s=[]
    col=[]
    for i in range (Nclus_m):
        s.append(20.*sqrt(pop[i]))
        col.append(i)
    ax_project.scatter(out[:,0],out[:,1],s=s,c=col,cmap=cmap)
    #plt.colorbar(ticks=range(Nclus_m))
    #plt.clim(-0.5, Nclus_m-0.5)
    for i in range (Nclus_m):
        ax_project.annotate(i,(out[i,0],out[i,1]))
    for i in range(Nclus_m):
        for j in range(Nclus_m):
            d_dis[i][j]=Rho_bord[i][j]
    rr=np.amax(d_dis)
    if (rr>0.):
        d_dis = d_dis/rr*100.
    start_idx,end_idx=np.where(out)
    segments=[[out[i,:],out[j,:]]
        for i in range (len(out)) for j in range (len(out))]
    values=np.abs(d_dis)
    lc = LineCollection(segments,zorder=0,norm=plt.Normalize(0,values.max()))
    lc.set_array(d_dis.flatten())
    lc.set_edgecolor(np.full(len(segments),'black'))
    lc.set_facecolor(np.full(len(segments),'black'))
    lc.set_linewidths(0.2*Rho_bord.flatten())
    ax_project.add_collection(lc)
    return ax_dendrogram,ax_project
def get_info_noise(labels,gt):
    pure_nmi=normalized_mutual_info_score(gt,labels,average_method="arithmetic")
    clas_mask_1=labels
    clas_mask_2=gt
    t1=clas_mask_1[(clas_mask_1>-1) & (clas_mask_2>-1)]
    t2=clas_mask_2[(clas_mask_1>-1) & (clas_mask_2>-1)]
    ign_nmi=normalized_mutual_info_score(t1,t2,average_method="arithmetic")
    FNR=len(clas_mask_1[(clas_mask_1>-1)& (clas_mask_2==-1)])/len(clas_mask_1[(clas_mask_2==-1)])
    FPR=len(clas_mask_1[(clas_mask_1==-1)& (clas_mask_2>-1)])/len(clas_mask_1[(clas_mask_2>-1)])
    return ign_nmi,FNR,FPR
def get_info_no_noise(labels,gt):
    pure_nmi=normalized_mutual_info_score(gt,labels,average_method="arithmetic")
    return pure_nmi
