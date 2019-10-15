import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects

from .plot_colors import COLOR_PALETTE

"""
extension pack for plots!
"""
def plot_connectivity_map(X, z, connect,
                xlabel=None, ylabel=None, clabel=None, label=None,
                centers=None,
                psize=20,
                out_file=None, title=None, show=True, cmap='gnuplot',
                remove_tick=False,
                rasterized=True,
                fontsize=15,
                vmax=None,
                vmin=None
                ):
    """Plots a 2D map given x,y coordinates and an intensity z for
    every data point, and the connectivity information

    Parameters
    ----------connectivity information. format = [No.ts, No.min1, No,min2]
    X : array-like, shape=[n_samples,2]
        Input points.
    z : array-like, shape=[n_samples]
        Density at every point
    connect : array-like, shape=[3, n_samples]. 
              connectivity information. format for each line = [No.ts, No.min1, No.min2]

    """

    # start the plots
    fig, ax = plt.subplots()

    x, y = X[:, 0], X[:, 1]
    z = np.asarray(z) 
    fontsize = fontsize

    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    dv = vmax - vmin
    
    beta = 8.
    sizelist = psize*np.exp(-beta*(z-vmin)/dv)

    nummin = np.nanmax(connect[[1,2],:])

    nummin = np.nanmax(connect[[1,2],:])
    print("number of minima: ", nummin)

    if label is not None:
        axscatter = ax.scatter(x[:nummin], y[:nummin], c=z[:nummin], cmap=cmap, s=sizelist[:nummin], alpha=1.0, rasterized=rasterized, label=label, vmax=vmax, vmin=vmin)
        ax.scatter(x[nummin:], y[nummin:], c=z[nummin:], cmap=cmap, s=sizelist[nummin:], alpha=0.5, rasterized=rasterized, label=label, vmax=vmax, vmin=vmin)
    else:
        axscatter = ax.scatter(x[:nummin], y[:nummin], c=z[:nummin], cmap=cmap, s=sizelist[:nummin], alpha=1.0, rasterized=rasterized, vmax=vmax, vmin=vmin)
        ax.scatter(x[nummin:], y[nummin:], c=z[nummin:], cmap=cmap, s=sizelist[nummin:], alpha=0.5, rasterized=rasterized, vmax=vmax, vmin=vmin)
    
    cb=fig.colorbar(axscatter)

    #"""
    for ts, min1, min2 in connect[:]:
        min1 -= 1
        min2 -= 1
        ts -= 1
        barriernow = (z[ts]+z[min1]+z[min2]-vmin*3.)/dv/3.
        colornow = plt.get_cmap(cmap)(barriernow)
        #lwnow = min(500*np.exp(-beta*barriernow),2)
        ax.plot([x[min1], x[ts], x[min2]],[y[min1], y[ts], y[min2]],'--',c=colornow,alpha=0.5,lw=0.5)
        #ax.plot([x[min1], x[min2]],[y[min1], y[min2]],'--',c=colornow,alpha=0.5, lw=0.5)
    #"""
    
    if remove_tick:
        ax.tick_params(labelbottom='off', labelleft='off')
    
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=fontsize)
    if clabel is not None:
        cb.set_label(label=clabel,labelpad=10)
    if title is not None:
        plt.title(title,fontsize=fontsize)
    if label is not None:
        plt.legend(loc='best')
        
    if centers is not None:
        ax.scatter(centers[:, 0],centers[:, 1], c='lightgreen', marker='*', s=200, edgecolor='black', linewidths=0.5)
    
    if out_file is not None:
        fig.savefig(out_file)
    if show:
        plt.show()

    return fig, ax

