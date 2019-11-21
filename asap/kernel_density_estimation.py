#!/usr/bin/python3

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from asaplib.kde import KDE
from asaplib.plot import plot_styles
from asaplib.io import str2bool


def main(fkmat, ftags, prefix, dimension, pc1, pc2, adtext):

    # if it has been computed before we can simply load it
    try:
        proj = np.genfromtxt(fkmat, dtype=float)[:,0:dimension]
    except:
        raise ValueError('Cannot load the coordinates')
    print("loaded", fkmat)
    # load tags if any
    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    density_model = KDE()        
    # fit density model to data
    density_model.fit(proj)        
    sigma_kij = density_model.bandwidth
    rho = density_model.evaluate_density(proj)
    # save the density
    np.savetxt(prefix+"-kde.dat", np.transpose([np.arange(len(rho)), rho]), header='index kernel_density_estimation', fmt='%d %4.8f')

    # color scheme
    plotcolor = rho
    colorlabel = 'local density of each data point (bandwith $\sigma(k_{ij})$ ='+"{:4.0e}".format(sigma_kij)+' )'
    [plotcolormin, plotcolormax] = [np.min(plotcolor),np.max(plotcolor)]

    # make plot
    plot_styles.set_nice_font()
    # density plot
    fig, ax = plot_styles.plot_density_map(proj[:,[pc1,pc2]], plotcolor,
                xlabel='Princple Axis '+str(pc1), ylabel='Princple Axis '+str(pc2), 
                clabel=colorlabel, label=None,
                centers=None,
                psize=20,
                out_file='KDE_4_'+prefix+'.png', 
                title='KDE for: '+prefix, 
                show=False, cmap='gnuplot',
                remove_tick=False,
                use_perc=False,
                rasterized=True,
                fontsize=15,
                vmax=plotcolormax,
                vmin=plotcolormin)

    fig.set_size_inches(18.5, 10.5)
    if ftags != 'none':
        texts = []
        for i in range(ndict):
            ax.scatter(proj[i,pc1],proj[i, pc2],marker='^',c='black')
            texts.append(ax.text(proj[i, pc1], proj[i, pc2], tags[i],
                         ha='center', va='center', fontsize=15, color='red'))
            #ax.annotate(tags[i], (proj[i,pc1], proj[i,pc2]))
        if (adtext):
            from adjustText import adjust_text
            adjust_text(texts,on_basemap=True,# only_move={'points':'', 'text':'x'},
                    expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                   force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                   ax=ax, precision=0.01,
                  arrowprops=dict(arrowstyle="-", color='black', lw=1,alpha=0.8))

    plt.show()
    fig.savefig('kde_4_'+prefix+'.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kmat', type=str, required=True, help='Location of low dimensional coordinate file.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--d', type=int, default=10, help='number of the first X dimensions to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False, help='Do you want to adjust the texts (True/False)?')
    args = parser.parse_args()

    main(args.kmat, args.tags, args.prefix, args.d, args.pc1, args.pc2, args.adjusttext)


