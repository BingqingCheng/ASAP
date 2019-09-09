#!/usr/bin/python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from asaplib.pca import kpca
from asaplib.plot import plot_styles
from asaplib.io import str2bool
from ase.io import read

def main(fkmat, fxyz, ftags, fcolor, prefix, kpca_d, pc1, pc2, adtext):

    # if it has been computed before we can simply load it
    try:
        kNN = np.genfromtxt(fkmat, dtype=float)
    except:
        raise ValueError('Cannot load the kernel matrix')

    print("loaded",fkmat)
    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    # main thing
    proj = kpca(kNN,kpca_d)

    # save
    np.savetxt(prefix+"-kpca-d"+str(kpca_d)+".coord", proj, fmt='%4.8f', header='low D coordinates of samples')

    # color scheme
    if fcolor != 'none':
        try:
            plotcolor = np.genfromtxt(fcolor, dtype=float)
        except:
            try: 
                frames = read(fxyz,':')
            except: 
                raise ValueError('Cannot load the xyz file')
            plotcolor = []
            try:
                for frame in frames:
                    if(fcolor == 'volume'):
                        plotcolor.append(frame.get_volume()/len(frame.get_positions()))
                    else:
                        plotcolor.append(frame.info[fcolor]/len(frame.get_positions()))
            except: 
                raise ValueError('Cannot load the property vector')
        if (len(plotcolor) != len(kNN)): 
            raise ValueError('Length of the vector of properties is not the same as number of samples')
        colorlabel = 'use '+fcolor+' for coloring the data points'
    else: # we use the index as the color scheme
        plotcolor = np.arange(len(proj))
        colorlabel = 'index of each data point'

    # make plot
    plot_styles.set_nice_font()
    #fig, ax = plt.subplots()

    fig, ax = plot_styles.plot_density_map(proj[:,[pc1,pc2]], plotcolor,
                xlabel='Princple Axis '+str(pc1), ylabel='Princple Axis '+str(pc2), 
                clabel=colorlabel, label=None,
                centers=None,
                psize=30,
                out_file='KPCA_4_'+prefix+'.png', 
                title='KPCA for: '+prefix, 
                show=False, cmap='gnuplot',
                remove_tick=False,
                use_perc=True,
                rasterized = True,
                fontsize = 15,
                vmax = None,
                vmin = None)

    fig.set_size_inches(18.5, 10.5)

    if ftags != 'none':
        texts = []
        for i in range(ndict):
            ax.scatter(proj[i, pc1],proj[i, pc2], marker='^', c='black')
            texts.append(ax.text(proj[i, pc1],proj[i, pc2], tags[i],
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
    fig.savefig('KPCA_4_'+prefix+'-c-'+fcolor+'.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-kmat', type=str, required=True, help='Location of kernel matrix file. You can use gen_kmat.py to compute it.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('-colors', type=str, default='none', help='Properties for all samples (N floats) used to color the scatter plot')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--d', type=int, default=10, help='number of the principle components to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False, help='Do you want to adjust the texts (True/False)?')

    args = parser.parse_args()

    main(args.kmat, args.fxyz, args.tags, args.colors, args.prefix, args.d, args.pc1, args.pc2, args.adjusttext)


