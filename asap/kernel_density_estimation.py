#!/usr/bin/python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from asaplib.kde import KDE
from asaplib.plot import plot_styles
from asaplib.io import str2bool
from ase.io import read


def main(fmat, fxyz, ftags, prefix, dimension, pc1, pc2, adtext):

    # try to read the xyz file
    if fxyz != 'none':
        try:
            frames = read(fxyz,':')
            nframes = len(frames)
            print('load xyz file: ',fxyz, ', a total of ', str(nframes), 'frames')
        except: 
            raise ValueError('Cannot load the xyz file')

        desc = []; ndesc = 0
        # load from xyze file
        if nframes > 1:
            for i, frame in enumerate(frames):
                if fmat in frame.info:
                     try:
                         desc.append(frame.info[fmat])
                         if ( ndesc > 0 and len(frame.info[fmat]) != ndesc): raise ValueError('mismatch of number of descriptors between frames')
                         ndesc = len(frame.info[fmat])
                     except:
                         raise ValueError('Cannot combine the descriptor matrix from the xyz file')
        else:
            # only one frame
            try: 
                desc = frames[0].get_array(fmat)
            except: ValueError('Cannot read the descriptor matrix from single frame')
    # we can also load the descriptor matrix from a standalone file
    if os.path.isfile(fmat):
        try:
            desc = np.genfromtxt(fmat, dtype=float)
            print("loaded the descriptor matrix from file: ", fmat)
        except:
            raise ValueError('Cannot load the descriptor matrix from file')
    if len(desc)==0: raise ValueError('Please supply descriptor in a xyz file or a standlone descriptor matrix')
    print("loaded", fmat, " with shape", np.shape(desc))
    # load tags if any
    if ftags != 'none':
        tags = np.loadtxt(ftags, dtype="str")
        ndict = len(tags)

    
    proj = np.asmatrix(desc)[:,0:dimension]
    density_model = KDE()        
    # fit density model to data
    try:
        density_model.fit(proj)
    except:
        raise RuntimeError('KDE did not work. Try smaller d.')  

    sigma_kij = density_model.bandwidth
    rho = density_model.evaluate_density(proj)
    # save the density
    np.savetxt(prefix+"-kde.dat", np.transpose([np.arange(len(rho)), rho]), header='index log_of_kernel_density_estimation', fmt='%d %4.8f')

    # color scheme
    plotcolor = rho
    colorlabel = 'Log of densities for every point (bandwith $\sigma(k_{ij})$ ='+"{:4.0e}".format(sigma_kij)+' )'
    [plotcolormin, plotcolormax] = [np.min(plotcolor),np.max(plotcolor)]

    # make plot
    plot_styles.set_nice_font()
    # density plot
    fig, ax = plot_styles.plot_density_map(np.asarray(proj[:,[pc1,pc2]]), plotcolor,
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
    parser.add_argument('-fmat', type=str, required=True, help='Location of low dimensional coordinate file or name of the tags in ase xyz file.')
    parser.add_argument('-fxyz', type=str, default='none', help='Location of xyz file for reading the properties.')
    parser.add_argument('-tags', type=str, default='none', help='Location of tags for the first M samples')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--d', type=int, default=10, help='number of the first X dimensions to keep')
    parser.add_argument('--pc1', type=int, default=0, help='Plot the projection along which principle axes')
    parser.add_argument('--pc2', type=int, default=1, help='Plot the projection along which principle axes')
    parser.add_argument('--adjusttext', type=str2bool, nargs='?', const=True, default=False, help='Do you want to adjust the texts (True/False)?')
    args = parser.parse_args()

    main(args.fmat, args.fxyz, args.tags, args.prefix, args.d, args.pc1, args.pc2, args.adjusttext)


