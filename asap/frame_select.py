#!/usr/bin/python3

import argparse
import os
from ase.io import read, write
from asaplib.compressor import fps
import numpy as np


def main(fxyz, fy, prefix, nkeep, algorithm, fmat):

    """

    Select frames from the supplied xyz file using one of the algorithm:
    ----------
    random: random selection
    fps: farthest point sampling selection. Need to supply a kermal matrix or descriptor matrix using -fmat
    sortmin/sortmax: select the frames with the largest/smallest value. Need to supply the vector of properties using -fy
    """

    # read frames
    frames = read(fxyz, ':')
    nframes = len(frames)
    print("read xyz file:", fxyz, ", a total of", nframes, "frames")

    if (nkeep == 0): nkeep = nframes

    if fy != 'none':
        y_all = []
        try:
            y_all = np.genfromtxt(fy, dtype=float)
        except:
            try:
                for frame in frames:
                    if(fy == 'volume' or fy == 'Volume'):
                        y_all.append(frame.get_volume()/len(frame.get_positions()))
                    elif(fy == 'size' or fy == 'Size'):
                        y_all.append(len(frame.get_positions()))
                    else:
                        y_all.append(frame.info[fy]/len(frame.get_positions()))
            except: raise ValueError('Cannot load the property vector')
        if len(y_all) != nframes:
            raise ValueError('Length of the vector of properties is not the same as number of samples')

    if algorithm == 'random' or algorithm == 'RANDOM':
        idx = np.asarray(range(nframes))
        sbs = np.random.choice(idx, nkeep, replace =False)

    elif algorithm == 'sortmax' or algorithm == 'sortmin':
        if fy == 'none': raise ValueError('must suply the vector of properties for sorting')
        
        idx = np.asarray(range(nframes))
        if algorithm == 'sortmax': 
            sbs = [x for _,x in sorted(zip(y_all,idx))][:nkeep]
        elif algorithm == 'sortmin': 
            sbs = [x for _,x in sorted(zip(y_all,idx))][nkeep:]

    elif algorithm == 'fps' or algorithm == 'FPS':
        desc = []; ndesc = 0
        for i, frame in enumerate(frames):
            if fmat in frame.info:
                 try:
                     desc.append(frame.info[fmat])
                     if ( ndesc > 0 and len(frame.info[fmat]) != ndesc): raise ValueError('mismatch of number of descriptors between frames')
                     ndesc = len(frame.info[fmat])
                 except:
                     raise ValueError('Cannot combine the descriptor matrix from the xyz file')
        #if (np.shape(desc)[1] != nframes):
            #desc = np.asmatrix(desc)
            #print(np.shape(desc))
            #desc.reshape((ndesc, nframes))

        if os.path.isfile(fmat):
            try:
                desc = np.genfromtxt(fmat, dtype=float)
            except: raise ValueError('Cannot load the kernel matrix')
        print("shape of the descriptor matrix: ", np.shape(desc), "number of descriptors: ", np.shape(desc[0]))
        sbs, dmax_remain = fps(desc, nkeep , 0)
        np.savetxt(prefix+"-"+algorithm+"-n-"+str(nkeep)+'.error', dmax_remain, fmt='%4.8f', header='the maximum remaining distance in FPS')

    # save
    selection = np.zeros(nframes, dtype=int)
    for i in sbs:
        write(prefix+"-"+algorithm+"-n-"+str(nkeep)+'.xyz',frames[i], append=True)
        selection[i] = 1
    np.savetxt(prefix+"-"+algorithm+"-n-"+str(nkeep)+'.index', selection, fmt='%d')
    #np.savetxt(prefix+"-"+algorithm+"-n-"+str(nkeep)+'.index', sbs, fmt='%d')
    if fy != 'none':
        np.savetxt(prefix+"-"+algorithm+"-n-"+str(nkeep)+'-'+fy, np.asarray(y_all)[sbs], fmt='%4.8f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-y', type=str, default='none', help='Location of the list of properties (N floats) or name of the tags in ase xyz file')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--n', type=int, default=0, help='number of the representative samples to select')
    parser.add_argument('--algo', type=str, default='random', help='the algotithm for selecting frames ([random], [fps], [sortmax], [sortmin])')
    parser.add_argument('-fmat', type=str, required=False, help='Location of descriptor or kernel matrix file. Needed if you select [fps]. You can use gen_kmat.py to compute it.')
    args = parser.parse_args()

    main(args.fxyz, args.y, args.prefix, args.n, args.algo, args.fmat)
