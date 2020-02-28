import argparse
import os
import sys

import numpy as np
from ase.io import read, write
from asaplib.io import str2bool
from qml.representations import generate_fchl_acsf

def repr_wrapper(frame, elements, is_periodic = False,
                                        nRs2=24, nRs3=20,
                                        nFourier=1, eta2=0.32, eta3=2.7,
                                        zeta=np.pi, rcut=8.0, acut=8.0,
                                        two_body_decay=1.8, three_body_decay=0.57,
                                        three_body_weight=13.4):
    '''
   Periodic systems not implemented for FCHL19.
   
   '''
    if is_periodic:
         raise NotImplementedError( 'Periodic system not implemented!')
         
    nuclear_charges,coordinates = frame.get_atomic_numbers(),frame.get_positions()
    rep = generate_fchl_acsf(nuclear_charges, coordinates, elements,
                                        nRs2=nRs2, nRs3=nRs3, nFourier=nFourier,
                                        eta2=eta2, eta3=eta3, zeta=zeta, 
                                        rcut=rcut, acut=acut,
                                        two_body_decay=two_body_decay, three_body_decay=three_body_decay,
                                        three_body_weight=three_body_weight,
                                        pad=False, gradients=False)
    rep_out = np.zeros((rep.shape[0],len(elements),rep.shape[1]))

    for i,z in enumerate(nuclear_charges):
            j = np.where(np.equal(z,elements))[0][0]
            rep_out[i,j] = rep[i]
    rep_out = rep_out.reshape(len(rep_out),-1)
    return rep_out
    
def main(fxyz = False, fdict = False, prefix= False , output= False , peratom= False, 
                    nRs2=24, nRs3=20, nFourier=1,
                    eta2=0.32, eta3=2.7, zeta=np.pi, 
                    rcut=8.0, acut=8.0,
                    two_body_decay=1.8, three_body_decay=0.57,
                    three_body_weight=13.4, 
                    periodic = False):
  


    """
    Generate the FCHL19 representation.

    Parameters
    ----------
    fxyz: string giving location of xyz file
    fdictxyz: string giving location of xyz file that is used as a dictionary
    prefix: string giving the filename prefix
    output: [xyz]: append the FCHL19 representation to extended xyz file; [mat] output as a standlone matrix
    """



    dictxyz = fdict
    periodic = bool(periodic)
    peratom = bool(peratom)
    fframes = []
    dictframes = []

    # read frames
    if fxyz != 'none':
        fframes = read(fxyz, ':')
        nfframes = len(fframes)
        print("read xyz file:", fxyz, ", a total of", nfframes, "frames")
    # read frames in the dictionary
    if dictxyz != 'none':
        dictframes = read(dictxyz, ':')
        ndictframes = len(dictframes)
        print("read xyz file used for a dictionary:", dictxyz, ", a total of",
              ndictframes, "frames")

    frames = dictframes + fframes
    nframes = len(frames)
    global_species = []
    for frame in frames:
        global_species.extend(frame.get_atomic_numbers())
        if not periodic:
            frame.set_pbc([False, False, False])
    global_species = np.unique(global_species)
    print("a total of", nframes, "frames, with elements: ", global_species)


    foutput = prefix + "_FCHL19" 
    desc_name = "FCHL19" 

    # prepare for the output
    if os.path.isfile(foutput + ".xyz"): os.rename(foutput + ".xyz", "bck." + foutput + ".xyz")
    if os.path.isfile(foutput + ".desc"): os.rename(foutput + ".desc", "bck." + foutput + ".desc")

    for i, frame in enumerate(frames):
        #print(i,frame)
        #print(frame.get_positions())
        rep = repr_wrapper(frame,global_species,periodic,nRs2=nRs2, nRs3=nRs3, nFourier=nFourier,
                                                eta2=eta2, eta3=eta3, zeta=zeta, 
                                                rcut=rcut, acut=acut,
                                                two_body_decay=two_body_decay, three_body_decay=three_body_decay,
                                                three_body_weight=three_body_weight)
        print(rep.shape,rep.mean(axis=0).shape)                              
        frame.info[desc_name] = rep.mean(axis=0)

        # save
        if output == 'matrix':
            with open(foutput + ".desc", "ab") as f:
                np.savetxt(f, frame.info[desc_name][None])
            if peratom or nframes == 1:
                with open(foutput + ".atomic-desc", "ab") as fatomic:
                    np.savetxt(fatomic, rep)
        elif output == 'xyz':
            # output per-atom info
            if peratom:
                frame.new_array(desc_name, rep)
            # write xyze
            write(foutput + ".xyz", frame, append=True)
        else:
            raise ValueError('Cannot find the output format')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, required=True, help='Location of xyz file')
    parser.add_argument('-fdict', type=str, default='none', help='Location of xyz file '
                                                                 'that is used for a dictionary')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='xyz', help='The format for output files ([xyz], [matrix])')
    parser.add_argument('--peratom', type=str2bool, nargs='?', const=True, default=False,
                        help='Do you want to output per atom representations for multiple frames (True/False)?')
                        
    parser.add_argument('--nRs2', type=int, default=24, help='number radial bins in the two-body term')
    parser.add_argument('--nRs3', type=int, default=20, help='number radial bins in the three-body term')
    parser.add_argument('--nFourier', type=int, default=1, help='Order of Fourier expansion of the angular part (Dont Change this!)')
    
    parser.add_argument('--eta2', type=float, default=0.32, help='Widths of the gaussians in the two-body term')
    parser.add_argument('--eta3', type=float, default=3.0, help='Widths of the gaussians in the three-body term')
    parser.add_argument('--zeta', type=float, default=np.pi, help='Width of the gaussian in the three-body angular part (Dont Change this!)')
    
    parser.add_argument('--rcut', type=float, default=8.0, help='Cutoff radius')
    parser.add_argument('--acut', type=float, default=6.0, help='Cutoff radius')
    
    parser.add_argument('--two_body_decay', type=float, default=1.8, help='exponent of the two-body scaling function')
    parser.add_argument('--three_body_decay', type=float, default=0.57, help='exponent of the three-body scaling function')    
    parser.add_argument('--three_body_weight', type=float, default=13.4, help='Relative weight of the three-body term')    
    
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Is the system periodic (True/False)?')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print( args)
    main(**vars(args))
