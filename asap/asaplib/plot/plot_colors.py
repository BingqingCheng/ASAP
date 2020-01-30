"""
TODO: Module-level description
"""

import os
import numpy as np
from ase.io import read


def set_color_function(fcolor=None, fxyz=None, colorscol=0, n_samples=0, peratom=False):

    if os.path.isfile(fxyz) and not os.path.isfile(fcolor):
        # use the information given in the xyz file
        try:
            frames = read(fxyz, ':')
            print('load xyz file: '+fxyz+' for color schemes')
        except:
            raise ValueError('Cannot load the xyz file')

        if len(frames) == 1:
            print('Only one frame so set the color function to the index of atoms')
            fcolor = 'index'
            plotcolor = np.arange(len(frames[0].get_positions()))

        elif len(frames) != n_samples:
            raise ValueError('Length of the xyz trajectory is not the same as number of samples')

        else:
            plotcolor = []
            plotcolor_atomic = []
            try:
                for index, frame in enumerate(frames):
                    natomsnow = len(frame.get_positions())
                    #print(natomsnow)
                    if fcolor == 'volume' or fcolor == 'Volume':
                        use_color_scheme = frame.get_volume()/natomsnow
                    elif fcolor == None or fcolor == 'none' or fcolor == 'Index' or fcolor == 'index':
                        # we use the index as the color scheme
                        use_color_scheme = index
                        fcolor = 'index'
                    elif fcolor in frame.info:
                        if fcolor == 'Pressure' or fcolor == 'pressure' or fcolor == 'Temperature' or fcolor == 'temperature':
                            use_color_scheme = frame.info[fcolor]
                        else:
                        use_color_scheme = frame.info[fcolor]/natomsnow
                    else:
                        raise ValueError('Cannot find the specified property from the xyz file')
                    plotcolor.append(use_color_scheme)
                    if peratom: plotcolor_atomic=np.append(plotcolor_atomic, use_color_scheme*np.ones(natomsnow))
            except:
                raise ValueError('Cannot load the property vector from the xyz file')

    elif os.path.isfile(fcolor):
        # load the column=colorscol for color functions
        try:
            loadcolor = np.genfromtxt(fcolor, dtype=float)
            print(np.shape(loadcolor))
            if colorscol > 0 or len(np.shape(loadcolor)) > 1:
                plotcolor = loadcolor[:,colorscol]
            else:
                plotcolor = loadcolor
            print('load file: '+fcolor+' for color schemes')
            if (len(plotcolor) != n_samples):
                raise ValueError('Length of the vector of properties is not the same as number of samples')

            if peratom:
                plotcolor_atomic = []
                try:
                    frames = read(fxyz, ':')
                    print('load xyz file: '+fxyz+' so that we know the number of atoms in each frame')
                except:
                    raise ValueError('Cannot load the xyz file')
                for index, frame in enumerate(frames):
                    natomsnow = len(frame.get_positions())
                    plotcolor_atomic = np.append(plotcolor_atomic, plotcolor[index]*np.ones(natomsnow))
        except:
            raise ValueError('Cannot load the '+str(colorscol)+'th column from the file '+fcolor)

    elif fcolor == None or fcolor == 'none' or fcolor == 'Index' or fcolor == 'index':
        # we use the index as the color scheme
        plotcolor = np.arange(n_samples)
        fcolor = 'sample index'

    else:
        raise ValueError('Cannot set the color function')

    colorlabel = 'use '+fcolor+' for coloring the data points'

    if peratom:
        print(np.shape(plotcolor_atomic))
        return plotcolor, np.asarray(plotcolor_atomic), colorlabel
    else:
        return plotcolor, colorlabel


class COLOR_PALETTE:
    def __init__(self, style=1):
        if style == 1:
            self.pal = ["#FF34FF", "#FF4A46","#008941", "#006FA6", "#A30059", "#0000A6", "#63FFAC","#B79762", "#004D43", "#8FB0FF", "#997D87","#5A0007", "#809693", 
    "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", "#BA0900","#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100","#DDEFFF",
    "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F","#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99",
    "#001E09","#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66","#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
    "#456648", "#0086ED", "#886F4C","#34362D", "#B4A8BD", "#00A6AA", "#452C2C","#636375", "#A3C8C9", "#FF913F", "#938A81","#575329", "#00FECF", "#B05B6F",
    "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00","#7900D7", "#A77500","#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700","#549E79",
    "#FFF69F", "#201625", "#72418F","#BC23FF","#99ADC0","#3A2465","#922329","#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C","#7A4900"]
        elif style ==2:
            self.pal = ["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b", "#006FA6", "#A30059", "#af8dc3", "#922329", "#1E6E00"]

        self.n_color = len(self.pal)

    def __getitem__(self, arg):  # color cycler
        assert arg > -1, "???"
        return self.pal[arg%self.n_color]
