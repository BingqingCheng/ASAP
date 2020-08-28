"""
Color functions
"""

import os
import numpy as np

from asaplib.data import ASAPXYZ


def set_color_function(fcolor='none', asapxyz=None, colorscol=0, n_samples=0, 
              peratom=False, project_atomic=False, use_atomic_species=None, color_from_zero=False, extensive=False):
    """ obtain the essential informations to define the colors of data points
    Parameters
    ----------
    fcolor: str
             the name of the file or the tag in the xyz to define the colors
    asapxyz: ASAPXYZ object, (optional)
    colorscol: int, (optional). 
              if the color file has more than one column, which column to use
    n_samples: int, (optional). 
              The number of data points
    peratom: bool
              return atomic color
    project_atomic: bool
              the samples are atomic descriptors
    use_atomic_species: int
              the atomic number of the selected species
    color_from_zero: bool
              set the min color to zero
    extensive: bool
              normalize the quatity by number of atoms
    """

    plotcolor = []
    plotcolor_atomic = []
    colorscale = [None, None]

    # if there is a file named "fcolor", we load it for the color scheme
    if os.path.isfile(fcolor):
        # load the column=colorscol for color functions
        try:
            loadcolor = np.genfromtxt(fcolor, dtype=float)
        except:
            raise IOError('Error in loading fcolor files for the color scheme')

        # print(np.shape(loadcolor))
        if colorscol > 0 or len(np.shape(loadcolor)) > 1:
            plotcolor = loadcolor[:, colorscol]
        else:
            plotcolor = loadcolor
        print('load file: ' + fcolor + ' for color schemes')

        if peratom or project_atomic:
            if asapxyz is None:
                raise IOError('Need the xyz so that we know the number of atoms in each frame')
            elif asapxyz.get_num_frames() == len(plotcolor):
                for index, natomnow in enumerate(asapxyz.get_natom_list_by_species(use_atomic_species)):
                    plotcolor_atomic = np.append(plotcolor_atomic, plotcolor[index] * np.ones(natomnow))
            elif asapxyz.get_total_natoms() == len(plotcolor):
                plotcolor_atomic = plotcolor
            else:
                raise ValueError('Length of the xyz trajectory is not the same as number of colors in the fcolor file')

    elif n_samples > 0 and (fcolor == None or fcolor == 'none' or fcolor == 'Index' or fcolor == 'index') and peratom == False:
        # we use the index as the color scheme
        plotcolor = np.arange(n_samples)
        fcolor = 'sample index'

    elif asapxyz is None:
        raise IOError('Cannot find the xyz or fcolor files for the color scheme')

    else:
        if fcolor == None or fcolor == 'none' or fcolor == 'Index' or fcolor == 'index':
            # we use the index as the color scheme
            plotcolor = np.arange(asapxyz.get_num_frames())
            fcolor = 'sample index'
            if peratom or project_atomic:
                for index, natomnow in enumerate(asapxyz.get_natom_list_by_species(use_atomic_species)):
                    plotcolor_atomic = np.append(plotcolor_atomic, plotcolor[index] * np.ones(natomnow))
        else:
            try:
                plotcolor = asapxyz.get_property(fcolor, extensive)
            except:
                raise ValueError('Cannot find the specified property from the xyz file for the color scheme')
            if peratom or project_atomic:
                try:
                    plotcolor_atomic = asapxyz.get_atomic_property(fcolor, extensive, [], use_atomic_species)
                    #print(np.shape(plotcolor_atomic))
                except:
                    raise ValueError('Cannot find the specified atomic property from the xyz file for the color scheme')

    if color_from_zero:
        # set the min to zero
        plotcolor -= np.ones(len(plotcolor))*np.nanmin(plotcolor)
        plotcolor_atomic -= np.ones(len(plotcolor_atomic))*np.nanmin(plotcolor)

    colorlabel = str(fcolor)
    if peratom and not project_atomic:
        # print(np.shape(plotcolor_atomic))
        colorscale = [np.nanmin(plotcolor_atomic), np.nanmax(plotcolor_atomic)]
        return plotcolor, np.asarray(plotcolor_atomic), colorlabel, colorscale
    elif project_atomic:
        colorscale = [None, None]
        return np.asarray(plotcolor_atomic), [], colorlabel, colorscale
    else:
        colorscale = [None, None]
        return plotcolor, [], colorlabel, colorscale


class COLOR_PALETTE:
    def __init__(self, style=1):
        if style == 1:
            self.pal = ["#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#0000A6", "#63FFAC", "#B79762",
                        "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
                        "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900",
                        "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF",
                        "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", "#372101",
                        "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99",
                        "#001E09", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1",
                        "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
                        "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375",
                        "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F",
                        "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9",
                        "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79",
                        "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534",
                        "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#7A4900"]
        elif style == 2:
            self.pal = ["#30a2da", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b", "#006FA6", "#A30059", "#af8dc3",
                        "#922329", "#1E6E00"]

        self.n_color = len(self.pal)

    def __getitem__(self, arg):  # color cycler
        assert arg > -1, "???"
        return self.pal[arg % self.n_color]
