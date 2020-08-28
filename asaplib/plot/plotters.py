"""
Wrappers to do plots
"""

import numpy as np
from matplotlib import pyplot as plt

from .plot_styles import *

class Plotters:
    """
    Object handing the making of plots.
    Notice that we make all the plots on the same graph.
    A good way of using this is to
    1. make a scatter plot
    2. add clustering results (if any)
    3. add tags to annotate important points

    """

    def __init__(self, fig_spec_dict={}):
        """
        Object handing the making of plots
        Parameters
        ----------
        fig_spec_dict: dictionaries that specify plot options
        e.g.
        fig_spec_dict = {
        'outfile': None,
        'show': True,
        'title': None,
        'xlabel': 'Principal Axis 1',
        'ylabel': 'Principal Axis 2',
        'xaxis': True,  'yaxis': True,
        'remove_tick': False,
        'rasterized': True,
        'fontsize': 16,
        'components':{ 
            "first_p": {"type": 'scatter', ...},
            "second_p": {"type": 'annotate', ...},
            "third_p": {"type": 'cluster', ...}
             }
        }

        Notice that we make all the plots on the same graph.
        A good way of using this is to 
        1. make a scatter plot
        2. add clustering results (if any)
        3. add tags to annotate important points
        """

        # required
        try:
            self.p_spec_dict = fig_spec_dict['components']
        except:
            raise ValueError("Didn't specify any options to do plots. Use `components` key to do so.")

        self.fig_spec = {
        'size': [16, 8],
        'outfile': None,
        'show': True,
        'title': None,
        'xlabel': 'Principal Axis 1',
        'ylabel': 'Principal Axis 2',
        'xaxis': True,  'yaxis': True,
        'remove_tick': False,
        'rasterized': True,
        'fontsize': 16,
        'cmap': 'gnuplot'
        }

        # fill in the values
        for k, v in fig_spec_dict.items():
            if k in self.fig_spec.keys():
                self.fig_spec[k] = v

        # pass down some key information, so all the components for the plot are consistent
        for element in self.p_spec_dict.keys():
            self.p_spec_dict[element]['cmap'] = self.fig_spec['cmap']
            self.p_spec_dict[element]['rasterized'] = self.fig_spec['rasterized']

        # list of plotting objects
        self.engines = {}
        self.acronym = ""

        # intialize plot
        set_nice_font()
        self.fig, self.ax = plt.subplots()
        # size
        self.fig.set_size_inches(self.fig_spec['size'][0], self.fig_spec['size'][1])
        # titles
        if self.fig_spec['xlabel'] is not None:
            self.ax.set_xlabel(self.fig_spec['xlabel'], fontsize=self.fig_spec['fontsize'], labelpad=-3)
        if self.fig_spec['ylabel'] is not None:
            self.ax.set_ylabel(self.fig_spec['ylabel'], fontsize=self.fig_spec['fontsize'], labelpad=-3)
        if self.fig_spec['title'] is not None:
            self.ax.set_title(self.fig_spec['title'], fontsize=self.fig_spec['fontsize'])

        self.bind()

    def add(self, p_spec, tag):
        """
        adding the specifications of a new kernel function
        Parameters
        ----------
        p_spec: dictionary
                specify which atomic descriptor to use 
        """
        self.p_spec_dict[tag] = p_spec

    def pack(self):
        return json.dumps(self.p_spec_dict, sort_keys=True, cls=NpEncoder)

    def get_acronym(self):
        if self.acronym == "":
            for element in self.p_spec_dict.keys():
                self.acronym += self.engines[element].get_acronym()
        return self.acronym

    def bind(self):
        """
        binds the objects that actually make the plots
        these objects need to have .create() method to compute 
        """
        # clear up the objects
        self.engines = {}
        for element in self.p_spec_dict.keys():
            self.engines[element] = self._call(self.p_spec_dict[element])
            self.p_spec_dict[element]['acronym'] = self.engines[element].get_acronym()

    def _call(self, p_spec):
        """
        call the specific kernel objects
        """
        if "type" not in p_spec.keys():
            raise ValueError("Did not specify the type of the kernel function.")
        if p_spec["type"] == "scatter":
            return Plot_Function_Scatter(p_spec)
        if p_spec["type"] == "annotate":
            return Plot_Function_Annotate(p_spec)
        if p_spec["type"] == "cluster":
            return Plot_Function_Cluster(p_spec)
        else:
            raise NotImplementedError 

    def plot(self, X, colors=[], labels=[], tags=[]):
        """Plots a 2D density map given 2D coordinates X
        and properties 
        and tags 
        for every data point.

        Parameters
        ----------
        X : array-like, shape=[n_samples,2]
            Input points.
        colors : array-like, float or integer, shape=[n_samples]
            properties used to color each point
        labels : array-like, float or integer, shape=[n_samples]
            additional properties for each point
        tags : array-like, str, shape=[n_samples]
            tags used to annotate selected points
        """

        for element in self.p_spec_dict.keys():
            self.fig, self.ax = self.engines[element].create(self.fig, self.ax, X, colors, labels, tags)

        # touch ups
        if self.fig_spec['remove_tick']:
            self.ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                               labelbottom=False, right=False, left=False, labelleft=False)
        if self.fig_spec['xaxis'] is not True:
            self.ax.set_xticklabels([])
        if self.fig_spec['yaxis'] is not True:
            self.ax.set_yticklabels([])

        # save and show
        if self.fig_spec['outfile'] is not None:
            self.fig.savefig(self.fig_spec['outfile'])
        if self.fig_spec['show']:
            plt.show()

class Plot_Function_Base:
    def __init__(self, p_spec):
        self.acronym = ""
        pass
    def get_acronym(self):
        # we use an acronym for each plot, so it's easy to find it and refer to it
        return self.acronym
    def create(self, fig, ax, X, colors, labels, tags):
        pass

class Plot_Function_Annotate(Plot_Function_Base):
    def __init__(self, p_spec):
        self.acronym = "annotate"

        self.p_spec = {
        'adtext': True,
        'marker': '^',
        'markersize': 10,
        'markercolor': 'black',
        'textsize': 12,
        'textcolor': 'red' # we can add more options for the adtext part
        }

        # fill in the values
        for k, v in p_spec.items():
            if k in self.p_spec.keys():
                self.p_spec[k] = v

        print("Using annotation plot ...")

    def create(self, fig, ax, X, z=[], labels=[], tags=[]):
        """
        annotate samples using tags.
        Parameters
        ----------
        X : array-like, shape=[n_samples,2]
        Input points.
        tags : array-like, str, shape=[n_samples]
        tags for each point.
        labels and z are not used for this plot style
        """
        texts = []
        for i in range(len(tags)):
            if tags[i] != 'None' and tags[i] != 'none' and tags[i] != '':
                ax.scatter(X[i, 0], X[i, 1], 
                           marker=self.p_spec['marker'], 
                           c=self.p_spec['markercolor'])
                texts.append(ax.text(X[i, 0], X[i, 1], tags[i],
                                     ha='center', va='center', 
                                     fontsize=self.p_spec['textsize'],
                                     color=self.p_spec['textcolor']))
        if self.p_spec['adtext']:
            """ adjust the position of the annotated text, so they don't overlap """
            from adjustText import adjust_text
            adjust_text(texts, on_basemap=True,  # only_move={'points':'', 'text':'x'},
                        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                        force_text=(0.03, 0.5), force_points=(0.01, 0.25),
                        ax=ax, precision=0.01,
                        arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.8))

        return fig, ax

class Plot_Function_Scatter(Plot_Function_Base):
    def __init__(self, p_spec):

        self.acronym = "scatter"

        self.p_spec = {
        'psize': None,
        'rasterized': True,
        'fontsize': 12,
        'cmap': 'gnuplot',
        'alpha': 1.0, # color transparency
        'clabel': None, # label of the colorbar
        'cbar_format': None, #'%1.1f',
        'use_perc': False, # mark the top/bottom ourliers
        'outlier_top_fraction': 0.05, # the fraction of the top ourliers
        'outlier_top_color': 'yellow', # color used to make the top ourliers
        'outlier_bottom_fraction': 0.05, # the fraction of the bottom ourliers
        'outlier_bottom_color': 'black', # color used to make the bottom ourliers
        'vmax': None,
        'vmin': None
        }

        # fill in the values
        for k, v in p_spec.items():
            if k in self.p_spec.keys():
                self.p_spec[k] = v

        print("Using scatter plot ...")

        self.cb = None

    def create(self, fig, ax, X, z, labels=[], tags=[]):
        """
        Plots a 2D scatter map given x,y coordinates and a color for
        every data point

        Parameters
        ----------
        X : array-like, shape=[n_samples,2]
        Input points.
        z : array-like, shape=[n_samples]
        color for each point.
        labels and tags are not used for this plot style
        """

        x, y = X[:, 0], X[:, 1]
        z = np.asarray(z)

        # automatically adjust the marker size according the number of samples
        if self.p_spec['psize'] == None:
            psize = 200*200/len(X)
        else:
            psize = self.p_spec['psize']

        if self.p_spec['use_perc']:
            n_sample = len(x)
            argz = np.argsort(z)
            n_bot_outliers = int(n_sample*self.p_spec['outlier_bottom_fraction'])
            n_top_outliers = int(n_sample*self.p_spec['outlier_top_fraction'])
            bot_outliers = argz[:n_bot_outliers]
            top_outliers = argz[-n_top_outliers:]
            typical = argz[n_bot_outliers:-top_outliers]
            # plot typical
            axscatter = ax.scatter(x[typical], y[typical], c=z[typical], 
                               cmap=self.p_spec['cmap'], 
                               s=psize, 
                               alpha=self.p_spec['alpha'],
                               rasterized=self.p_spec['rasterized'])
            # plot bot outliers
            ax.scatter(x[bot_outliers], y[bot_outliers], c=self.p_spec['outlier_bottom_color'], 
                               cmap=self.p_spec['cmap'], 
                               s=psize, 
                               alpha=self.p_spec['alpha'],
                               rasterized=self.p_spec['rasterized'])
            # plot top outliers
            ax.scatter(x[top_outliers], y[top_outliers], c=self.p_spec['outlier_top_color'], 
                               cmap=self.p_spec['cmap'], 
                               s=psize, 
                               alpha=self.p_spec['alpha'],
                               rasterized=self.p_spec['rasterized'])
        else:
            # check if the labels are discrete
            discrete_label = True
            for iz in z: 
                if not np.equal(np.mod(iz, 1), 0): discrete_label = False
            if discrete_label:
                print("Use discrete colormap ......")
                cmap = plt.cm.get_cmap(self.p_spec['cmap'], int(np.max(z)-np.min(z)) + 1)
                vmin = np.min(z) - 0.5
                vmax = np.max(z) + 0.5
            else:
                cmap = self.p_spec['cmap']
                vmin = self.p_spec['vmin']
                vmax = self.p_spec['vmax']

            axscatter = ax.scatter(x, y, c=z, 
                               cmap=cmap, 
                               s=psize, 
                               alpha=self.p_spec['alpha'],
                               rasterized=self.p_spec['rasterized'],
                               vmax=vmax,
                               vmin=vmin)

        if self.p_spec['cbar_format'] is None:
            color_spread = np.nanmax(z) - np.nanmin(z)
            if color_spread > 2:
                self.p_spec['cbar_format'] = '%d'
            elif color_spread > 0.1:
                self.p_spec['cbar_format'] = '%1.1f'
            else:
                self.p_spec['cbar_format'] = '%1.1e'
        if self.p_spec['clabel'] is not None and self.cb is None: 
            self.cb = fig.colorbar(axscatter, format=self.p_spec['cbar_format'])
            self.cb.ax.locator_params(nbins=5)
        if self.p_spec['clabel'] is not None:
            self.cb.set_label(label=self.p_spec['clabel'], labelpad=1)

        return fig, ax


class Plot_Function_Cluster(Plot_Function_Base):
    """
    Plots a 2D clustering plot given x,y coordinates and a label z for
    every data point.
    Basically we draw a circle centered arround the mean position of the samples 
    belonging to each cluster,
    with a size propotional to log(cluster_size)
    """
    def __init__(self, p_spec):

        self.acronym = "cluster"

        self.p_spec = {
        'w_label': False,
        'circle_size': 20, 
        'facecolor': 'none',
        'edgecolor': 'gray',
        'fontsize': 16,
        'cmap': 'gnuplot',
        'alpha': 1.0 # color transparency
        }

        # fill in the values
        for k, v in p_spec.items():
            if k in self.p_spec.keys():
                self.p_spec[k] = v

        print("Using cluster plot ...")

    def create(self, fig, ax, X, z=[], y=[], tags=[]):
        """
        Parameters
        ----------
        X : array-like, shape=[n_samples,2]
        Input points.
        y : array-like, shape=[n_samples]
        label for every point
        z and tags are not used for this plot style
        """

        # get the cluster size and mean position
        from ..cluster import get_cluster_size, get_cluster_properties
        y_unique = np.unique(y)
        [_, cluster_mx] = get_cluster_properties(y, X[:, 0], 'mean')
        [_, cluster_my] = get_cluster_properties(y, X[:, 1], 'mean')
        [_, cluster_size] = get_cluster_size(y)
        s = {}
        for k in y_unique:
            s[k] = np.log(cluster_size[k]) # default is using log(frequency)

        for k in y_unique:
            ax.plot(cluster_mx[k], cluster_my[k], 'o', 
                    markerfacecolor=self.p_spec['facecolor'],
                    markeredgecolor=self.p_spec['edgecolor'], 
                    markersize=self.p_spec['circle_size'] * s[k])

        if self.p_spec['w_label'] is True:
            for k in y_unique:
                # Position of each label.
                txt = ax.annotate(str(k), xy=(cluster_mx[k], cluster_my[k]),
                              xytext=(0, 0), textcoords='offset points',
                              fontsize=self.p_spec['fontsize'], 
                              horizontalalignment='center', verticalalignment='center')
                txt.set_path_effects([
                              PathEffects.Stroke(linewidth=5, foreground='none'),
                              PathEffects.Normal()])

        return fig, ax

        
