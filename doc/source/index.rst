.. asapdoc documentation master file, created by
   sphinx-quickstart on Mon Aug  3 19:06:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Automatic Selection And Prediction tools for materials and molecules
====================================================================

Basic usage of the command line tool
************************************

Type ``asap`` and use the sub-commands for various tasks.

* ``asap gen_desc``: generate global or atomic descriptors based on the input [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html)) xyze file. 

* ``asap map``: make 2D plots using the specified design matrix. Currently PCA ``pca``, sparsified kernel PCA ``skpca``, UMAP ``umap``, and t-SNE ``tsne`` are implemented. 

* ``asap cluster``: perform density based clustering. Currently supports DBSCAN ``dbscan`` and [Fast search of density peaks](https://science.sciencemag.org/content/344/6191/1492) ``fdb``.

* ``asap fit``: fast fit ridge regression ``ridge`` or sparsified kernel ridge regression model ``kernelridge`` based on the input design matrix and labels.

* ``asap kde``: quick kernel density estimation on the design matrix. Several versions of kde available.

* ``asap select``: select a subset of frames using sparsification algorithms.

.. note::  To get help string:
    ``asap --help`` .or. ``asap subcommand --help`` .or. ``asap subcommand subcommand --help`` depending which level of help you are interested in.


Using the PYTHON library
************************

In ``python3``, one can import ``asap`` as a libaray using

.. code-block:: python

    import asaplib

Please refer to the documentation of each module for available functionalities.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   install.rst
   example.rst
   gen_desc.rst
   map.rst
   fit.rst
   kde.rst
   cluster.rst
   select.rst
   howto_asaplib.rst
   tutorials.rst
   advance.rst
   modules.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
