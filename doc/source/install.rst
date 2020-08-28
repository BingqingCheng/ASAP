.. _installation:

Installation
*******************************

``asap`` supports ``python 3``


Direct installation
-------------------

.. code-block:: sh

    python3 setup.py install --user

.. note:: This command should automatically install any depedencies.


Install in a conda environment
----------------------------------

.. code-block:: sh

    conda create -n myenv python=3.6
    conda activate myenv
    python3 setup.py install


List of requirements
---------------------

* numpy scipy scikit-learn json ase dscribe umap-learn PyYAML click

Add-Ons:

* (for finding symmetries of crystals) spglib 

* (for annotation without overlaps) adjustText

* The FCHL19 representation requires code from the development brach of the QML package. Instructions on how to install the QML package can be found on https://www.qmlcode.org/installation.html.


