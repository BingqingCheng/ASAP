<p align="left">
  <img src="ASAP-logo.png" width="500" title="logo">
</p>

# ASAP 
Automatic Selection And Prediction tools for materials and molecules

### Basic usage

Type `asap` and use the sub-commands for various tasks.

To get help string:

`asap --help` .or. `asap subcommand --help` .or. `asap subcommand subcommand --help` depending which level of help you are interested in.

* `asap gen_desc`: generate global or atomic descriptors based on the input [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html)) xyze file. 

* `asap map`: make 2D plots using the specified design matrix. Currently PCA `pca`, sparsified kernel PCA `skpca`, UMAP `umap`, and t-SNE `tsne` are implemented. 

* `asap cluster`: perform density based clustering. Currently supports DBSCAN `dbscan` and [Fast search of density peaks](https://science.sciencemag.org/content/344/6191/1492) `fdb`.

* `asap fit`: fast fit ridge regression `ridge` or sparsified kernel ridge regression model `kernelridge` based on the input design matrix and labels.

* `asap kde`: quick kernel density estimation on the design matrix. Several versions of kde available.

* `asap select`: select a subset of frames using sparsification algorithms.

### Quick & basic example

The first step for a machine-learning analysis or visualization is to generate a "design matrix" made from either global descriptors or atomic descriptors. To do this, we supply `asap gen_desc` with an input file that contains the atomic coordintes. Many formats are supported; anything can be read using [ase.io](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) is supported. You can use a wildcard to specify the list of input files that matches the pattern (e.g. `POSCAR*`, `H*`, or `*.cif`).

As a quick example, in the folder ./tests/

to generate SOAP descriptors:

```bash
asap gen_desc --fxyz small_molecules-1000.xyz soap
```

for columb matrix:

```bash
asap gen_desc -f small_molecules-1000.xyz --no-periodic cm
```

After generating the descriptors, one can make a two-dimensional map (`asap map`), or regression model (`asap fit`), or clustering (`asap cluster`), or select a subset of frames (`asap select`), or do a clustering analysis (`asap cluster`), or estimate the probablity of observing each sample (`asap kde`).

For instance, to make a pca map:

```bash
asap map -f small_molecules-SOAP.xyz -dm '[SOAP-n4-l3-c1.9-g0.23]' -c dft_formation_energy_per_atom_in_eV pca
```

You can specify a list of descriptor vectors to include in the design matrix, e.g. `'[SOAP-n4-l3-c1.9-g0.23, SOAP-n8-l3-c5.0-g0.3]'`

one can use a wildcard to specify the name of all the descriptors to use for the design matrix, e.g.

```bash
asap map -f small_molecules-SOAP.xyz -dm '[SOAP*]' -c dft_formation_energy_per_atom_in_eV pca
```

or even

```bash
asap map -f small_molecules-SOAP.xyz -dm '[*]' -c dft_formation_energy_per_atom_in_eV pca
```



### Installation & requirements

python 3

Installation:

```bash
python3 setup.py install --user
```

*This should automatically install any depedencies.*

List of requirements:

+ numpy scipy scikit-learn json ase dscribe umap-learn PyYAML click

Add-Ons:
+ (for finding symmetries of crystals) spglib 
+ (for annotation without overlaps) adjustText
+ The FCHL19 representation requires code from the development brach of the QML package. Instructions on how to install the QML package can be found on https://www.qmlcode.org/installation.html.

### How to add your own atomic or global descriptors

* To add a new atomic descriptor, add a new `Atomic_Descriptor` class in the asaplib/descriptors/atomic_descriptors.py. As long as it has a `__init__()` and a `create()` method, it should be competitable with the ASAP code. The `create()` method takes an ASE Atoms object as input (see: [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html))

We have a template class for this
```python
class Atomic_Descriptor_Base:
    def __init__(self, desc_spec):
        self._is_atomic = True
        self.acronym = ""
        pass
    def is_atomic(self):
        return self._is_atomic
    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym
    def create(self, frame):
        # notice that we return the acronym here!!!
        return self.acronym, []
```

* To add a new global descriptor, add a new `Global_Descriptor` class in the asaplib/descriptors/global_descriptors.py. As long as it has a `__init__()` and a `create()` method, it is fine. The `create()` method also takes the Atoms object as input.

The template is similar with the atomic one:
```python
class Global_Descriptor_Base:
    def __init__(self, desc_spec):
        self._is_atomic = False
        self.acronym = ""
        pass
    def is_atomic(self):
        return self._is_atomic
    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym
    def create(self, frame):
        # return the dictionaries for global descriptors and atomic descriptors (if any)
        return {'acronym': self.acronym, 'descriptors': []}, {}
```

### Additional tools
In the directory ./scripts/ and ./tools/ you can find a selection of other python tools.

### Tab completion
Tab completion can be enabled by sourcing the `asap_completion.sh` script in the ./scripts/ directory. 
If a conda environment is used, you can copy this file to `$CONDA_PREFIX/etc/conda/activate.d/` to automatically load the completion upon environment activation.
