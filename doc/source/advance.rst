Advanced topics
=========

How to add your own atomic or global descriptors
*********************

**To add a new atomic descriptor**, add a new ``Atomic_Descriptor`` class in the asaplib/descriptors/atomic_descriptors.py. As long as it has a ``__init__()`` and a ``create()`` method, it should be competitable with the ASAP code. The ``create()`` method takes an ASE Atoms object as input (see: [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html))

We have a template class for this

.. code-block:: python

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

**To add a new global descriptor**, add a new ``Global_Descriptor`` class in the asaplib/descriptors/global_descriptors.py. As long as it has a ``__init__()`` and a ``create()`` method, it is fine. The ``create()`` method also takes the Atoms object as input.

The template is similar with the atomic one:

.. code-block:: python

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

