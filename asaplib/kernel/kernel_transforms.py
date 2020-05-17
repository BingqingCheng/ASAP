"""
Methods and functions to convert descriptors to kernels for samples

Kernels are measures of similarity, 
i.e. s(a, b) > s(a, c) if objects a and b are considered “more similar” than objects a and c.
A kernel must also be positive semi-definite.

Essentially, for each pair of samples a and b we compute
k(a,b)
based on the coordinates of descriptors d(a) and d(b) 
"""
import numpy as np
import json
from .ml_kernel_operations import normalizekernel

class Descriptors_to_Kernels:
    def __init__(self, k_spec_dict={}):
        """
        Object handing the specification and the computation of atomic descriptors
        Parameters
        ----------
        k_spec_dict: dictionaries that specify which way to convert descriptors into kernel matrix
        e.g.
        k_spec_dict = {
        "first_kernel": {"type": 'linear', "normalize" = True},
        "second_kernel": {"type": 'cosine'},
        "third_kernel": {"type": 'polynormial', "d":3, "normalize" = True}
        }

        Notice that we can specify multiple kernels here.
        What we do is that:
        1. compute k(a,b) for all these kernel functions
        2. sum up all k(a,b).
        """

        self.k_spec_dict = k_spec_dict
        # list of kernel (similarity measurement) objects
        self.engines = {}
        self.acronym = ""

        self.bind()

    def add(self, k_spec, tag):
        """
        adding the specifications of a new kernel function
        Parameters
        ----------
        k_spec: a dictionary that specify which atomic descriptor to use 
        """
        self.k_spec_dict[tag] = k_spec

    def pack(self):
        return json.dumps(self.k_spec_dict, sort_keys=True, cls=NpEncoder)

    def get_acronym(self):
        if self.acronym == "":
            for element in self.k_spec_dict.keys():
                self.acronym += self.engines[element].get_acronym()
        return self.acronym

    def bind(self):
        """
        binds the objects that actually compute the kernels
        these objects need to have .transform() method to compute 
        kernels from decriptor matrix [n_descriptors, n_samples]
        """
        # clear up the objects
        self.engines = {}
        for element in self.k_spec_dict.keys():
            self.engines[element] = self._call(self.k_spec_dict[element])
            self.k_spec_dict[element]['acronym'] = self.engines[element].get_acronym()

    def _call(self, k_spec):
        """
        call the specific kernel objects
        """
        if "type" not in k_spec.keys():
            raise ValueError("Did not specify the type of the kernel function.")
        if k_spec["type"] == "linear":
            return Kernel_Function_Linear(k_spec)
        if k_spec["type"] == "polynomial":
            return Kernel_Function_Polynomial(k_spec)
        if k_spec["type"] == "cosine":
            return Kernel_Function_Cosine(k_spec)
        else:
            raise NotImplementedError 

    def compute(self, desc_a, desc_b=None):
        """
        compute the global descriptor vector for a frame from atomic contributions
        Parameters
        ----------
        desc : array-like, shape=[n_descriptors, n_samples]
            design matrix

        Returns
        -------
        k_mat : array-like, shape=[n_samples, n_samples]
            design matrix
        """
        if desc_b is None:
            desc_b = desc_a

        n_a = len(desc_a)
        n_b = len(desc_b)

        k_mat = np.zeros((n_a,n_b), dtype=float)
        for element in self.k_spec_dict.keys():
            #print(np.shape(k_mat), np.shape(self.engines[element].transform(desc_a, desc_b)))
            k_mat += self.engines[element].transform(desc_a, desc_b)
            
        # this is not normalized!
        return k_mat

class Kernel_Function_Base:
    def __init__(self, k_spec):
        self.acronym = ""
        pass
    def get_acronym(self):
        # we use an acronym for each descriptor, so it's easy to find it and refer to it
        return self.acronym
    def transform(self, desc_a, desc_b):
        return []

class Kernel_Function_Linear(Kernel_Function_Base):
    def __init__(self, k_spec):
        self.acronym = "linear"
        try: 
            self.normalize =  k_spec['normalize']
        except:
            self.normalize = False
    def transform(self, desc_a, desc_b):
        if self.normalize and len(desc_a) == len(desc_b):
            return normalizekernel(np.dot(desc_a, desc_b.T))
        else:
            return np.dot(desc_a, desc_b.T)

class Kernel_Function_Polynomial(Kernel_Function_Base):
    def __init__(self, k_spec):
        self.acronym = "poly"
        self.d =  k_spec['d']
        try: 
            self.normalize =  k_spec['normalize']
        except:
            self.normalize = False
    def transform(self, desc_a, desc_b):
        if self.normalize and len(desc_a) == len(desc_b):
            return normalizekernel(np.power(np.dot(desc_a, desc_b.T),self.d))
        else:
            return np.power(np.dot(desc_a, desc_b.T),self.d)

class Kernel_Function_Cosine(Kernel_Function_Base):
    def __init__(self, k_spec):
        self.acronym = 'cos'
    def transform(self, desc_a, desc_b):
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim
        return sk_cos_sim(desc_a, desc_b)

