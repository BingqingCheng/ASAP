"""
sparsifier class
"""

from asaplib.compressor import random_split, fps, CUR_deterministic

class Sparsifier:
    def __init__(self, sparse_mode):
        """
        Object handing the sparsification of data
        Parameters
        ----------
        sparse_mode: str
                     Type of the sparse mode
        """
        self._possible_modes = ['fps', 'cur', 'random', 'sequential']
        if sparse_mode.lower() not in self._possible_modes:
            raise NotImplementedError("Do not recognize the selected sparsification mode. \
                                    Use ([cur], [fps], [random],[sequential]).")
        else:
            self.sparse_mode = sparse_mode.lower()

    def _check(self, n_sparse, n_total):
        # sanity check
        if n_sparse > n_total:
            print("the number of representative structure is too large, please select n <= ", n_total)

    def sparsify(self, desc_or_ntotal, n_or_ratio, sparse_param=0):
        """
        Function handing the sparsification of data
        Parameters
        ----------
        desc_or_ntotal: np.matrix or int
                        Either a design matrix [n_sample, n_desc],
                        or simply the total number of samples
        n_or_ratio: int or float 
                  Either the number or the fraction of sparsified points
        sparse_param: int
                additional parameter that may be needed for the specific sparsifier used

        Returns
        ----------
        sbs: list
        a list of the indexes for the sparsified points
        """
        if isinstance(desc_or_ntotal, int):
            n_total = desc_or_ntotal
            input_desc = False
        else:
            desc = desc_or_ntotal
            n_total = len(desc_or_ntotal)
            input_desc = True

        if n_or_ratio == 1 or isinstance(n_or_ratio, float):
            n_sparse = n_total * n_or_ratio
        elif isinstance(n_or_ratio, int):
            n_sparse = n_or_ratio
        else:
            raise ValueError("the sparsification ratio/number should be a float or int.")
 
        self._check(n_sparse, n_total)

        if self.sparse_mode == 'fps':
            if not input_desc: 
                raise ValueError("fps needs design matrix")
            sbs, _ = fps(desc, n_sparse, int(sparse_param))
        elif self.sparse_mode == 'cur':
            if not input_desc:
                raise ValueError("cur needs design matrix")
            import numpy as np
            cov = np.dot(np.asmatrix(desc), np.asmatrix(desc).T)
            sbs, _ = CUR_deterministic(cov, n_sparse)
        elif self.sparse_mode == 'random':
            _, sbs = random_split(n_total, n_sparse/n_total)
        elif self.sparse_mode == 'sequential':
            sbs = range(n_sparse)
        else:
            raise ValueError("sparse mode not right")

        return sbs
