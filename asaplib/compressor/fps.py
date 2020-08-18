"""
Farthest Point Sampling methods for sparsification
"""

import numpy as np


def fps(x, d=0, r=None):
    """
    Farthest Point Sampling

    Parameters
    ----------
    x: np.matrix
       [n_samples, n_dim] coordinates of all the samples to be sparsified.
    d: int
       number of samples to keep
    r: int
       starting from sample of index r

    Returns
    -------
    sample_index: list
        a list of selected samples, remaining error

    """

    if d == 0:
        d = len(x)
    n = len(x)
    iy = np.zeros(d, int)
    # faster evaluation of Euclidean distance
    n2 = np.einsum("ai,ai->a", x, x)
    if r is None:
        iy[0] = np.random.randint(0, n)
    else:
        iy[0] = r
    dl = n2 + n2[iy[0]] - 2 * np.dot(x, x[iy[0]])
    lmin = np.zeros(d)
    for i in range(1, d):
        iy[i] = np.argmax(dl)
        lmin[i - 1] = dl[iy[i]]
        nd = n2 + n2[iy[i]] - 2 * np.dot(x, x[iy[i]])
        dl = np.minimum(dl, nd)
    return iy, lmin


def fast_fps(x, d=0, r=None):
    if d == 0:
        d = len(x)
    n = len(x)
    iy = np.zeros(d, int)
    # faster evaluation of Euclidean distance
    n2 = np.einsum("ai,ai->a", x, x)
    if r is None:
        iy[0] = np.random.randint(0, n)
    else:
        iy[0] = r

    ldmin = n2 + n2[iy[0]] - 2 * np.dot(x, x[iy[0]])
    ivor = np.zeros(n, int)  # voronoi assignment of each point
    rvor = np.zeros(d)  # maximum distance within a voronoi ("voronoi radius")
    lmin = np.zeros(d)

    yx = np.zeros((d, len(x[0])))
    yx[0] = x[iy[0]]
    rvor[0] = np.max(ldmin)
    ndist = 0
    for i in tqdm_notebook(range(1, d)):
        inew = np.argmax(ldmin)  # index of the maximum minimum distance
        iy[i] = inew

        dnew = ldmin[inew]  # keep track of the maximum minimum distance
        lmin[i - 1] = dnew

        yx[i] = x[inew]

        # now gets the distance between this point and all the points selected
        # up to now (could be eliminated by caching!)
        syd = np.sqrt(np.abs(n2[iy[:i]] + n2[inew] -
                             2 * np.dot(yx[:i], x[inew])))
        sdnew = np.sqrt(dnew)

        # list of active voronoi (we check which of the voronoi cells contain points
        # that might contain points closer than dmax)
        iactive = np.where(rvor[:i] > dnew + syd[:i]
                           * (syd[:i] - 2 * sdnew))[0]

        rvor[iactive] = 0
        for j in xrange(n):
            if ivor[j] in iactive:
                if ldmin[j] > dnew + syd[ivor[j]] * (syd[ivor[j]] - 2 * sdnew):
                    dj = n2[j] + n2[inew] - 2 * np.dot(x[j], x[inew])
                    ndist += 1
                    if dj < ldmin[j]:
                        ivor[j] = i
                        ldmin[j] = dj
                    rvor[ivor[j]] = max(rvor[ivor[j]], ldmin[j])
    print(" computed ", ndist, "distances")
    return iy, lmin, ivor, rvor
