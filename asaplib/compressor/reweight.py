"""
Select samples using a re-weighted distribution
The original distribution (KDE) of the sample is
\rho = exp(-F)
and we select the samples using a well-tempered distribution
\rho(\lambda) = exp(-F/\lambda)  
"""

import numpy as np

def reweight(logkde, reweight_lambda):

    """
    Parameters:
    ------------
    logkde: list, type=float. The (log of) kernel density for each sample
    reweight_lambda: float, reweighting factor

    Return:
    ------------
    sbs: list, type=int. A list of selected samples
    """
    nframes = len(logkde)

    new_kde = np.zeros(nframes)
    for i in range(nframes):
        new_kde[i] = np.exp(logkde[i] / reweight_lambda) / np.exp(logkde[i])
        # compute the normalization factor so we expect to select n samples in the end
        normalization = nkeep / np.sum(new_kde)
        new_kde *= normalization
        sbs = []
        randomchoice = np.random.rand(nframes)
        for i in range(nframes):
            if randomchoice[i] < new_kde[i]:
                sbs.append(i)

    return sbs
