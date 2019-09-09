"""
adapted from Felix Musil's ml_tools
"""

from .base import RegressorBase
from .base import np


class KRR(RegressorBase):
    _pairwise = True
    
    def __init__(self, jitter):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter  # noise level^2
        self.coninv = None  # inverse of the covariance matrix
    
    def fit(self, kernel, y):
        '''Train the krr model with trainKernel and trainLabel.'''

        reg = np.eye(kernel.shape[0])*self.jitter
        self.coninv = np.linalg.inv(kernel+reg)
        self.alpha = np.linalg.solve(kernel+reg, y)
        
    def predict(self, kernel):
        '''kernel.shape is expected as (nPred, nTrain)'''
        return np.dot(kernel, self.alpha.flatten()).reshape((-1))

    def predict_error(self, k, delta):
        '''
        k.shape is expected as (nPred, nTrain), delta is the variance of y
        '''
        n_k = len(k)
        y_error = np.zeros(n_k)
        for i in range(n_k):
            y_error[i] = np.sqrt(delta*(1.-np.dot(k[i],np.dot(self.coninv,k[i]))))
        return y_error

    def get_params(self, deep=True):
        return dict(sigma=self.jitter)
        
    def set_params(self, params, deep=True):
        self.jitter = params['jitter']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha, jitter=self.jitter)
        return state

    def unpack(self, state):
        self.alpha = state['weights']
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter, state['jitter'])
        assert self.jitter == state['jitter'], err_m

    def loads(self, state):
        self.alpha = state['weights']
        self.jitter = state['jitter']


class KRRSparse(RegressorBase):
    _pairwise = True
    
    def __init__(self, jitter, delta, sigma):
        # Weights of the krr model
        self.alpha = None
        self.jitter = jitter
        self.delta = delta  # variance of the prior
        self.sigma = sigma  # noise
    
    def fit(self, kMM, kNM,y):
        '''N train structures, M sparsified representative structures '''
        '''kMM: the kernel matrix of the representative structures with shape (M,M)'''
        '''kNM: the kernel matrix between the representative and the train structures with shape (N,M)'''

        #if (kMM.shape[0] != kMM.shape[1]):# or kMM.shape[0] != kNM.shape[1] or kNM.shape[0] != y.shape[0]):
            #raise ValueError('Shape of the kernel matrix is not consistent!')        

        sparseK = kMM * self.delta * self.sigma**2 + np.dot(kNM.T,kNM)*self.delta**2
        sparseY = np.dot(kNM.T, y)
        reg = np.eye(kMM.shape[0])*self.jitter
        
        self.alpha = np.linalg.solve(sparseK+reg, sparseY)
        
    def predict(self, kernel):
        '''kernel.shape is expected as (nPred,nTrain)'''
        return np.dot(self.delta**2*kernel,self.alpha.flatten()).reshape((-1))

    def get_params(self, deep=True):
        return dict(jitter=self.jitter, delta=self.delta, sigma=self.sigma)
        
    def set_params(self, params,deep=True):
        self.jitter = params['jitter']
        self.delta = params['delta']
        self.sigma = params['sigma']
        self.alpha = None

    def pack(self):
        state = dict(weights=self.alpha,jitter=self.jitter,delta=self.delta,sigma=self.sigma)
        return state

    def unpack(self,state):
        self.alpha = state['weights']
        self.delta = state['delta']
        self.sigma = state['sigma']
        err_m = 'jitter are not consistent {} != {}'.format(self.jitter ,state['jitter'])
        assert self.jitter == state['jitter'], err_m

    def loads(self,state):
        self.alpha = state['weights']
        self.jitter = state['jitter']
        self.delta = state['delta']
        self.sigma = state['sigma']


class KRRFastCV(RegressorBase):
    """ 
    taken from:
    An, S., Liu, W., & Venkatesh, S. (2007). 
    Fast cross-validation algorithms for least squares support vector machine and kernel ridge regression. 
    Pattern Recognition, 40(8), 2154-2162. https://doi.org/10.1016/j.patcog.2006.12.015
    """
    _pairwise = True
    
    def __init__(self,jitter,delta,cv):
        self.jitter = jitter
        self.cv = cv
        self.delta = delta
    
    def fit(self, kernel,y):
        '''Fast cv scheme. Destroy kernel.'''
        np.multiply(self.delta**2, kernel, out=kernel)
        kernel[np.diag_indices_from(kernel)] += self.jitter
        kernel = np.linalg.inv(kernel)
        alpha = np.dot(kernel, y)
        Cii = []
        beta = np.zeros(alpha.shape)
        self.y_pred = np.zeros(y.shape)
        self.error = np.zeros(y.shape)
        for _, test in self.cv.split(kernel):
            Cii = kernel[np.ix_(test,test)]
            beta = np.linalg.solve(Cii,alpha[test]) 
            self.y_pred[test] = y[test] - beta
            self.error[test] = beta # beta = y_true - y_pred 

        del kernel

    def predict(self, kernel=None):
        '''kernel.shape is expected as (nPred, nTrain)'''
        return self.y_pred

    def get_params(self, deep=True):
        return dict(sigma=self.jitter, cv=self.cv)

    def set_params(self, params, deep=True):
        self.jitter = params['jitter']
        self.cv = params['cv']
        self.delta = params['delta']
        self.y_pred = None

    def pack(self):
        state = dict(y_pred=self.y_pred, cv=self.cv.pack(),
                     jitter=self.jitter, delta=self.delta)
        return state

    def unpack(self, state):
        self.y_pred = state['y_pred']
        self.cv.unpack(state['cv'])
        self.delta = state['delta']

        err_m = 'jitter are not consistent {} != {}'.format(self.jitter, state['jitter'])
        assert self.jitter == state['jitter'], err_m

    def loads(self, state):
        self.y_pred = state['y_pred']
        self.cv.loads(state['cv'])
        self.jitter = state['jitter']
        self.delta = state['delta']
