import numpy
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn import kernel_ridge


class CLF(object):
    def __init__(self, **kwargs):
        '''
        Self initialization code goes here.
        '''
        pass

    def fit(self, X, y):
        '''
        X is a (N_samples, N_features) array.
        y is a (N_samples, ) array.
        NOTE: These are arrays and NOT matrices. To do matrix-like operations
        on them you need to convert them to a matrix with
        numpy.matrix(X) (or you can use numpy.dot(X, y), and etc).
        Note: This method does not return anything, it only stores state
        for later calls to self.predict()
        '''
        raise NotImplementedError

    def predict(self, X):
        '''
        X is a (N_samples, N_features) array.
        NOTE: This input is also an array and NOT a matrix.
        '''
        raise NotImplementedError


# Example Linear Regression
class LinearRegression(CLF):
    '''
    A basic implementation of Linear Regression.
    '''
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weights = None

    def fit(self, X, y):
        X = numpy.matrix(X)
        y = numpy.matrix(y).T
        self.weights = numpy.linalg.pinv(X.T * X) * X.T * y

    def predict(self, X):
        return X * self.weights


def laplace_kernel_gen(sigma):
    '''
    This is a hack to be able to pass aribitry sigma values to SVMLaplace in
    the same way that can be done with the normal RBF kernel.
    '''
    def func(X, Y):
        return numpy.exp(-sigma*cdist(X, Y, metric='cityblock'))
    return func


def laplace_kernel(X, Y, gamma=1.):
    '''
    A simple implementation of the Laplace Kernel for KRR.
    '''
    return numpy.exp(-gamma * cdist(X, Y, metric='cityblock'))



class SVMLaplace(svm.SVR):
    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=1e-3,
                 C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1):
        super(SVMLaplace, self).__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
                 C=C, epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,
                 max_iter=max_iter)
        self.kernel = laplace_kernel_gen(gamma)


class KRRLaplace(kernel_ridge.KernelRidge):
    def __init__(self, alpha=1, kernel="precomputed", gamma=1., degree=3, coef0=1, kernel_params=None):
        super(KRRLaplace, self).__init__(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree,
            coef0=coef0, kernel_params=kernel_params)
        self.kernel = "precomputed"

    def fit(self, X, y):
        self._input_X = X
        K = laplace_kernel(X, X, gamma=self.gamma)
        return super(KRRLaplace, self).fit(K, y)

    def predict(self, X):
        K = laplace_kernel(X, self._input_X, gamma=self.gamma)
        return super(KRRLaplace, self).predict(K)
