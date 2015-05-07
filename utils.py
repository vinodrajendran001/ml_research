import re
from itertools import product
from multiprocessing import Pool, cpu_count

from scipy.spatial.distance import cdist
import numpy
from numpy.linalg import norm

from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error


def true_strip(string, left, right):
    if string.startswith(left):
        string = string[len(left):]
    if string.endswith(right):
        string = string[:-len(right)]
    return string


def read_file_data(path):
    elements = []
    numbers = []
    coords = []
    types = {'C': 6, 'H': 1, 'O': 8, 'N': 7}
    with open(path, 'r') as f:
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            numbers.append(types[ele])
            coords.append(point)
    return elements, numbers, numpy.matrix(coords)


def get_coulomb_matrix(numbers, coords):
    top = numpy.outer(numbers, numbers)
    r = get_distance_matrix(coords, power=1)
    numpy.divide(top, r, top)
    numpy.fill_diagonal(top, 0.5 * numpy.array(numbers) ** 2.4)
    return top


def get_distance_matrix(coords, power=-1, inf_val=1):
    dist = cdist(coords, coords)
    numpy.power(dist, power, dist)
    dist[dist == numpy.Infinity] = inf_val
    return dist


def homogenize_lengths(vectors):
    n = max(len(x) for x in vectors)
    feat = numpy.zeros((len(vectors), n))
    for i, x in enumerate(vectors):
        feat[i, 0:len(x)] = x
    return numpy.matrix(feat)


def get_thermometer_encoding(X, step=1):
    '''
    This is another method of encoding floating point values so that they work
    better with neural nets.

    This is based off the work in:
    Yunho Jeon and Chong-Ho Choi. IJCNN, (3) 1685-1690, 1999.
    and the recommendation from:
    Gregoire Montavon. On Layer-Wise Representations in Deep Neural Networks.
    '''
    b = numpy.arange(0, X.max() + step, step)
    temp = numpy.tanh(numpy.subtract.outer(X,b) / step)
    return temp.reshape(-1)


def get_eigenvalues(X):
    '''
    This returns the eigenvalues of a matrix in descending order.
    '''
    eigvals = numpy.linalg.eigvals(X)
    eigvals.sort()
    return eigvals[::-1]


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


ARYL = ['2', '3', '4', '6', '11', '12', '13']
ARYL0 = ['2', '3', '11']
RGROUPS = ['a', 'd', 'e', 'f', 'h', 'i', 'l']


def tokenize(string, explicit_flips=False):
    '''
    Tokenizes a given string into the proper name segments. This includes the
    addition of '*' tokens for aryl groups that do not support r groups.

    >>> tokenize('4al')
    ['4', 'a', 'l']
    >>> tokenize('4al12ff')
    ['4', 'a', 'l', '12', 'f', 'f']
    >>> tokenize('3')
    ['3', '*', '*']
    >>> tokenize('BAD')
    ValueError: Bad Substituent Name(s): ['BAD']
    '''

    match = '(\(\d+\)|1?\d|-|[%s])' % ''.join(RGROUPS)
    tokens = [x for x in re.split(match, string) if x]

    valid_tokens = set(ARYL + RGROUPS + ['-'])

    invalid_tokens = set(tokens).difference(valid_tokens)
    if invalid_tokens:
        raise ValueError("Bad Substituent Name(s): %s" % str(list(invalid_tokens)))

    new_tokens = []
    flipped = False
    for i, token in enumerate(tokens):
        if explicit_flips and i and token in ARYL:
            new_tokens.append(flipped*"-")
            flipped = False
        elif token == "-":
            flipped = True

        if not explicit_flips or token != "-":
            new_tokens.append(token)
        if token in ARYL0:
            new_tokens.extend(['*', '*'])
    if explicit_flips:
        new_tokens.append(flipped*"-")
    return new_tokens


def decay_function(distance, power=1, H=1, factor=1):
    return (factor * (distance ** -H)) ** power


def gauss_decay_function(x, sigma=6):
    return numpy.exp(-(x / float(sigma)) ** 2)


def get_cross_validation_iter(X, y, groups, folds):
    '''
    Generate an iterator that returns the train/test splits for X, y, and
    groups.

    '''
    for train_idx, test_idx in cross_validation.KFold(
                                                        groups.max(),
                                                        n_folds=folds,
                                                        shuffle=True,
                                                        random_state=1):
        train_mask = numpy.in1d(groups, train_idx)
        test_mask = numpy.in1d(groups, test_idx)

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask].T.tolist()[0]
        y_test = y[test_mask].T.tolist()[0]
        groups_train = groups[train_mask].T.tolist()[0]
        groups_test = groups[test_mask].T.tolist()[0]

        yield X_train, X_test, y_train, y_test, groups_train, groups_test


def _parallel_params(params):
    '''
    This is a helper function to run the parallel code. It contains the same
    code that the cross_clf_kfold had in the inner loop.
    '''
    X_train, y_train, groups_train, clf_base, param_names, p_vals, test_folds = params
    params = dict(zip(param_names, p_vals))
    clf = clf_base(**params)

    X_use = numpy.matrix(X_train)
    y_use = numpy.matrix(y_train).T
    groups_use = numpy.matrix(groups_train).T

    test_mean, test_std = test_clf_kfold(X_use, y_use, groups_use, clf, folds=test_folds)
    return test_mean


def test_clf_kfold(X, y, groups, clf, folds=10):
    results = numpy.zeros(folds)

    loop = get_cross_validation_iter(X, y, groups, folds)
    for i, (X_train, X_test, y_train, y_test, _, _) in enumerate(loop):
        clf.fit(X_train, y_train)
        results[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return results.mean(), results.std()


def cross_clf_kfold(X, y, groups, clf_base, params_sets, cross_folds=10, test_folds=10):
    '''
    This runs cross validation of a clf given a set of hyperparameters to
    test. It does this by splitting the data into testing and training data,
    and then it passes the training data into the test_clf_kfold function
    to get the error. The hyperparameter set that has the lowest test error is
    then returned from the function and its respective error.
    '''
    param_names = params_sets.keys()

    n_sets = len(list(product(*params_sets.values())))
    cross = numpy.zeros((cross_folds, n_sets))

    # Calculate the cross validation errors for all of the parameter sets.
    loop = get_cross_validation_iter(X, y, groups, cross_folds)
    for i, (X_train, X_test, y_train, y_test, groups_train, groups_test) in enumerate(loop):
        data = []
        # This parallelization could probably be more efficient with an
        # iterator
        for p_vals in product(*params_sets.values()):
            data.append((X_train, y_train, groups_train, clf_base, param_names, p_vals, test_folds))

        pool = Pool(processes=min(cpu_count(), len(data)))
        results = pool.map(_parallel_params, data)
        pool.close()
        pool.terminate()
        pool.join()

        cross[i,:] = results

    # Get the set of parameters with the lowest cross validation error
    idx = numpy.argmin(cross.mean(0))
    for j, p_vals in enumerate(product(*params_sets.values())):
        if j == idx:
            params = dict(zip(param_names, p_vals))
            break

    # Get test error for set of params with lowest cross val error
    # The random_state used for the kfolds must be the same as the one used
    # before
    clf = clf_base(**params)
    return params, test_clf_kfold(X, y, groups, clf, folds=cross_folds)