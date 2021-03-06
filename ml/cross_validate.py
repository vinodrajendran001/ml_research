import sys
from itertools import product

import numpy
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error

from utils import p_map

def data_split(X, y, groups, train_sel, test_sel):
    '''
    This splits the data into test/train sets.
    '''
    X_train = X[train_sel]
    X_test = X[test_sel]
    y_train = y[train_sel].T.tolist()[0]
    y_test = y[test_sel].T.tolist()[0]
    groups_train = groups[train_sel].T.tolist()[0]
    groups_test = groups[test_sel].T.tolist()[0]
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def get_cross_validation_iter(X, y, groups, folds):
    '''
    Generate an iterator that returns the train/test splits for X, y, and
    groups.
    '''
    unique_groups = list(set([x[0,0] for x in groups]))
    for train_idx, test_idx in cross_validation.KFold(
                                        len(unique_groups),
                                        n_folds=min(folds, len(unique_groups)),
                                        shuffle=True,
                                        random_state=1):
        # This is to fix cases when some of the groups may not exist
        # when running cross validation.
        train_idx = [unique_groups[x] for x in train_idx]
        test_idx  = [unique_groups[x] for x in test_idx]

        train_mask = numpy.in1d(groups, train_idx)
        test_mask = numpy.in1d(groups, test_idx)
        yield data_split(X, y, groups, train_mask, test_mask)


def get_cross_validation_pair_iter(X, y, groups):
    '''
    Splits test/train into groups if they are already specified by their
    groups. Once they are split into test and train, all the values in both of
    the sets are given new group numbers (otherwise all of them would be in
    the same group).

    This can be used if the test/train split is known beforehand.
    '''
    train_idx = numpy.where(groups == 0)[0].tolist()[0]
    test_idx = numpy.where(groups == 1)[0].tolist()[0]
    new_groups = numpy.matrix(numpy.arange(max(train_idx + test_idx) + 1)).T
    yield data_split(X, y, new_groups, train_idx, test_idx)


def _parallel_params(params):
    '''
    This is a helper function to run the parallel code. It contains the same
    code that the cross_clf_kfold had in the inner loop.
    '''
    coord, X_train, y_train, groups_train, clf_base, param_names, p_vals, test_folds = params
    params = dict(zip(param_names, p_vals))
    clf = clf_base(**params)

    X_use = numpy.matrix(X_train)
    y_use = numpy.matrix(y_train).T
    groups_use = numpy.matrix(groups_train).T

    test_mean, test_std = test_clf_kfold(X_use, y_use, groups_use, clf, folds=test_folds)
    print "EASY", test_mean, params
    sys.stdout.flush()
    return coord, test_mean


def test_clf_kfold(X, y, groups, clf, folds=10):
    '''
    This is the main function for the evaluation of the kfolds error.
    '''
    group_set = set(groups.T.tolist()[0])
    single_split = len(group_set) == 2 and group_set == set([0, 1])
    if single_split:
        loop = get_cross_validation_pair_iter(X, y, groups)
    else:
        loop = get_cross_validation_iter(X, y, groups, folds)

    results = []
    for i, (X_train, X_test, y_train, y_test, _, _) in enumerate(loop):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        results.append(mean_absolute_error(pred, y_test))

    results = numpy.array(results)
    if single_split:
        return results.mean(), numpy.abs(pred - y_test).std()
    else:
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

    group_set = set(groups.T.tolist()[0])
    single_split = len(group_set) == 2 and group_set == set([0, 1])
    if single_split:
        loop = get_cross_validation_pair_iter(X, y, groups)
    else:
        loop = get_cross_validation_iter(X, y, groups, cross_folds)

    data = []
    # Calculate the cross validation errors for all of the parameter sets.
    # The `_` variables correspond to the test groups that get left out of the
    # cross validation portion. They are like this just to reduce confusion
    # since they are otherwise not used.
    for i, (X_train, _, y_train, _, groups_train, _) in enumerate(loop):
        # This parallelization could probably be more efficient with an
        # iterator
        for j, p_vals in enumerate(product(*params_sets.values())):
            coord = i, j
            data.append((coord, X_train, y_train, groups_train, clf_base, param_names, p_vals, test_folds))

    results_groups = p_map(_parallel_params, data)
    for coord, x in results_groups:
        cross[coord] = x

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

