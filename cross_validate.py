import sys
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error


def get_cross_validation_iter(X, y, groups, folds):
    '''
    Generate an iterator that returns the train/test splits for X, y, and
    groups.
    '''
    unique_groups = list(set([x[0,0] for x in groups]))
    for train_idx, test_idx in cross_validation.KFold(
                                        len(unique_groups),
                                        n_folds=folds,
                                        shuffle=True,
                                        random_state=1):
        # This is to fix cases when some of the groups may not exist
        # when running cross validation.
        train_idx = [unique_groups[x] for x in train_idx]
        test_idx  = [unique_groups[x] for x in test_idx]

        train_mask = numpy.in1d(groups, train_idx)
        test_mask = numpy.in1d(groups, test_idx)

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask].T.tolist()[0]
        y_test = y[test_mask].T.tolist()[0]
        groups_train = groups[train_mask].T.tolist()[0]
        groups_test = groups[test_mask].T.tolist()[0]
        yield X_train, X_test, y_train, y_test, groups_train, groups_test


def get_cross_validation_iter2(X, y, groups):
    train_idx = numpy.where(groups == 0)[0].tolist()[0]
    test_idx = numpy.where(groups == 1)[0].tolist()[0]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx].T.tolist()[0]
    y_test = y[test_idx].T.tolist()[0]
    groups_train = numpy.arange(len(train_idx))
    groups_test = numpy.arange(len(test_idx))
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
    print test_mean, params
    sys.stdout.flush()
    return test_mean


def test_clf_kfold(X, y, groups, clf, folds=10):
    if len(set(groups.T.tolist()[0])) == 2:
        loop = get_cross_validation_iter2(X, y, groups)
        results = numpy.zeros(1)
    else:
        loop = get_cross_validation_iter(X, y, groups, folds)
        results = numpy.zeros(folds)

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

    single_split = len(set(groups.T.tolist()[0])) == 2
    if single_split:
        loop = get_cross_validation_iter2(X, y, groups)
    else:
        loop = get_cross_validation_iter(X, y, groups, cross_folds)

    # Calculate the cross validation errors for all of the parameter sets.
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

