import sys
import time
import pprint
import os
from itertools import product

from sklearn import linear_model
from sklearn import svm
from sklearn import dummy
import numpy

import features
import clfs
from init_data import init_data
from utils import cross_clf_kfold, tokenize, ARYL, true_strip


def main(features, properties, groups, clfs, cross_validate,
                        test_folds=5, cross_folds=2):
    results = {}
    for prop_name, prop in properties:
        print prop_name
        results[prop_name] = {}
        for feat_name, feat in features.items():
            results[prop_name][feat_name] = {}
            print "\t" + feat_name
            for clf_name, clf, clf_kwargs in clfs:
                start = time.time()
                opt_params, test = cross_validate(
                                                    feat,
                                                    prop,
                                                    groups,
                                                    clf,
                                                    clf_kwargs,
                                                    test_folds=test_folds,
                                                    cross_folds=cross_folds,
                                                )
                finished = time.time() - start
                string = "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (
                                                        clf_name,
                                                        test[0],
                                                        test[1],
                                                        finished
                                                    )
                print string, opt_params
                results[prop_name][feat_name][clf_name] = test
            print
            sys.stdout.flush()
        print
    return results


def load_data(calc_set, opt_set, struct_set, prop_set=None):
    '''
    Load data from data sets and return lists of structure names, full paths
    to the geometry data, the properties, and the meta data.
    '''
    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    for j, base_path in enumerate(opt_set):
        for i, file_path in enumerate(calc_set):
            for m, atom_set in enumerate(struct_set):
                path = os.path.join('../mol_data', base_path, atom_set, file_path)
                with open(path + ".txt", 'r') as f:
                    for line in f:
                        temp = line.split()
                        name, props = temp[0], temp[1:]

                        names.append(name)
                        datasets.append((base_path, file_path, atom_set))

                        geom_path = os.path.join('../mol_data', base_path, 'geoms', 'out', name + '.out')
                        geom_paths.append(geom_path)

                        properties.append([float(x) for x in props])

                        # Add part to feature vector to account for the 4 different data sets.
                        base_part = [i == k for k, x in enumerate(opt_set)]
                        # Add part to feature vector to account for the 3 different methods.
                        method_part = [j == k for k, x in enumerate(calc_set)]
                        # Add part to feature vector to account for the addition of N.
                        atom_part = [m == k for k, x in enumerate(struct_set)]
                        # Add bias feature
                        bias = [1]
                        meta.append(base_part + method_part + atom_part + bias)

                        tokens = tokenize(name, explicit_flips=True)
                        aryl_count = sum([1 for x in tokens if x in ARYL])
                        lengths.append(aryl_count)

    return names, datasets, geom_paths, zip(*properties), meta, lengths



if __name__ == '__main__':
    # Select the data set to use
    calc_set = ("b3lyp", )#"cam", "m06hf")
    opt_set = tuple("opt/" + x for x in calc_set)
    struct_set = ['O']#, 'N']
    prop_set = ["homo", "lumo", "gap"]


    feature_sets = (
        # (features.get_flip_binary_feature, {}),
        # (features.get_coulomb_feature, {}),
        # (features.get_distance_feature, {"power": [-2, -1, 1, 2]}),
        (features.get_eigen_new_distance_feature, {"power": [-2, -1, 1, 2]}),
        # (features.get_eigen_coulomb_feature, {}),
        (features.get_eigen_distance_feature, {"power": [-2, -1, 1, 2]}),
        # (features.get_mul_feature, {}),
    )

    FEATURE_FUNCTIONS = []
    for function, kwargs_sets in feature_sets:
        for x in product(*kwargs_sets.values()):
            temp = (function, dict(zip(kwargs_sets.keys(), x)))
            FEATURE_FUNCTIONS.append(temp)

    CLFS = (
        (
            "Mean",
            dummy.DummyRegressor,
            {}
        ),
        (
            "LinearRidge",
            linear_model.Ridge,
            {
                "alpha": [10. ** x for x in xrange(-3, 4)]
            }
        ),
        (
            "SVM",
            svm.SVR,
            {
                'C': [10. ** x for x in xrange(-1, 4)],
                "gamma": [10. ** x for x in xrange(-4, 0)],
            }
        ),
    )



    start = time.time()
    names, datasets, geom_paths, properties, meta, lengths = load_data(
                                                        calc_set,
                                                        opt_set,
                                                        struct_set,
                                                        prop_set,
                                                    )

    features, properties, groups = init_data(
                                            FEATURE_FUNCTIONS,
                                            names,
                                            datasets,
                                            geom_paths,
                                            meta,
                                            lengths,
                                            properties,
                                            prop_set,
                                        )

    print "Took %.4f secs to load %d data points." % ((time.time() - start), properties[0][1].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in features.items():
        print "\t" + name, feat.shape
    print
    sys.stdout.flush()
    results = main(features, properties, groups, CLFS, cross_clf_kfold)
    pprint.pprint(results)