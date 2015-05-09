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
                opt_params, (test_mean, test_std) = cross_validate(
                                                    feat,
                                                    prop,
                                                    groups,
                                                    clf,
                                                    clf_kwargs,
                                                    test_folds=test_folds,
                                                    cross_folds=cross_folds,
                                                )
                time_taken = time.time() - start
                string = "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (
                                                        clf_name,
                                                        test_mean,
                                                        test_std,
                                                        time_taken
                                                    )
                print string, opt_params
                results[prop_name][feat_name][clf_name] = (test_mean, test_std, opt_params)
            print
            sys.stdout.flush()
        print
    return results


def print_property_statistics(properties, groups, cross_validate, test_folds=5, cross_folds=2):
    results = {}
    print "Property Statistics"
    for prop_name, prop in properties:
        feat = numpy.zeros(groups.shape)
        _, (test_mean, test_std) = cross_validate(
                                                feat,
                                                prop,
                                                groups,
                                                dummy.DummyRegressor,
                                                {},
                                                test_folds=test_folds,
                                                cross_folds=cross_folds,
                                            )
        print "\t%s" % prop_name
        print "\t\tValue range: [%.4f, %.4f] eV" % (prop.min(), prop.max())
        print "\t\tExpected value: %.4f +- %.4f eV" % (prop.mean(), prop.std())
        print "\t\tExpected error: %.4f +/- %.4f eV" % (test_mean, test_std)
        results[prop_name] = (test_mean, test_std)
    print
    return results


def load_data(calc_set, opt_set, struct_set, prop_set=None):
    '''
    Load data from data sets and return lists of structure names, full paths
    to the geometry data, the properties, and the meta data.
    '''
    print "Dataset options used"
    print "\tCalculation methods:", calc_set
    print "\tOptimization methods:", opt_set
    print "\tStructure sets:", struct_set
    print "\tProperties:", prop_set
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

    print "Loaded data"
    print "\t%d datapoints" % len(names)
    print "\t%d unique molecules" % len(set(names))
    print "\t%d unique geometries" % len(set(geom_paths))
    print
    return names, datasets, geom_paths, zip(*properties), meta, lengths


def print_best_methods(results):
    for prop, prop_data in results.items():
        best = None
        for feat_name, feat_data in prop_data.items():
            for clf_name, value in feat_data.items():
                if best is None or value[0] < best[2][0]:
                    best = (feat_name, clf_name, value)
        print prop
        print best
        print


if __name__ == '__main__':
    # Select the data set to use
    calc_set = ("b3lyp", )#"cam", "m06hf")
    opt_set = tuple("opt/" + x for x in calc_set)
    struct_set = ('O', )# 'N', '4']
    prop_set = ("homo", "lumo", "gap")


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
    sys.stdout.flush()
    dummy_results = print_property_statistics(properties, groups, cross_clf_kfold)
    results = main(features, properties, groups, CLFS, cross_clf_kfold)
    print_best_methods(results)
    # pprint.pprint(results)