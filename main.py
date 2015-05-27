import sys
import time
import pprint
from itertools import product

from sklearn import linear_model
from sklearn import svm
from sklearn import dummy
from sklearn import kernel_ridge
import numpy
import matplotlib.pyplot as plt

import features
import clfs
from load_data import load_mol_data, load_gdb7_data, load_dave_data
from init_data import init_data, init_data_multi, init_data_length
from utils import erf_over_r, one_over_sqrt, lennard_jones
from cross_validate import cross_clf_kfold


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

        n, bins, patches = plt.hist(prop, 50, normed=1, histtype='stepfilled')
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        plt.show()

    print
    return results


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


def print_load_stats(names, paths):
    print "Loaded data"
    print "\t%d datapoints" % len(names)
    print "\t%d unique molecules" % len(set(names))
    print "\t%d unique geometries" % len(set(geom_paths))
    print


if __name__ == '__main__':
    # Select the data set to use
    calc_set = ("b3lyp", )#"cam", "m06hf")
    opt_set = tuple("opt/" + x for x in calc_set)
    struct_set = ('O', 'N', '4', '8')
    prop_set = ("homo", "lumo", "gap")


    feature_sets = (
        # (features.get_null_feature, {}),
        # (features.get_atom_feature, {}),
        # (features.get_atom_thermo_feature, {}),
        # (features.get_connective_feature, {}),
        # (features.get_bond_feature, {}),
        # (features.get_angle_feature, {}),
        # (features.get_angle_bond_feature, {}),
        # (features.get_dihedral_feature, {}),
        # (features.get_trihedral_feature, {}),
        # (features.get_atom_and_bond_feature, {}),
        # (features.get_atom_bond_and_angle_feature, {}),
        # (features.get_atom_bond_and_angle_bond_feature, {}),
        # (features.get_atom_bond_angle_and_dihedral_feature, {}),
        # (features.get_atom_bond_angle_bond_and_dihedral_feature, {}),
        # (features.get_atom_bond_angle_dihedral_and_trihedral_feature, {}),
        # (features.get_local_zmatrix, {}),
        # (features.get_bin_coulomb_feature, {}),
        # (features.get_bin_eigen_coulomb_feature, {}),
        # (features.get_flip_binary_feature, {}),
        # (features.get_coulomb_feature, {}),
        # (features.get_sum_coulomb_feature, {}),
        # (features.get_eigen_coulomb_feature, {}),
        # (features.get_sorted_coulomb_feature, {}),
        # (features.get_distance_feature, {"power": [-2, -1]}),
        # (features.get_eigen_distance_feature, {"power": [-2, -1]}),#[-2, -1, -0.5, 0.5, 1, 2]}),
        # (features.get_custom_distance_feature, {"f": [lennard_jones, erf_over_r, one_over_sqrt]}),
        # (features.get_eigen_custom_distance_feature, {"f": [lennard_jones, erf_over_r, one_over_sqrt]}),
        # (features.get_fingerprint_feature, {"size": [128, 1024, 2048]}),
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
            "KernelRidge",
            kernel_ridge.KernelRidge,
            {
                "alpha": [10. ** x for x in xrange(-7, 1, 2)],
                "kernel": ["rbf"],
                "gamma": [10. ** x for x in xrange(-7, 7, 2)],
            }
        ),
        # (
        #     "SVM",
        #     svm.SVR,
        #     {
        #         'C': [10. ** x for x in xrange(-1, 4)],
        #         "gamma": [10. ** x for x in xrange(-4, 0)],
        #     }
        # ),
        # (
        #     "SVMLaplace",
        #     clfs.SVMLaplace,
        #     {
        #         'C': [10. ** x for x in xrange(-1, 4)],
        #         "gamma": [10. ** x for x in xrange(-4, 0)],
        #     }
        # ),
    )

    # names, datasets, geom_paths, properties, meta, lengths = load_dave_data()
    names, datasets, geom_paths, properties, meta, lengths = load_gdb7_data()
    # names, datasets, geom_paths, properties, meta, lengths = load_mol_data(
    #                                                     calc_set,
    #                                                     opt_set,
    #                                                     struct_set,
    #                                                     prop_set,
    #                                                 )
    print_load_stats(names, geom_paths)
    sys.stdout.flush()
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
    dummy_results = print_property_statistics(properties, groups, cross_clf_kfold)
    sys.stdout.flush()
    results = main(features, properties, groups, CLFS, cross_clf_kfold)
    print_best_methods(results)
    # pprint.pprint(results)