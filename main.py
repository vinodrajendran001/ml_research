import sys
import time
import pprint
from itertools import product

from sklearn import linear_model
from sklearn import svm
from sklearn import dummy
from sklearn import kernel_ridge
import numpy

import features
import clfs
from load_data import load_mol_data, load_qm7_data, load_dave_data, load_gdb13_data
from init_data import init_data, init_data_multi, init_data_length
from utils import erf_over_r, one_over_sqrt, lennard_jones, cosine_distance
from cross_validate import cross_clf_kfold


def main(features, properties, groups, clfs, cross_validate,
                        test_folds=5, cross_folds=2):
    results = {}
    for prop_name, units, prop in properties:
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
                string = "\t\t%s: %.4f +/- %.4f %s (%.4f secs)" % (
                                                        clf_name,
                                                        test_mean,
                                                        test_std,
                                                        units,
                                                        time_taken,
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
    for prop_name, units, prop in properties:
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
        print "\t\tValue range: [%.4f, %.4f] %s" % (prop.min(), prop.max(), units)
        print "\t\tExpected value: %.4f +- %.4f %s" % (prop.mean(), prop.std(), units)
        print "\t\tExpected error: %.4f +/- %.4f %s" % (test_mean, test_std, units)
        results[prop_name] = (test_mean, test_std)

        try:
            import matplotlib.pyplot as plt
            n, bins, patches = plt.hist(prop, 50, normed=1, histtype='stepfilled')
            plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
            plt.show()
        except:
            pass
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
    distance_functions = [cosine_distance, lennard_jones, erf_over_r, one_over_sqrt]
    powers = [-2, -1, -0.5, 0.5, 1, 2]
    slopes = [1., 5., 10., 20., 50., 100.]
    segments = [10, 25, 50, 100]

    feature_sets = (
        # ((features.get_null_feature, {}), ),
        # ((features.get_local_atom_zmatrix, {}), ),
        # ((features.get_atom_feature, {}), ),
        # ((features.get_atom_thermo_feature, {}), ),
        # ((features.get_connective_feature, {}), ),
        # ((features.get_bond_feature, {}), ),
        # ((features.get_fractional_bond_feature, {"slope": slopes}), ),
        # ((features.get_encoded_bond_feature, {"slope": slopes[:3], "segments": segments}), ),
        # ((features.get_angle_feature, {}), ),
        # ((features.get_angle_bond_feature, {}), ),
        # ((features.get_dihedral_feature, {}), ),
        # ((features.get_dihedral_bond_feature, {}), ),
        # ((features.get_trihedral_feature, {}), ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_fractional_bond_feature, {"slope": slopes[:4]}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_fractional_bond_feature, {"slope": slopes[:4]}),
        #     (features.get_angle_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_fractional_bond_feature, {"slope": slopes[:4]}),
        #     (features.get_angle_feature, {}),
        #     (features.get_dihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_fractional_bond_feature, {"slope": slopes[:4]}),
        #     (features.get_angle_feature, {}),
        #     (features.get_dihedral_feature, {}),
        #     (features.get_trihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_encoded_bond_feature, {"slope": slopes[:3], "segments": segments}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_encoded_bond_feature, {"slope": slopes[:3], "segments": segments}),
        #     (features.get_angle_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_encoded_bond_feature, {"slope": slopes[:3], "segments": segments}),
        #     (features.get_angle_feature, {}),
        #     (features.get_dihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_encoded_bond_feature, {"slope": slopes[:3], "segments": segments}),
        #     (features.get_angle_feature, {}),
        #     (features.get_dihedral_feature, {}),
        #     (features.get_trihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        #     (features.get_angle_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        #     (features.get_angle_feature, {}),
        #     (features.get_dihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        #     (features.get_angle_feature, {}),
        #     (features.get_dihedral_feature, {}),
        #     (features.get_trihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        #     (features.get_angle_bond_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        #     (features.get_angle_bond_feature, {}),
        #     (features.get_dihedral_feature, {}),
        # ),
        # (
        #     (features.get_atom_feature, {}),
        #     (features.get_bond_feature, {}),
        #     (features.get_angle_bond_feature, {}),
        #     (features.get_dihedral_feature, {}),
        #     (features.get_trihedral_feature, {}),
        # ),
        # ((features.get_local_zmatrix, {}), ),
        # ((features.get_full_local_zmatrix, {}), ),
        # ((features.get_bin_coulomb_feature, {}), ),
        # ((features.get_bin_eigen_coulomb_feature, {}), ),
        # ((features.get_flip_binary_feature, {}), ),
        # ((features.get_coulomb_feature, {}), ),
        # ((features.get_sum_coulomb_feature, {}), ),
        # ((features.get_eigen_coulomb_feature, {}), ),
        # ((features.get_sorted_coulomb_feature, {}), ),
        # ((features.get_sorted_coulomb_vector_feature, {}), ),
        # ((features.get_distance_feature, {"power": powers}), ),
        # ((features.get_eigen_distance_feature, {"power": powers}), ),
        # ((features.get_custom_distance_feature, {"f": distance_functions}), ),
        # ((features.get_eigen_custom_distance_feature, {"f": distance_functions}), ),
        # ((features.get_fingerprint_feature, {"size": [128, 1024, 2048]}), ),
    )

    FEATURE_FUNCTIONS = []
    for feature_group in feature_sets:
        multi_feature_sets = []
        for function, kwargs_sets in feature_group:
            single_feature_set = []
            for x in product(*kwargs_sets.values()):
                temp = (function, dict(zip(kwargs_sets.keys(), x)))
                single_feature_set.append(temp)
            multi_feature_sets.append(single_feature_set)
        FEATURE_FUNCTIONS.extend(product(*multi_feature_sets))

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
        #     "KernelRidgeLaplace",
        #     clfs.KRRLaplace,
        #     {
        #         "alpha": [10. ** x for x in xrange(-7, 1, 2)],
        #         "gamma": [10. ** x for x in xrange(-7, 7, 2)],
        #     }
        # ),
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

    if len(sys.argv) < 2:
        print "Needs a dataset argument"
        exit(0)
    if sys.argv[1] == "dave":
        names, datasets, geom_paths, properties, meta, lengths = load_dave_data()
    elif sys.argv[1] == "qm7":
        names, datasets, geom_paths, properties, meta, lengths = load_qm7_data()
    elif sys.argv[1] == "gdb13":
        names, datasets, geom_paths, properties, meta, lengths = load_gdb13_data()
    elif sys.argv[1] == "mol":
        # Select the data set to use
        calc_set = ("b3lyp", "cam", "m06hf")
        opt_set = tuple("opt/" + x for x in calc_set)
        struct_set = ('O', 'N', '4', '8')
        prop_set = ("homo", "lumo", "gap")
        names, datasets, geom_paths, properties, meta, lengths = load_mol_data(
                                                            calc_set,
                                                            opt_set,
                                                            struct_set,
                                                            prop_set,
                                                        )

    print_load_stats(names, geom_paths)
    sys.stdout.flush()
    feats, properties, groups = init_data(
                                            FEATURE_FUNCTIONS,
                                            names,
                                            datasets,
                                            geom_paths,
                                            meta,
                                            lengths,
                                            properties,
                                        )
    dummy_results = print_property_statistics(properties, groups, cross_clf_kfold)
    sys.stdout.flush()
    results = main(feats, properties, groups, CLFS, cross_clf_kfold)
    print_best_methods(results)

