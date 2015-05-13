import sys
import time
import pprint
import os
from itertools import product
import cPickle

from sklearn import linear_model
from sklearn import svm
from sklearn import dummy
from sklearn import kernel_ridge
import numpy

import features
import clfs
from init_data import init_data, init_data_multi, init_data_length
from utils import cross_clf_kfold, tokenize, ARYL, true_strip, erf_over_r, \
        one_over_sqrt, lennard_jones


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


def load_data2():
    '''
    '''
    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    atoms = {1: 'H', 6: "C", 7: "N", 8: "O", 16: "S"}
    with open("qm7.pkl", "r") as f:
        temp = cPickle.load(f)
        X = temp['X'].reshape(7165, 23*23)
        Z = temp['Z']
        R = temp['R']
        T = temp['T']
        P = temp['P']

    for i, (zs, coords, t) in enumerate(zip(Z, R, T)):
        # if 16 in zs.astype(int):
        #     continue

        name = "qm-%04d" % i
        path = "qm/" + name + ".out"
        # with open(path, "w") as f:
        #     for z, coord in zip(zs, coords):
        #         z = int(z)
        #         if z:
        #             f.write("%s %.8f %.8f %.8f\n" % (atoms[z], coord[0] * 0.529177249, coord[1] * 0.529177249, coord[2] * 0.529177249))
        names.append(name)
        datasets.append((1, ))
        geom_paths.append(path)
        properties.append([t])
        meta.append([])
        lengths.append(1)

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
    struct_set = ('O', 'N', '4')
    prop_set = ("homo", "lumo", "gap")


    feature_sets = (
        (features.get_null_feature, {}),
        (features.get_atom_feature, {}),
        # (features.get_bond_feature, {}),
        # (features.get_angle_feature, {}),
        # (features.get_dihedral_feature, {}),
        # (features.get_atom_and_bond_feature, {}),
        # (features.get_atom_bond_and_angle_feature, {}),
        # (features.get_atom_bond_angle_and_dihedral_feature, {}),
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
        (
            "SVM",
            svm.SVR,
            {
                'C': [10. ** x for x in xrange(-1, 4)],
                "gamma": [10. ** x for x in xrange(-4, 0)],
            }
        ),
        (
            "SVMLaplace",
            clfs.SVMLaplace,
            {
                'C': [10. ** x for x in xrange(-1, 4)],
                "gamma": [10. ** x for x in xrange(-4, 0)],
            }
        ),
    )



    names, datasets, geom_paths, properties, meta, lengths = load_data2()
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
    properties = [("all", numpy.concatenate([x for _, x in properties], axis=1))]
    dummy_results = print_property_statistics(properties, groups, cross_clf_kfold)
    sys.stdout.flush()
    results = main(features, properties, groups, CLFS, cross_clf_kfold)
    print_best_methods(results)
    # pprint.pprint(results)