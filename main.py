import sys
import time
import pprint
from itertools import product
from functools import partial

from sklearn import linear_model
import numpy

import ml.features as features
import ml.clfs as clfs
from ml.load_data import load_mol_data, load_qm7_data, load_dave_data, load_gdb13_data, load_quambo_data
from ml.init_data import init_data, init_data_multi, init_data_length
from ml.utils import erf_over_r, one_over_sqrt, lennard_jones, cosine_distance
from ml.utils import print_property_statistics, print_best_methods ,print_load_stats
from ml.cross_validate import cross_clf_kfold


def main(features, properties, groups, clfs, cross_validate,
                        test_folds=5, cross_folds=5):
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


if __name__ == '__main__':
    distance_functions = [cosine_distance, lennard_jones, erf_over_r, one_over_sqrt]
    powers = [-2, -1, -0.5, 0.5, 1, 2]
    slopes = [10., 20., 30.]
    segments = [10, 25, 50, 100]

    atom_features = [
        # ((features.get_atom_feature, {}), ),
        # ((features.get_atom_env_feature, {}), ),
        # ((features.get_atom_thermo_feature, {}), ),
    ]
    bond_features = [
        ((features.get_bond_feature, {}), ),
        # ((features.get_fractional_bond_feature, {"slope": slopes}), ),
        # ((features.get_encoded_bond_feature, {"slope": slopes, "segments": segments}), ),
    ]
    angle_features = [
        # ((features.get_angle_feature, {}), ),
        # ((features.get_angle_bond_feature, {}), ),
        # ((features.get_encoded_angle_feature, {"slope": slopes, "segments": segments}), ),
    ]
    dihedral_features = [
        # ((features.get_dihedral_feature, {}), ),
        # ((features.get_dihedral_bond_feature, {}), ),
    ]
    trihedral_features = [
        # ((features.get_trihedral_feature, {}), ),
    ]
    other_features = [
        # ((features.get_null_feature, {}), ),
        # ((features.get_local_atom_zmatrix_feature, {}), ),
        # ((features.get_connective_feature, {}), ),
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
    ]

    extended_features = []
    for group in product(
                        atom_features,
                        [None] + bond_features,
                        [None] + angle_features,
                        [None] + dihedral_features,
                        [None] + trihedral_features):
        try:
            idx = group.index(None)
            new_group = [x[0] for x in group[:idx]]
            if all(x is None for x in group[idx:]):
                extended_features.append(new_group)
        except ValueError:
            pass


    feature_sets = atom_features \
                + bond_features \
                + angle_features \
                + dihedral_features \
                + trihedral_features \
                + other_features \
                + extended_features

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
            clfs.KRR,
            {
                "alpha": [10. ** x for x in xrange(-7, 1, 2)],
                "kernel": ["rbf", "laplace"],
                "gamma": [10. ** x for x in xrange(-7, 7, 2)],
            }
        ),
        # (
        #     "SVM",
        #     clfs.SVM,
        #     {
        #         'C': [10. ** x for x in xrange(-1, 4)],
        #         "gamma": [10. ** x for x in xrange(-4, 0)],
        #     }
        # ),
    )
    calc_set = ("b3lyp", "cam", "m06hf")
    opt_set = tuple("opt/" + x for x in calc_set)
    struct_set = ('O', 'N', '4', '8')
    prop_set = ("homo", "lumo", "gap")

    options = {
        "dave": load_dave_data,
        "qm7": load_qm7_data,
        "gdb13": load_gdb13_data,
        "mol": partial(load_mol_data, calc_set, opt_set, struct_set, prop_set),
        "quambo": load_quambo_data,
    }

    try:
        func = options[sys.argv[1]]
    except KeyError:
        print "Not a valid dataset. Must be one of %s" % options.keys()
        exit(0)
    except IndexError:
        print "Needs a dataset argument"
        exit(0)

    names, datasets, geom_paths, properties, meta, lengths = func()

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

