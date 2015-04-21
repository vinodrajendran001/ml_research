import sys
import time
import pprint
import os

from sklearn import linear_model
from sklearn import svm
from sklearn import dummy
import numpy

import features
import clfs
from utils import cross_clf_kfold, tokenize, ARYL

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

    return names, geom_paths, zip(*properties), meta, lengths


def init_data(functions, names, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = {}
    groups = numpy.matrix(xrange(len(names))).T

    for function in functions:
        key = function.__name__.lstrip("get_").rstrip("_feature")
        temp = function(names, geom_paths)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        features[key] = numpy.concatenate((temp, meta), 1)
    properties = [numpy.matrix(x).T for x in properties]

    return features, properties, groups


def init_data_multi(functions, names, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = {}
    temp_groups = numpy.matrix(xrange(len(names))).T
    groups = numpy.concatenate([temp_groups for x in properties])

    for function in functions:
        key = function.__name__.lstrip("get_").rstrip("_feature")
        temp = function(names, geom_paths)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        temps = []
        for i, x in enumerate(properties):
            bla = numpy.matrix(numpy.zeros((temp.shape[0], len(properties))))
            bla[:,i] = 1
            temps.append(numpy.concatenate((temp, meta, bla), 1))
        features[key] = numpy.concatenate(temps)
    properties = numpy.concatenate([numpy.matrix(x).T for x in properties])


    return features, properties, groups


def get_splits(names, lengths, split_length=2):
    name_idxs = {name: i for i, name in enumerate(names)}
    for name, length in zip(names, lengths):
        try:
            if length > 2:
                tokens = tokenize(name, explicit_flips=True)

                parts = []
                max_idx = 4 * split_length
                for i in xrange(0, max_idx, 4):
                    parts.append(''.join(tokens[i:i + 3]).replace('*', ''))
                parts.append(''.join(parts))

                yield [name_idxs[(x, dataset)] for x in parts], name_idxs[(name, dataset)]
        except KeyError:
            continue


def init_data_length(functions, names, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = {}

    for function in functions:
        key = function.__name__.lstrip("get_").rstrip("_feature")
        temp = function(names, geom_paths)

        groups = []

        new_temp = []
        other_props = []

        new_meta  = []
        new_properties = []
        new_lengths = []

        for i, (other_idxs, long_idx) in enumerate(get_splits(names, lengths)):
            short_props = [[x[idx] for x in properties] for idx in other_idxs]
            other_props.append(sum(short_props, []))

            new_meta.append(meta[long_idx])
            new_properties.append([x[long_idx] for x in properties])
            new_temp.append(temp[long_idx,:].tolist()[0])
            new_lengths.append(lengths[long_idx])

            groups.append(i)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        features[key] = numpy.concatenate((new_temp, other_props, new_meta), 1)

    groups = numpy.matrix(groups).T
    properties = [numpy.matrix(x).T for x in properties]
    return features, properties, groups



if __name__ == '__main__':
    # Select the data set to use
    calc_set = ("b3lyp", )#"cam", "m06hf")
    opt_set = tuple("opt/" + x for x in calc_set)
    struct_set = ['O']#, 'N']
    prop_set = ["homo", "lumo", "gap"]


    FEATURE_FUNCTIONS = (
        features.get_flip_binary_feature,
    )

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
    names, geom_paths, properties, meta, lengths = load_data(
                                                        calc_set,
                                                        opt_set,
                                                        struct_set,
                                                        prop_set,
                                                    )

    features, properties, groups = init_data(
                                            FEATURE_FUNCTIONS,
                                            names,
                                            geom_paths,
                                            meta,
                                            lengths,
                                            properties,
                                        )
    # properties = [["all", properties]]
    properties = zip(prop_set, properties)
    print "Took %.4f secs to load %d data points." % ((time.time() - start), properties[0][1].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in features.items():
        print "\t" + name, feat.shape
    print
    sys.stdout.flush()
    results = main(features, properties, groups, CLFS, cross_clf_kfold)
    pprint.pprint(results)