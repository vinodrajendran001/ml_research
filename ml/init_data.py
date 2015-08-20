import time
import sys
from collections import OrderedDict
import hashlib
import os

import numpy

from utils import true_strip, tokenize, mkdir_p


def get_name_groups(names):
    '''
    Return a matrix of groups based on unique names.
    '''
    groups = []
    seen = {}
    count = 0
    for name in names:
        if name not in seen:
            seen[name] = count
            count += 1
        groups.append(seen[name])

    return numpy.matrix(groups).T


def get_feature_vector(function, key, names, geom_paths, kwargs):
    try:
        base_path = os.path.join(sys.argv[2], sys.argv[1])
    except IndexError:
        base_path = os.path.join("cache", sys.argv[1])
    mkdir_p(base_path)
    cleaned_key = key.replace(" ", "_")
    path = os.path.join(base_path, cleaned_key + ".npy")
    try:
        with open(path, 'rb') as f:
            temp = numpy.load(f)
            if temp.shape[0] != len(names):
                raise IOError
    except IOError:
        temp = function(names, geom_paths, **kwargs)
        with open(path, 'wb') as f:
            numpy.save(f, temp)
    return temp


def get_base_features(function_sets, names, geom_paths):
    '''
    Return an OrderedDict of final features.
    '''
    components = {}
    final_features = OrderedDict()

    print "Sizes of Feature Matrices"
    sys.stdout.flush()
    for function_set in function_sets:
        start = time.time()
        keys = []
        for function, kwargs in function_set:
            key = true_strip(function.__name__, "get_", "_feature") + " " + repr(kwargs)
            if key not in components:
                components[key] = get_feature_vector(function, key, names, geom_paths, kwargs)
            keys.append(key)

        final_key = " | ".join(keys)
        final_features[final_key] = numpy.concatenate([components[key] for key in keys], 1)
        print "\t%s %s (%.4f secs)" % (final_key, final_features[final_key].shape, time.time() - start)
        sys.stdout.flush()
    return final_features


def init_data(functions, names, datasets, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = OrderedDict()
    groups = get_name_groups(names)
    base_features = get_base_features(functions, names, geom_paths)

    for key, feature in base_features.items():
        features[key] = numpy.concatenate((feature, meta), 1)

    print
    sys.stdout.flush()
    properties = [(name, units, numpy.matrix(y).T) for name, units, y in properties]
    return features, properties, groups


def init_data_multi(functions, names, datasets, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = OrderedDict()
    temp_groups = get_name_groups(names)
    groups = numpy.concatenate([temp_groups for x in properties])
    base_features = get_base_features(functions, names, geom_paths)

    for key, feature in base_features.items():
        temps = []
        for i, x in enumerate(properties):
            prop_meta = numpy.matrix(numpy.zeros((feature.shape[0], len(properties))))
            prop_meta[:,i] = 1
            temps.append(numpy.concatenate((feature, meta, prop_meta), 1))
        features[key] = numpy.concatenate(temps)

    print
    sys.stdout.flush()
    temp_properties = numpy.concatenate([numpy.matrix(y).T for name, units, y in properties])
    properties = [("all", "units", temp_properties)]
    return features, properties, groups


def get_length_splits(names, datasets, lengths, split_length=2):
    name_idxs = {pair: i for i, pair in enumerate(zip(names, datasets))}
    for name, dataset, length in zip(names, datasets, lengths):
        try:
            if length > split_length:
                tokens = tokenize(name, explicit_flips=True)

                parts = []
                max_idx = 4 * split_length
                for i in xrange(0, max_idx, 4):
                    parts.append(''.join(tokens[i:i + 3]).replace('*', ''))
                parts.append(''.join(parts))

                yield [name_idxs[(x, dataset)] for x in parts], name_idxs[(name, dataset)]
        except KeyError:
            continue


def init_data_length(functions, names, datasets, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = OrderedDict()
    base_features = get_base_features(functions, names, geom_paths)

    for key, feature in base_features.items():
        groups = []

        new_feature = []
        other_props = []

        new_meta  = []
        new_properties = []
        new_lengths = []

        for i, (other_idxs, long_idx) in enumerate(get_length_splits(names, datasets, lengths)):
            short_props = [[x[idx] for x in properties] for idx in other_idxs]
            other_props.append(sum(short_props, []))

            new_meta.append(meta[long_idx])
            new_properties.append([x[long_idx] for x in properties])
            new_feature.append(feature[long_idx,:].tolist()[0])
            new_lengths.append(lengths[long_idx])

            groups.append(i)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        features[key] = numpy.concatenate((new_feature, other_props, new_meta), 1)

    print
    sys.stdout.flush()
    groups = numpy.matrix(groups).T
    properties = [(x, numpy.matrix(y).T) for x, y in zip(prop_set, properties)]
    return features, properties, groups
