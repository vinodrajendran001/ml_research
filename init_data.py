import numpy

from utils import true_strip, tokenize


def get_name_groups(names, datasets):
    groups = []
    seen = {}
    count = 0
    for name in names:
        if name not in seen:
            seen[name] = count
            count += 1
        groups.append(seen[name])

    return numpy.matrix(groups).T


def init_data(functions, names, datasets, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = {}
    groups = get_name_groups(names, datasets)

    for function, kwargs in functions:
        key = true_strip(function.__name__, "get_", "_feature") + " " + repr(kwargs)
        temp = function(names, geom_paths, **kwargs)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        features[key] = numpy.concatenate((temp, meta), 1)
    properties = [numpy.matrix(x).T for x in properties]

    return features, properties, groups


def init_data_multi(functions, names, datasets, geom_paths, meta, lengths, properties):
    # Construct (name, vector) pairs to auto label features when iterating over them
    features = {}
    temp_groups = get_name_groups(names, datasets)
    groups = numpy.concatenate([temp_groups for x in properties])

    for function, kwargs in functions:
        key = true_strip(function.__name__, "get_", "_feature") + " " + repr(kwargs)
        temp = function(names, geom_paths, **kwargs)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        temps = []
        for i, x in enumerate(properties):
            bla = numpy.matrix(numpy.zeros((temp.shape[0], len(properties))))
            bla[:,i] = 1
            temps.append(numpy.concatenate((temp, meta, bla), 1))
        features[key] = numpy.concatenate(temps)
    properties = numpy.concatenate([numpy.matrix(x).T for x in properties])


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
    features = {}

    for function, kwargs in functions:
        key = true_strip(function.__name__, "get_", "_feature") + " " + repr(kwargs)
        temp = function(names, geom_paths, **kwargs)

        groups = []

        new_temp = []
        other_props = []

        new_meta  = []
        new_properties = []
        new_lengths = []

        for i, (other_idxs, long_idx) in enumerate(get_length_splits(names, datasets, lengths)):
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