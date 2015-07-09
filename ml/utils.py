import re
import os
from multiprocessing import Pool, cpu_count

from sklearn import dummy
from scipy.special import erf
import numpy

from constants import ELE_TO_NUM, ARYL, ARYL0, RGROUPS, BOND_LENGTHS
try:
    from plots import get_histogram_plot, get_matrix_plot
except ImportError:
    pass

def mkdir_p(path):
    '''
    This function acts the same way as the unix command:
    $ mkdir -p some/path/
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def p_map(f, data):
    '''
    A wrapper function to make parallel mapping easier. This function collects
    together all of the cleanup code that is required.
    '''
    pool = Pool(processes=min(cpu_count(), len(data)))
    results = pool.map(f, data)
    pool.close()
    pool.terminate()
    pool.join()
    return results


def true_strip(string, left, right):
    '''
    A function the properly does left and right strip of strings.
    The default python methods will remove any characters from end
    strings that appear at the end of a string.
    EX:
    >>> "test_td".rstrip("_td")
    "tes"
    This fixes the issue and returns does the following
    >>> true_strip("test_td", "", "_td")
    "test"
    '''
    if string.startswith(left):
        string = string[len(left):]
    if string.endswith(right):
        string = string[:-len(right)]
    return string


def read_file_data(path):
    """
    Reads a file and extracts the molecules geometry

    The file should be in the format
    ele0 x0 y0 z0
    ele1 x1 y1 z1
    ...
    """
    elements = []
    numbers = []
    coords = []
    with open(path, 'r') as f:
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            numbers.append(ELE_TO_NUM[ele])
            coords.append(point)
    return elements, numbers, numpy.matrix(coords)



def tokenize(string, explicit_flips=False):
    '''
    Tokenizes a given string into the proper name segments. This includes the
    addition of '*' tokens for aryl groups that do not support r groups.

    >>> tokenize('4al')
    ['4', 'a', 'l']
    >>> tokenize('4al12ff')
    ['4', 'a', 'l', '12', 'f', 'f']
    >>> tokenize('3')
    ['3', '*', '*']
    >>> tokenize('BAD')
    ValueError: Bad Substituent Name(s): ['BAD']
    '''

    match = '(\(\d+\)|1?\d|-|[%s])' % ''.join(RGROUPS)
    tokens = [x for x in re.split(match, string) if x]

    valid_tokens = set(ARYL + RGROUPS + ['-'])

    invalid_tokens = set(tokens).difference(valid_tokens)
    if invalid_tokens:
        raise ValueError("Bad Substituent Name(s): %s" % str(list(invalid_tokens)))

    new_tokens = []
    flipped = False
    for i, token in enumerate(tokens):
        if explicit_flips and i and token in ARYL:
            new_tokens.append(flipped*"-")
            flipped = False
        elif token == "-":
            flipped = True

        if not explicit_flips or token != "-":
            new_tokens.append(token)
        if token in ARYL0:
            new_tokens.extend(['*', '*'])
    if explicit_flips:
        new_tokens.append(flipped*"-")
    return new_tokens


def decay_function(distance, power=1, H=1, factor=1):
    '''
    A simple power based decay function.
    '''
    return (factor * (distance ** -H)) ** power


def gauss_decay_function(x, sigma=6):
    '''
    A gaussian based decay function.
    '''
    return numpy.exp(-(x / float(sigma)) ** 2)


def erf_over_r(r):
    mat = erf(r) / r
    mat[mat == numpy.Infinity] = 1
    mat[numpy.isnan(mat)] = 1
    return mat


def one_over_sqrt(r):
    mat = 1. / numpy.sqrt(1 + numpy.square(r))
    mat[mat == numpy.Infinity] = 1
    mat[numpy.isnan(mat)] = 1
    return mat


def lennard_jones(r):
    six = r ** -6
    mat = six ** 2 - six
    mat[mat == numpy.Infinity] = 1
    mat[numpy.isnan(mat)] = 1
    return mat


def cosine_distance(r, cutoff=6.):
    '''
    This is based off the distance function from:
    Jorg Behler and Michele Parrinello
    Phys. Rev. Lett. 98, 146401 Published 2 April 2007
    '''
    temp = 0.5 * (numpy.cos(numpy.pi * r / cutoff) + 1)
    temp[r > 6] = 0
    return temp


def calculate_forces(clf, numbers, coords, meta=None, h=1e-6):
    '''
    A function that uses finite differences to calculate forces of a molecule
    using the given clf. The default feature vector for this to use is the
    coulomb matrix.
    '''
    if meta is None:
        meta = [1]

    vectors = []
    for i, coord in enumerate(coords):
        for j in xrange(3):
            for sign in [-1, 1]:
                new_coords = coords.copy()
                new_coords[i, j] += sign * h / 2
                mat = get_coulomb_matrix(numbers, new_coords)
                mat[mat < 0] = 0
                vectors.append(mat[numpy.tril_indices(mat.shape[0])].tolist() + meta)

    results = clf.predict(numpy.matrix(vectors))

    forces = numpy.zeros(coords.shape)
    for i, coord in enumerate(coords):
        for j in xrange(3):
            forces[i, j] = (results[i * len(coord) * 2 + j * 2 + 1] - results[i * len(coord) * 2 + j * 2]) / h
    return forces


def calculate_surface(clf, numbers, coords, atom_idx, max_displacement=.5, steps=25, meta=None):
    '''
    A function that uses plots the value of the clf as a function of `atom_ix`
    displacement.
    '''
    if meta is None:
        meta = [1]

    values = numpy.linspace(-max_displacement, max_displacement, steps)

    results = numpy.zeros((steps, steps))
    for i, x in enumerate(values):
        for j, y in enumerate(values):
            new_coords = coords.copy()
            new_coords[atom_idx, 0] += x
            new_coords[atom_idx, 1] += y
            mat = get_coulomb_matrix(numbers, new_coords)
            mat[mat < 0] = 0
            vector = mat[numpy.tril_indices(mat.shape[0])].tolist() + meta
            results[i, j] = clf.predict(numpy.matrix(vector))[0]

    extent = [-max_displacement,max_displacement,-max_displacement,max_displacement]
    get_matrix_plot(results, extent)
    print results.max(), results.min(), results.std()
    return results


def map_atom(element):
    return [int(x == element) for x in BOND_LENGTHS]


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
            get_histogram_plot(prop_name, prop, units, title="Distribution of %s Values" % prop_name)
        except NameError:
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
    print "\t%d unique geometries" % len(set(paths))
    print
