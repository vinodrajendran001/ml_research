'''
A collection of functions that create feature vectors that are dependent
of the size of the system. These are things such as distance interactions
between atoms/bonds in the system.

NOTE: The exact scaling of the size is dependent on the function itself. Some
of these functions scale O(n) while others may scale O(n^2) in length.
'''

import os
from itertools import product

import numpy

from sklearn import decomposition

from ..utils import read_file_data
from .utils import get_coulomb_matrix, homogenize_lengths, \
        get_distance_matrix, get_thermometer_encoding, get_eigenvalues, \
        get_connectivity_matrix, get_eq_bond_length
from ..constants import ELE_TO_NUM


def get_connective_feature(names, paths, **kwargs):
    '''
    A simple feature vector based on the connectivity matrix of a molecule.
    This also takes into account the type of the bond between atoms.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_connectivity_matrix(elements, coords)
        mat[mat==''] = 0
        mat[mat=='A'] = 1.5
        mat = mat.astype(int)
        cache[path] = mat[numpy.tril_indices(mat.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_coulomb_feature(names, paths, max_depth=None, **kwargs):
    '''
    This feature vector is based on a distance matrix between all of the atoms
    in the structure with each element multiplied by the number of protons in
    each of atom in the pair. The diagonal is 0.5 * protons ^ 2.4. The
    exponent comes from a fit.
    This is based off the following work:
    M. Rupp, et al. Physical Review Letters, 108(5):058301, 2012.

    The `max_depth` parameter allows specifying a point at which the value is
    set to zero based off the number of bonds between the atoms.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords, max_depth=max_depth)
        cache[path] = mat[numpy.tril_indices(mat.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_coulomb_connect_feature(names, paths, **kwargs):
    '''
    This feature vector is the same thing as the coulomb matrix with the
    alteration that the distances between atoms that are bonded are replaced
    with the equillibrium bond length. This is done to look at what happens if
    those distances are not directly known.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        mat2 = get_connectivity_matrix(elements, coords)

        idxs = numpy.where(mat2 != '')
        for i, j in zip(*idxs):
            length = get_eq_bond_length(elements[i], elements[j], mat2[i, j])
            mat[i, j] = numbers[i] * numbers[j] / length
        cache[path] = mat[numpy.tril_indices(mat.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_bag_of_bonds_connect_feature(names, paths, **kwargs):
    '''
    This is the same thing that was done for just the coulomb feature but mapped
    to the bag of bonds feature.

    NOTE: This feature vector still scales O(N^2).
    '''
    # Add all possible bond pairs (C, C), (C, O)...
    keys = set(tuple(sorted(x)) for x in product(ELE_TO_NUM, ELE_TO_NUM))
    # Add single element types (for the diag)
    keys |= set(ELE_TO_NUM)

    # Sort the keys to remove duplicates later ((C, H) instead of (H, C))
    sorted_keys = sorted(ELE_TO_NUM.keys())

    # Initialize the bags for all the molecules at the same time
    # This is to make it easier to make each of the bags of the same type the
    # same length at the end
    bags = {key: [] for key in keys}
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        # Sort the elements, numbers, and coords based on the element
        bla = sorted(zip(elements, numbers, coords.tolist()), key=lambda x: x[0])
        elements, numbers, coords = zip(*bla)
        coords = numpy.matrix(coords)

        ele_array = numpy.array(elements)
        ele_set = set(elements)

        mat = get_coulomb_matrix(numbers, coords)
        mat2 = get_connectivity_matrix(elements, coords)

        idxs = numpy.where(mat2 != '')
        for i, j in zip(*idxs):
            length = get_eq_bond_length(elements[i], elements[j], mat2[i, j])
            mat[i, j] = numbers[i] * numbers[j] / length
        mat = numpy.array(mat)

        diag = numpy.diagonal(mat)

        for key in keys:
            bags[key].append([])

        for i, ele1 in enumerate(sorted_keys):
            if ele1 not in ele_set:
                continue
            # Select only the rows that are of type ele1
            first = ele_array == ele1
            # Select the diag elements if they match ele1 and store them,
            # highest to lowest
            bags[ele1][-1] = sorted(diag[first].tolist(), reverse=True)
            for j, ele2 in enumerate(sorted_keys):
                if i > j or ele2 not in ele_set:
                    continue
                # Select only the cols that are of type ele2
                second = ele_array == ele2
                # Select only the rows/cols that are in the upper triangle
                # (This could also be the lower), and are in a row, col with
                # ele1 and ele2 respectively
                mask = numpy.triu(numpy.logical_and.outer(first, second), k=1)
                # Add to correct double element bag
                # highest to lowest
                bags[ele1, ele2][-1] = sorted(mat[mask].tolist(), reverse=True)

    # Make all the bags of the same type the same length, and form matrix
    new = [homogenize_lengths(x) for x in bags.values()]
    return numpy.hstack(new)


def get_bag_of_bonds_feature(names, paths, max_depth=None, **kwargs):
    '''
    This feature vector is a reordering of the coulomb matrix so that it does
    not have the same sorts of sorting issuses that the coulomb matrix has.

    The `max_depth` parameter allows specifying a point at which the value is
    set to zero based off the number of bonds between the atoms.

    NOTE: This feature vector still scales O(N^2).
    '''
    # Add all possible bond pairs (C, C), (C, O)...
    keys = set(tuple(sorted(x)) for x in product(ELE_TO_NUM, ELE_TO_NUM))
    # Add single element types (for the diag)
    keys |= set(ELE_TO_NUM)

    # Sort the keys to remove duplicates later ((C, H) instead of (H, C))
    sorted_keys = sorted(ELE_TO_NUM.keys())

    # Initialize the bags for all the molecules at the same time
    # This is to make it easier to make each of the bags of the same type the
    # same length at the end
    bags = {key: [] for key in keys}
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        # Sort the elements, numbers, and coords based on the element
        bla = sorted(zip(elements, numbers, coords.tolist()), key=lambda x: x[0])
        elements, numbers, coords = zip(*bla)
        coords = numpy.matrix(coords)

        ele_array = numpy.array(elements)
        ele_set = set(elements)
        mat = get_coulomb_matrix(numbers, coords, max_depth=max_depth)
        mat = numpy.array(mat)
        diag = numpy.diagonal(mat)

        for key in keys:
            bags[key].append([])

        for i, ele1 in enumerate(sorted_keys):
            if ele1 not in ele_set:
                continue
            # Select only the rows that are of type ele1
            first = ele_array == ele1
            # Select the diag elements if they match ele1 and store them,
            # highest to lowest
            bags[ele1][-1] = sorted(diag[first].tolist(), reverse=True)
            for j, ele2 in enumerate(sorted_keys):
                if i > j or ele2 not in ele_set:
                    continue
                # Select only the cols that are of type ele2
                second = ele_array == ele2
                # Select only the rows/cols that are in the upper triangle
                # (This could also be the lower), and are in a row, col with
                # ele1 and ele2 respectively
                mask = numpy.triu(numpy.logical_and.outer(first, second), k=1)
                # Add to correct double element bag
                # highest to lowest
                bags[ele1, ele2][-1] = sorted(mat[mask].tolist(), reverse=True)

    # Make all the bags of the same type the same length, and form matrix
    new = [homogenize_lengths(x) for x in bags.values()]
    return numpy.hstack(new)


def get_sum_coulomb_feature(names, paths, **kwargs):
    '''
    This feature vector is based off the idea that the cols/rows of the
    coulomb matrix can be summed together to use as features.

    NOTE: This feature vector scales O(N) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        cache[path] = get_coulomb_matrix(numbers, coords).sum(0)

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_bin_coulomb_feature(names, paths, step=1, **kwargs):
    '''
    This is a feature vector based on the coulomb matrix. It adds more data
    by expanding the size of the vector encoding the floating point values as
    a larger set of data. This has the potential to be an infinitely large
    feature vector as step->0.

    This is based off the work in:
    Yunho Jeon and Chong-Ho Choi. IJCNN, (3) 1685-1690, 1999.
    and the recommendation from:
    Gregoire Montavon. On Layer-Wise Representations in Deep Neural Networks.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        cache[path] = mat[numpy.tril_indices(mat.shape[0])]

    vectors = [cache[path] for path in paths]
    return get_thermometer_encoding(homogenize_lengths(vectors), step=step)


def get_bin_eigen_coulomb_feature(names, paths, step=1, **kwargs):
    '''
    This is a feature vector based on the eigenvalues coulomb matrix. It adds
    more data by expanding the size of the vector encoding the floating point
    values as a larger set of data. This has the potential to be an infinitely
    large feature vector as step->0.

    This is based off the work in:
    Yunho Jeon and Chong-Ho Choi. IJCNN, (3) 1685-1690, 1999.
    and the recommendation from:
    Gregoire Montavon. On Layer-Wise Representations in Deep Neural Networks.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        cache[path] = get_eigenvalues(mat)

    vectors = [cache[path] for path in paths]
    return get_thermometer_encoding(homogenize_lengths(vectors), step=step)



def get_distance_feature(names, paths, power=-1, **kwargs):
    '''
    This feature vector is based on a distance matrix between all of the atoms
    in the structure. The value of `power` determines what power each of the
    values in the matrix will be raised to. Leaving it at -1 will result in
    same base as the coulomb matrix just without the inclusion of the Z
    values.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_distance_matrix(coords, power)
        cache[path] = mat[numpy.tril_indices(mat.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_custom_distance_feature(names, paths, f=None, **kwargs):
    '''
    This allows the insertion of custom functions for how the distance
    interactions should be handled. If no function is passed into `f` it will
    default to using 1/(exp(-r) + r).

    All of the functions used should take a single argument which will be a
    numpy array with the (i,j) pairwise distances (The diagonal will be all
    zeros).

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
    if f is None:
        f = lambda r: 1/(numpy.exp(-r) + r)

    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_distance_matrix(coords, 1)
        temp = f(mat)
        cache[path] = temp[numpy.tril_indices(temp.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)

def get_sorted_coulomb_feature(names, paths, **kwargs):
    '''
    This feature vector is based on a distance matrix between all of the atoms
    in the structure with each element multiplied by the number of protons in
    each of atom in the pair. The diagonal is 0.5 * protons ^ 2.4. The
    exponent comes from a fit.
    This is based off the following work:
    M. Rupp, et al. Physical Review Letters, 108(5):058301, 2012.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        order = numpy.linalg.norm(mat,axis=0).argsort()[::-1]
        temp = (mat[:,order])[order,:]
        cache[path] = temp[numpy.tril_indices(temp.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_sorted_coulomb_vector_feature(names, paths, **kwargs):
    '''
    This feature vector is based on a distance matrix between all of the atoms
    in the structure with each element multiplied by the number of protons in
    each of atom in the pair. The diagonal is 0.5 * protons ^ 2.4. The
    exponent comes from a fit.
    This is based off the following work:
    M. Rupp, et al. Physical Review Letters, 108(5):058301, 2012.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
    feat = get_coulomb_feature(names, paths)
    feat.sort()
    return feat[:,::-1]


def get_eigen_coulomb_feature(names, paths, **kwargs):
    '''
    This feature vector is from the eigenvalues of the coulomb matrix.
    The eigenvalues are sorted so that the largest values come first.

    NOTE: This feature vector scales O(N) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        cache[path] = get_eigenvalues(mat)

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_eigen_distance_feature(names, paths, power=-1, **kwargs):
    '''
    This feature vector is from the eigenvalues of the distance matrix. The
    eigenvalues are sorted so that the largest values come first. The `power`
    parameter defines the power that the distance matrix should be raised to
    before getting the eigenvalues.

    NOTE: This feature vector scales O(N) where N is the number of atoms in
    largest structure.
    '''
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_distance_matrix(coords, power)
        cache[path] = get_eigenvalues(mat)

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_eigen_custom_distance_feature(names, paths, f=None, **kwargs):
    '''
    This is the same as the custom_distance_feature except it returns the
    eigenvalues of the matrices. If no function is passed into `f` it will
    default to using 1/(exp(-r) + r).

    All of the functions used should take a single argument which will be a
    numpy array with the (i,j) pairwise distances (The diagonal will be all
    zeros).

    NOTE: This feature vector scales O(N) where N is the number of atoms in
    largest structure.
    '''
    if f is None:
        f = lambda r: 1/(numpy.exp(-r) + r)
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_distance_matrix(coords, 1)
        temp = f(mat)
        cache[path] = get_eigenvalues(temp)

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)
