'''
A collection of functions that create feature vectors that are independent
of the size of the system.

NOTE: This does not mean that these features are small, it just means that
the addition of more atoms to the system does not increase the length of
the feature vector.
'''

import os

import numpy

from sklearn import decomposition

from ..utils import read_file_data
from .utils import homogenize_lengths, get_atom_counts, \
        get_bond_counts, get_angle_counts, get_dihedral_counts, \
        get_trihedral_counts, get_angle_bond_counts, get_dihedral_angle, \
        get_angle_angle, get_bond_length, set_vector_length, \
        construct_zmatrix_addition, get_dihedral_bond_counts, \
        remove_zero_cols, get_fractional_bond_counts, get_encoded_lengths, \
        get_encoded_angles, get_atom_env_counts


def get_full_local_zmatrix_feature(names, paths, **kwargs):
    '''
    A feature vector that uses the idea of a local zmatrix. This expands
    the standard local zmatrix by being applied to every bond within the
    structure. Because of this, this feature vector can be used for any
    molecule unlike the local zmatrix which can only be used for the bond
    based dataset.

    NOTE: This feature vector scales VERY poorly, both in build time and in
    size. Both of these things need to be optimized.
    '''
    vectors = []
    for name, path in zip(names, paths):

        elements, numbers, coords = read_file_data(path)
        _, bonds = get_bond_counts(elements, coords.tolist())
        _, angles = get_angle_counts(elements, coords.tolist(), bonds=bonds)
        _, dihedrals = get_dihedral_counts(elements, coords.tolist(), angles=angles)
        vector = []
        for start in bonds:

            # length must be 2x3
            single_bonds = []
            single_bond_lengths = []
            # length must be 1
            double_bonds = []
            double_bond_lengths = []
            for bond in bonds:
                bond = bond[:2]
                value = get_bond_length(bond, coords)
                if start[0] == bond[1]:
                    bond = bond[1], bond[0]
                elif start[1] == bond[1]:
                    if start[0] not in bond:
                        bond = bond[1], bond[0]

                if start[0] == bond[0] and start[1] == bond[1]:
                    double_bonds.append(bond)
                    double_bond_lengths.append(value)
                elif start[0] == bond[0] or start[1] == bond[0]:
                    single_bonds.append(bond)
                    single_bond_lengths.append(value)

            single_bonds = set_vector_length(single_bonds, 2 * 3, fill=None)
            single_bond_lengths = set_vector_length(single_bond_lengths, 2 * 3)
            double_bonds = set_vector_length(double_bonds, 1, fill=None)
            double_bond_lengths = set_vector_length(double_bond_lengths, 1)

            vector += construct_zmatrix_addition(elements, double_bonds, double_bond_lengths, [0, 1])
            vector += construct_zmatrix_addition(elements, single_bonds, single_bond_lengths, [1])


            # length must be 2x3
            single_angles = []
            single_angle_thetas = []
            # length must be 2x3
            double_angles = []
            double_angle_thetas = []
            for angle in angles:
                value = get_angle_angle(angle, coords)
                middle = angle[1]
                if start[0] == middle:
                    if start[1] in angle:
                        double_angles.append(angle)
                        double_angle_thetas.append(value)
                    else:
                        single_angles.append(angle)
                        single_angle_thetas.append(value)
                elif start[1] == middle:
                    if start[0] in angle:
                        double_angles.append(angle[::-1])
                        double_angle_thetas.append(value)
                    else:
                        single_angles.append(angle)
                        single_angle_thetas.append(value)

            single_angles = set_vector_length(single_angles, 2 * 3, fill=None)
            single_angle_thetas = set_vector_length(single_angle_thetas, 2 * 3)
            double_angles = set_vector_length(double_angles, 2 * 3, fill=None)
            double_angle_thetas = set_vector_length(double_angle_thetas, 2 * 3)

            vector += construct_zmatrix_addition(elements, double_angles, double_angle_thetas, idxs=[2])
            vector += construct_zmatrix_addition(elements, single_angles, single_angle_thetas, idxs=[0, 2])


            # length must be 3x3
            new_dihedrals = []
            new_dihedral_phis = []
            for dihedral in dihedrals:
                middle = dihedral[1:3]
                value = get_dihedral_angle(dihedral, coords)
                if start[0] == middle[0] and start[1] == middle[1]:
                    new_dihedrals.append(dihedral)
                    new_dihedral_phis.append(value)
                elif start[1] == middle[0] and start[0] == middle[1]:
                    new_dihedrals.append(dihedral[::-1])
                    new_dihedral_phis.append(value)

            new_dihedrals = set_vector_length(new_dihedrals, 3 * 3, fill=None)
            new_dihedral_phis = set_vector_length(new_dihedral_phis, 3 * 3)

            vector += construct_zmatrix_addition(elements, new_dihedrals, new_dihedral_phis, idxs=[0, 3])
        vectors.append(vector)
    return homogenize_lengths(vectors)


def get_atom_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of atoms in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        vectors.append(get_atom_counts(elements))
    return remove_zero_cols(vectors)


def get_atom_env_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of atoms in the structure.
    This differs from just counting atoms because it segments based on
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        vectors.append(get_atom_env_counts(elements, coords))
    return remove_zero_cols(vectors)


def get_atom_thermo_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of atoms in the structure.
    '''

    temp = get_atom_feature(names, paths, **kwargs)
    lengths = temp.max(0).tolist()[0]

    vectors = []
    for row in temp:
        temp_vec = []
        for ele_count, max_count in zip(row.tolist()[0], lengths):
            temp_vec += ele_count * [1] + (max_count - ele_count) * [0]
        vectors.append(temp_vec)
    return remove_zero_cols(vectors)


def get_bond_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of bonds in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_bond_counts(elements, coords.tolist())
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_fractional_bond_feature(names, paths, slope=10., **kwargs):
    '''
    This feature vector makes bond type into a continuous function.
    This is done by looking at all the atoms in the molecule and applying
    a sigmoid function based on how close the atoms are to a given bond type.
    These values are then summed together just like the normal bond feature.
    The `slope` parameter defines how step the sigmoid is. The limit as this
    value goes to infinity is the normal bond feature. As it goes to zero, it
    it turns into just a summation of all possible pairwise interactions.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts = get_fractional_bond_counts(elements, coords.tolist(), slope=slope)
        vectors.append(counts)
    return remove_zero_cols(vectors)



def get_encoded_bond_feature(names, paths, segments=10, slope=1., max_depth=0, **kwargs):
    '''
    This is another feature vector attempting to make the bond feature
    continuous. This is done by applying a thermometer-like encoding
    to the distance between any two atoms. These interactions are then
    summed together just like with the bond counting method. This feature
    vector can be seen as creating a `segments` amount of bond types that are
    evenly spaced. The `slope` parameter defines the sharpness of the
    transition.
    Note: Increasing the value of `segments` directly increases the size of
    the resulting feature vector.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts = get_encoded_lengths(elements, coords.tolist(), segments=segments, slope=slope, max_depth=max_depth)
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_encoded_angle_feature(names, paths, segments=10, sigma=1., sigma2=1., **kwargs):
    '''
    This feature vector works in the same sort of way as the encoded bond
    feature. The `segments` parameter sets the granularity of the resulting
    feature vector. `sigma` sets the width of the gaussian used for the angle
    values. `sigma2` is used to apply a decay for when the atoms are far away
    from the center of the angle.
    Note: Increasing the value of `segments` directly increases the size of
    the resulting feature vector.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts = get_encoded_angles(elements, coords, segments=segments, sigma=sigma, sigma2=sigma2)
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_angle_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of angles in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_angle_counts(elements, coords.tolist())
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_angle_bond_feature(names, paths, **kwargs):
    '''
    This feature vector is the same as the angle feature with the addition
    that it includes the type of bond that is added to the so instead of
    just (C, C, C), this feature can take into account things like
    ((C, C, 2), (C, C, 1)) as being separate from ((C, C, 2), (C, C, 2)).
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_angle_bond_counts(elements, coords.tolist())
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_dihedral_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of dihedrals in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_dihedral_counts(elements, coords.tolist())
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_dihedral_bond_feature(names, paths, **kwargs):
    '''
    This feature vector is an expansion in the same way the angle_bond feature
    is. It converts (C, C, C, C) type dihedrals into something like
    ((C, C, 1), (C, C, 2), (C, C, 1)) which is different from
    ((C, C, 2), (C, C, 1), (C, C, 1)).
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_dihedral_bond_counts(elements, coords.tolist())
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_trihedral_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of trihedrals in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_trihedral_counts(elements, coords.tolist())
        vectors.append(counts)
    return remove_zero_cols(vectors)


def get_pca_coulomb_feature(names, paths, dimensions=100, **kwargs):
    '''
    This feature vector takes the feature matrix from get_coulomb_feature and
    does Principal Component Analysis on it to extract the N most influential
    dimensions. The goal of this is to reduce the size of the feature vector
    which can reduce overfitting, and most importantly dramatically reduce
    running time.

    In principal, the number of dimensions used should correspond
    to at least 95% of the variability of the features (This is denoted by the
    `sum(pca.explained_variance_ratio_)`. For a full listing of the influence of
    each dimension look at pca.explained_variance_ratio_.

    This method works by taking the N highest eigenvalues of the matrix (And
    their corresponding eigenvectors) and mapping the feature matrix into
    this new lower dimensional space.
    '''
    feat = get_coulomb_feature(names, paths)
    pca = decomposition.PCA(n_components=dimensions)
    pca.fit(feat)
    # print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    return numpy.matrix(pca.transform(feat))
