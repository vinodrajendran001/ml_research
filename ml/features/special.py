'''
This is a collection of functions that have special requirements to run. These
often involve external data, or they are otherwise akward to use with the
other feature vectors.
'''

import numpy

from sklearn import decomposition

from ..utils import read_file_data
from .utils import get_coulomb_matrix, homogenize_lengths, \
        get_bond_counts, get_angle_counts, get_dihedral_counts, \
        get_dihedral_angle, get_angle_angle, \
        set_vector_length, construct_zmatrix_addition


def get_local_zmatrix(names, paths, **kwargs):
    '''
    A feature vector that uses the idea of a local zmatrix.

    This feature vector is initialized from a pair of atoms. The exact
    pair is based off the name. The name should be made of two ints that
    are comma separated indicating the index of the atoms to use.

    From there, all of the atoms that are bonded to these two atoms are
    collected and added to the feature vector based on their distance.
    Then the all of the angles with either 1) one of the starting two
    atoms as the center of angle are included, or 2) both of the atoms
    are in the angle. Then all of the dihedrals where both of the starts
    are included AND are the middle two points in the dihedral.
    '''
    vectors = []
    for name, path in zip(names, paths):
        start = [int(x) for x in name.split(',')][:2]
        elements, numbers, coords = read_file_data(path)
        vector = []

        _, bonds = get_bond_counts(elements, coords.tolist())
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


        _, angles = get_angle_counts(elements, coords.tolist())
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


        _, dihedrals = get_dihedral_counts(elements, coords.tolist(), angles=angles)
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
    return vectors


def get_random_coulomb_feature(names, paths, size=1, **kwargs):
    '''
    This is the same as the coulomb matrix feature with the addition that it
    adds additional randomly permuted coulomb matricies based on the value of
    `size`.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    NOTE: The number of feature vectors this returns is len(names) * size.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)
        vectors.append(mat)

    for mat in vectors[:]:
        shape = mat.shape
        for x in xrange(size - 1):
            order = numpy.arange(shape[0])
            out = numpy.random.permutation(order)
            perm = numpy.zeros(shape)
            perm[order, out] = 1
            vectors.append(perm.T * mat * perm)
    vectors = [mat[numpy.tril_indices(mat.shape[0])] for mat in vectors]
    return homogenize_lengths(vectors)


def get_fingerprint_feature(names, paths, size=2048, **kwargs):
    '''
    This feature vector is constructed from a chemical fingerprint algorithm.
    Basically, this ends up being a boolean vector of whether or not different
    structural features occur within the molecule. These could be any sort of
    bonding chain or atom pairing. The specifics of the fingerprinting can be
    found here.
    http://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
    '''
    try:
        from rdkit import Chem
        from rdkit.Chem.Fingerprints import FingerprintMols
    except ImportError:
        print "Please install RDkit."
        return numpy.matrix([[] for path in paths])

    vectors = []
    for path in paths:
        path = path.replace("out", "mol2")
        m = Chem.MolFromMol2File(path, sanitize=False)
        f = FingerprintMols.FingerprintMol(m, fpSize=size, minPath=1,
                                        maxPath=7, bitsPerHash=2, useHs=True,
                                        tgtDensity=0, minSize=size)
        vectors.append([x == '1' for x in f.ToBitString()])
    return numpy.matrix(vectors)

def get_mul_feature(names, paths):
    vectors = []
    for path, name in zip(paths, names):
        elements, numbers, coords = read_file_data(path)
        charges = []
        with open("mul/%s.mul" % name, "r") as f:
            for i, line in enumerate(f):
                ele, val = line.strip().split()
                charges.append(numbers[i] + float(val))
        mat = get_coulomb_matrix(numbers, coords)
        vectors.append(mat[numpy.tril_indices(mat.shape[0])])
    return homogenize_lengths(vectors)
