'''
A collection of functions that produce feature vectors that may or may not be
of a constant length, but they should be done at the per atom/bond level. Thus
resulting in feature matricies that have samples on the order of the number of
atoms/bonds in the system instead of per molecule.
'''

import numpy

from ..utils import read_file_data, map_atom
from .utils import get_bond_counts, get_all_bond_types, get_type_data


def get_local_atom_connections(names, paths, **kwargs):
    '''
    This feature vector returns the types of bonds that are

    v = [Z_atom, bond_count(Z_atom), mol_idx]
    '''
    # [atom_map, bond_map_sum, molecule_idx]
    vectors = []

    types = get_all_bond_types()
    typemap, counts_base = get_type_data(types)

    for i, path in enumerate(paths):
        molecule = []
        elements, numbers, coords = read_file_data(path)

        for element in elements:
            molecule.append((map_atom(element), counts_base[:], [i]))

        _, bonds = get_bond_counts(elements, coords.tolist())
        for bond in bonds:
            bond_key = elements[bond[0]], elements[bond[1]], bond[2]
            type_idx = typemap[bond_key]
            for idx in bond[:2]:
                molecule[idx][1][type_idx] += 1

        for row in molecule:
            vectors.append(sum(row, []))
    return numpy.matrix(vectors)


def get_local_atom_zmatrix(names, paths, **kwargs):
    '''
    The local atom-wise zmatrix.

    v = [Z_atom] +
        [(Z_bond_1, r_1), ..., (Z_bond_n, r_m)] +
        [(Z_angle_1^1, Z_angle_1^2, theta_1), ..., (Z_angle_m^1, Z_angle_m^2, theta_m)] +
        [(Z_dihedral_1^1, Z_dihedral_1^2, Z_dihedral_1^3, phi_1), ..., (Z_dihedral_p^1, Z_dihedral_p^2, Z_dihedral_p^3, phi_p)]
    m =  4
    n =  6  # 3 + 2 + 1
    p = 36  # 4 * 3 * 3
    '''
    pass


def get_local_bond_zmatrix(names, paths, **kwargs):
    '''
    The local bond-wise zmatrix.

    v = [(Z_atom_1, Z_atom_2, r)] +
        [(Z_dihedral_1^1, Z_dihedral_1^2, phi_1), ..., (Z_dihedral_p^1, Z_dihedral_p^2, phi_1)]

    for i in sortZ([1, 2]):
        v += [(Z_bond_1^i, r_1^i), ..., (Z_bond_m^i, r_m^i)]
        v += [(Z_angle_1^i, theta_1^i), ..., (Z_angle_n^i, theta_1^i)]
    m = 3
    n = 3
    p = 9  # 3 * 3
    '''
    pass
