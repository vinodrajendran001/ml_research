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
