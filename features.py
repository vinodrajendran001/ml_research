import os

import numpy

from sklearn import decomposition

from utils import tokenize, \
        decay_function, gauss_decay_function, read_file_data, \
        get_coulomb_matrix, homogenize_lengths, get_distance_matrix, \
        get_thermometer_encoding, get_eigenvalues, get_atom_counts, \
        get_bond_counts, get_angle_counts, get_dihedral_counts, get_trihedral_counts, \
        get_angle_bond_counts, get_dihedral_angle, get_angle_angle, get_bond_length, \
        map_atom, get_connectivity_matrix, set_vector_length, construct_zmatrix_addition, \
        get_all_bond_types, get_type_data, get_dihedral_bond_counts, remove_zero_cols, \
        get_fractional_bond_counts, get_encoded_lengths, get_encoded_angles
from constants import ARYL, RGROUPS


# Example Feature function
def get_null_feature(names, paths, **kwargs):
    '''
    `names`: a list of strings with the name of the structure (['4aa'])
    `paths`: a list of locations of the geometry files for that structures
        (['data/noopt/geoms/4aa'])
    This function returns a matrix of feature vectors (N_names, N_features).

    There is no need to add a bias term or try to split the structures based on
    which data set they came from, both of these will be handled as the data is
    loaded.
    '''
    return numpy.matrix(numpy.zeros((len(names), 0)))


def get_local_atom_zmatrix(names, paths, **kwargs):
    '''
    pass
    '''
    # [atom_map, bond_map_sum, molecule_idx]
    vectors = []

    types = get_all_bond_types()
    typemap, counts_base = get_type_data(types)

    for i, (name, path) in enumerate(zip(names, paths)):
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


def get_atom_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of atoms in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        vectors.append(get_atom_counts(elements))
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



def get_encoded_bond_feature(names, paths, segments=10, slope=1., **kwargs):
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
        counts = get_encoded_lengths(elements, coords.tolist(), segments=segments, slope=slope)
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


def get_binary_feature(names, paths, limit=4, **kwargs):
    '''
    Creates a simple boolean feature vector based on whether or not a part is
    in the name of the structure.
    NOTE: This feature vector size scales O(N), where N is the limit.
    NOTE: Any parts of the name larger than the limit will be stripped off.

    >>> get_binary_feature(['4aa'], ['path/'], limit=1)
    matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    >>> get_binary_feature(['3'], ['path/'], limit=1)
    matrix([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    >>> get_binary_feature(['4aa4aa'], ['path/'], limit=1)
    matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    >>> get_binary_feature(['4aa4aa'], ['path/'], limit=2)
    matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)

    vectors = []
    for name in names:
        features = []
        name = name.replace('-', '')  # no support for flipping yet
        count = 0
        for token in tokenize(name):
            base = second
            if token in first:
                if count == limit:
                    break
                count += 1
                base = first
            temp = [0] * len(base)
            temp[base.index(token)] = 1
            features.extend(temp)

        # fill features to limit amount of groups
        features += [0] * length * (limit - count)
        vectors.append(features)
    return numpy.matrix(vectors)


def get_flip_binary_feature(names, paths, limit=4, **kwargs):
    '''
    This creates a feature vector that is the same as the normal binary one
    with the addition of an additional element for each triplet to account
    for if the aryl group is flipped.
    NOTE: This feature vector size scales O(N), where N is the limit.
    NOTE: Any parts of the name larger than the limit will be stripped off.
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)

    vectors = []
    for name in names:
        features = []
        count = 0
        flips = []
        for token in tokenize(name):
            if token == '-':
                flips[-1] = 1
                continue

            base = second
            if token in first:
                if count == limit:
                    break
                count += 1
                flips.append(0)
                base = first
            temp = [0] * len(base)
            temp[base.index(token)] = 1
            features.extend(temp)

        # fill features to limit amount of groups
        features += [0] * length * (limit - count)
        flips += [0] * (limit - count)
        vectors.append(features + flips)

    return numpy.matrix(vectors)


def get_decay_feature(names, paths, power=1, H=1, factor=1, **kwargs):
    '''
    This feature vector works about the same as the binary feature vector
    with the exception that it does not have O(N) scaling as the length of
    the molecule gains more rings. This is because it treats the
    interaction between rings as some decay as they move further from the
    "start" of the structure (the start of the name).
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:

        name = name.replace('-', '')  # no support for flipping yet
        end = tokenize(name)
        temp = [0] * length
        for i, char in enumerate(end):
            # Use i / 3 because the tokens come in sets of 3 (Aryl, R1, R2)
            # Use i % 3 to get which part it is in the set (Aryl, R1, R2)
            count, part = divmod(i, 3)

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # Needs to be optimized for power, H, and factor
            # count + 1 is used so that the first value will be 1, and
            # subsequent values will have their respective scaling.
            temp[idx] += decay_function(count + 1, power, H, factor)
        vectors.append(temp)
    return numpy.matrix(vectors)


def get_gauss_decay_feature(names, paths, sigma=2, **kwargs):
    '''
    This feature vector works the exact same as the normal decay feature
    vector with the exception that it uses a Gaussian distribution for the
    decay. This was picked because looking at the PCA components for the
    parts of the structure and their relative influence as they were farther
    in the name from the start in the binary feature vector.
    In the future, this might need to be a per component decay.

    NOTE: The sigma value is kind of arbitrary. With a little bit of tuning
    sigma=2 produced a reasonably low error. (From the PCA, the expected
    value was sigma=6)
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:

        name = name.replace('-', '')  # no support for flipping yet
        end = tokenize(name)
        temp = [0] * length
        for i, char in enumerate(end):
            # Use i / 3 because the tokens come in sets of 3 (Aryl, R1, R2)
            # Use i % 3 to get which part it is in the set (Aryl, R1, R2)
            count, part = divmod(i, 3)

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # This starts from 0 and goes out unlike the other decay function.
            temp[idx] += gauss_decay_function(count, sigma)
        vectors.append(temp)
    return numpy.matrix(vectors)


def get_centered_decay_feature(names, paths, power=1, H=1, factor=1, **kwargs):
    '''
    This feature vector takes the same approach as the decay feature vector
    with the addition that it does the decay from the center of the structure.
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:

        name = name.replace('-', '')  # no support for flipping yet

        end = tokenize(name)
        partfeatures = [0] * length

        # Get the center index (x / 3 is to account for the triplet sets)
        # The x - 0.5 is to offset the value between index values.
        center = len(end) / 3. / 2. - 0.5
        for i, char in enumerate(end):
            # abs(x) is used to not make it not differentiate which
            # direction each half of the structure is going relative to
            # the center
            count = abs((i / 3) - center)
            part = i % 3

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # Needs to be optimized for power, H, and factor
            partfeatures[idx] += decay_function(count + 1, power, H, factor)
        vectors.append(partfeatures)
    return numpy.matrix(vectors)


def get_signed_centered_decay_feature(names, paths, power=1, H=1, factor=1,
                                                                    **kwargs):
    '''
    This feature vector works the same as the centered decay feature vector
    with the addition that it takes into account the side of the center that
    the rings are on instead of just looking at the magnitude of the distance.
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:
        name = name.replace('-', '')  # no support for flipping yet

        end = tokenize(name)
        # One set is for the left (negative) side and the other is for the
        # right side.
        partfeatures = [[0] * length, [0] * length]

        # Get the center index (x / 3 is to account for the triplet sets)
        # The x - 0.5 is to offset the value between index values.
        center = len(end) / 3. / 2. - 0.5
        for i, char in enumerate(end):
            # abs(x) is used to not make it not differentiate which
            # direction each half of the structure is going relative to
            # the center
            count = (i / 3) - center
            # This is used as a switch to pick the correct side
            is_negative = count < 0
            count = abs(count)
            part = i % 3

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # Needs to be optimized for power, H, and factor
            partfeatures[is_negative][idx] += decay_function(count + 1, power,
                                                            H, factor)
        vectors.append(partfeatures[0] + partfeatures[1])
    return numpy.matrix(vectors)


def get_coulomb_feature(names, paths, **kwargs):
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
        cache[path] = mat[numpy.tril_indices(mat.shape[0])]

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_bag_of_bonds_feature(names, paths, **kwargs):
    from itertools import product
    from constants import ELE_TO_NUM

    keys = set(tuple(sorted(x)) for x in product(ELE_TO_NUM, ELE_TO_NUM))

    bags = {key: [] for key in keys}
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        mat = get_coulomb_matrix(numbers, coords)

        for key in keys:
            bags[key].append([])

        for i, ele1 in enumerate(elements):
            for j, ele2 in enumerate(elements):
                if i >= j:
                    continue
                if ele1 < ele2:
                    bags[ele1, ele2][-1].append(mat[i, j])
                else:
                    bags[ele2, ele1][-1].append(mat[i, j])

        for key, vectors in bags.items():
            for vector in vectors:
                vector.sort(reverse=True)

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
