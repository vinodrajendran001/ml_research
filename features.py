import os

import numpy

from sklearn import decomposition

from utils import tokenize, ARYL, RGROUPS, \
        decay_function, gauss_decay_function, read_file_data, \
        get_coulomb_matrix, homogenize_lengths, get_distance_matrix, \
        get_thermometer_encoding, get_eigenvalues, get_atom_counts, \
        get_bond_counts, get_angle_counts, get_dihedral_counts, get_trihedral_counts, \
        get_angle_bond_counts, get_dihedral_angle, get_angle_angle, get_bond_length, \
        map_atom



# Example Feature function
def get_null_feature(names, paths, **kwargs):
    '''
    names is a list of strings with the name of the structure (['4aa'])
    paths is a list of locations of the geometry files for that structures
        (['data/noopt/geoms/4aa'])
    This function returns a matrix of feature vectors (N_names, N_features).

    There is no need to add a bias term or try to split the structures based on
    which data set they came from, both of these will be handled as the data is
    loaded.
    '''
    return numpy.matrix(numpy.zeros((len(names), 0)))


def get_local_zmatrix(names, paths, **kwargs):
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


        if len(single_bonds) < 2 * 3:
            for i in xrange(2 * 3 - len(single_bonds)):
                single_bonds.append(None)
                single_bond_lengths.append(0.0)
        assert len(single_bonds) == 2 * 3
        if len(double_bonds) < 1:
            double_bonds.append(None)
            double_bond_lengths.append(0.0)
        assert len(double_bonds) == 1


        for length, bond in zip(double_bond_lengths, double_bonds):
            if bond is not None:
                vector += map_atom(elements[bond[0]]) + map_atom(elements[bond[1]]) + [length]
            else:
                vector += map_atom(None) + map_atom(None) + [length]

        for length, bond in zip(single_bond_lengths, single_bonds):
            if bond is not None:
                vector += map_atom(elements[bond[1]]) + [length]
            else:
                vector += map_atom(None) + [length]



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

        if len(single_angles) < 2 * 3:
            for i in xrange(2 * 3 - len(single_angles)):
                single_angles.append(None)
                single_angle_thetas.append(0.0)
        assert len(single_angles) == 2 * 3
        if len(double_angles) < 2 * 3:
            for i in xrange(2 * 3 - len(double_angles)):
                double_angles.append(None)
                double_angle_thetas.append(0.0)
        assert len(double_angles) == 2 * 3
        vector += single_angle_thetas + double_angle_thetas


        for theta, angle in zip(double_angle_thetas, double_angles):
            if angle is not None:
                vector += map_atom(elements[angle[2]]) + [theta]
            else:
                vector += map_atom(None) + [theta]

        for theta, angle in zip(single_angle_thetas, single_angles):
            if angle is not None:
                vector += map_atom(elements[angle[0]]) + map_atom(elements[angle[2]]) + [theta]
            else:
                vector += map_atom(None) + map_atom(None) + [theta]


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

        if len(new_dihedrals) < 3 * 3:
            for i in xrange(3 * 3 - len(new_dihedrals)):
                new_dihedrals.append(None)
                new_dihedral_phis.append(0.0)
        assert len(new_dihedrals) == 3 * 3


        for phi, dihedral in zip(new_dihedral_phis, new_dihedrals):
            if dihedral is not None:
                vector += map_atom(elements[dihedral[0]]) + map_atom(elements[dihedral[3]]) + [phi]
            else:
                vector += map_atom(None) + map_atom(None) + [phi]

        vectors.append(vector)
    return vectors


def get_atom_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of atoms in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        vectors.append(get_atom_counts(elements))
    return numpy.matrix(vectors)


def get_bond_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of bonds in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_bond_counts(elements, coords.tolist())
        vectors.append(counts)
    return numpy.matrix(vectors)


def get_angle_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of angles in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_angle_counts(elements, coords.tolist())
        vectors.append(counts)
    return numpy.matrix(vectors)


def get_dihedral_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of dihedrals in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_dihedral_counts(elements, coords.tolist())
        vectors.append(counts)
    return numpy.matrix(vectors)


def get_trihedral_feature(names, paths, **kwargs):
    '''
    A feature vector based entirely off the number of dihedrals in the structure.
    '''
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        counts, _ = get_trihedral_counts(elements, coords.tolist())
        vectors.append(counts)
    return numpy.matrix(vectors)


def get_atom_and_bond_feature(names, paths, **kwargs):
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        a_counts = get_atom_counts(elements, coords.tolist())
        b_counts, bonds = get_bond_counts(elements, coords.tolist())
        vectors.append(a_counts + b_counts)
    return numpy.matrix(vectors)


def get_atom_bond_and_angle_feature(names, paths, **kwargs):
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        a_counts = get_atom_counts(elements, coords.tolist())
        b_counts, bonds = get_bond_counts(elements, coords.tolist())
        an_counts, angles = get_angle_counts(elements, coords.tolist(), bonds=bonds)
        vectors.append(a_counts + b_counts + an_counts)
    return numpy.matrix(vectors)


def get_atom_bond_angle_and_dihedral_feature(names, paths, **kwargs):
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        a_counts = get_atom_counts(elements, coords.tolist())
        b_counts, bonds = get_bond_counts(elements, coords.tolist())
        an_counts, angles = get_angle_counts(elements, coords.tolist(), bonds=bonds)
        d_counts, _ = get_dihedral_counts(elements, coords.tolist(), angles=angles)
        vectors.append(a_counts + b_counts + an_counts + d_counts)
    return numpy.matrix(vectors)


def get_atom_bond_angle_dihedral_and_trihedral_feature(names, paths, **kwargs):
    vectors = []
    for path in paths:
        elements, numbers, coords = read_file_data(path)
        a_counts = get_atom_counts(elements, coords.tolist())
        b_counts, bonds = get_bond_counts(elements, coords.tolist())
        an_counts, angles = get_angle_counts(elements, coords.tolist(), bonds=bonds)
        d_counts, dihedrals = get_dihedral_counts(elements, coords.tolist(), angles=angles)
        t_counts, _ = get_trihedral_counts(elements, coords.tolist(), dihedrals=dihedrals)
        vectors.append(a_counts + b_counts + an_counts + d_counts + t_counts)
    return numpy.matrix(vectors)



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


def get_bin_coulomb_feature(names, paths, step=1, **kwargs):
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_distance_matrix(coords, -1)
        temp = mat[numpy.tril_indices(mat.shape[0])]
        cache[path] = get_thermometer_encoding(temp, step=step)

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)


def get_bin_eigen_coulomb_feature(names, paths, step=1, **kwargs):
    cache = {}
    for path in paths:
        if path in cache:
            continue
        elements, numbers, coords = read_file_data(path)
        mat = get_distance_matrix(coords, -1)
        eigvals = get_eigenvalues(mat)
        cache[path] = get_thermometer_encoding(eigvals, step=step)

    vectors = [cache[path] for path in paths]
    return homogenize_lengths(vectors)



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
        f = lambda mat: 1/(numpy.exp(-r) + r)

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
        f = lambda mat: 1/(numpy.exp(-r) + r)
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
