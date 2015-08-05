from itertools import product

from scipy.spatial.distance import cdist
from scipy.special import expit
import scipy.stats
import numpy
from numpy.linalg import norm

from ..constants import ELE_TO_NUM, BOND_LENGTHS, TYPE_ORDER, BHOR_TO_ANGSTROM
from ..utils import gauss_decay_function


def get_coulomb_matrix(numbers, coords):
    """
    Return the coulomb matrix for the given `coords` and `numbers`
    """
    ANGSTROM_TO_BHOR = 1. / BHOR_TO_ANGSTROM
    top = numpy.outer(numbers, numbers).astype(numpy.float64)
    r = get_distance_matrix(ANGSTROM_TO_BHOR * coords, power=1)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.divide(top, r, top)
    numpy.fill_diagonal(top, 0.5 * numpy.array(numbers) ** 2.4)
    top[top == numpy.Infinity] = 0
    top[numpy.isnan(top)] = 0
    return top


def invert_distance_matrix(D, dim=3):
    '''
    This function converts a distance matrix back into a set of coordinates
    of dimension `dim`. It does this by a bit of magic and a bit of PCA.

    NOTE: The distance matrix going in must be squared for this to work
    correctly.

    This implementation is based off the discussion in:
    http://programmers.stackexchange.com/questions/243199/find-points-whose-pairwise-distances-approximate-a-given-distance-matrix
    '''
    n = D.shape[0]
    J = numpy.matrix(numpy.eye(n) - 1./n)
    B = -.5 * J * D * J
    L, U = numpy.linalg.eig(B)
    idx = L.argsort()
    L = L[idx]
    U = numpy.matrix(U[:,idx])
    X = numpy.matrix(numpy.diag(L[-dim:] **.5)) * U.T[-dim:,:]
    X[numpy.isnan(X)] = 0.
    return X.T


def invert_coulomb_matrix(M):
    diag = numpy.diag(M)
    sel = numpy.nonzero(diag)[0].max() + 1
    base = M[:sel,:sel]
    base_diag = numpy.diag(base)
    ele_nums = (2 * base_diag) ** (1/2.4)
    D = numpy.outer(ele_nums, ele_nums) / base
    numpy.fill_diagonal(D, 0)
    return ele_nums, invert_distance_matrix(D ** 2)


def get_distance_matrix(coords, power=-1, inf_val=1):
    dist = cdist(coords, coords)
    with numpy.errstate(divide='ignore'):
        numpy.power(dist, power, dist)
    dist[dist == numpy.Infinity] = inf_val
    return dist


def homogenize_lengths(vectors):
    n = max(len(x) for x in vectors)
    feat = numpy.zeros((len(vectors), n))
    for i, x in enumerate(vectors):
        feat[i, 0:len(x)] = x
    return numpy.matrix(feat)


def remove_zero_cols(vectors):
    temp = numpy.array(vectors)
    sums = temp.sum(0)
    return temp[:,sums>0]


def get_thermometer_encoding(X, step=1):
    '''
    This is another method of encoding floating point values so that they work
    better with neural nets.

    This is based off the work in:
    Yunho Jeon and Chong-Ho Choi. IJCNN, (3) 1685-1690, 1999.
    and the recommendation from:
    Gregoire Montavon. On Layer-Wise Representations in Deep Neural Networks.
    '''
    X = numpy.array(X)
    max_vals = X.max(0).flatten()
    # This splitting based on the size is to optimize the speed/space
    # trade off. The actual splitting amount just comes from some trial and
    # error for what worked the best.
    if X.shape[1] > X.shape[0]:
        b = numpy.arange(0, max_vals.max() + step, step)
        c = max_vals + step
        Xexp = numpy.subtract.outer(X, b)[:, numpy.greater.outer(c, b)]
        numpy.divide(Xexp, step, Xexp)
        numpy.tanh(Xexp, Xexp)
        return Xexp
    else:
        Xexp = []
        for i in xrange(X.shape[1]):
            b = numpy.arange(0, max_vals[i] + step, step)
            temp = numpy.subtract.outer(X[:,i], b)
            Xexp.append(temp)
        Xexp = numpy.concatenate(Xexp, 1)
        numpy.divide(Xexp, step, Xexp)
        numpy.tanh(Xexp, Xexp)
        return Xexp


def get_eigenvalues(X):
    '''
    This returns the eigenvalues of a matrix in descending order.
    '''
    eigvals = numpy.linalg.eigvals(X)
    eigvals.sort()
    return numpy.real(eigvals[::-1])


def get_connectivity_matrix(elements, coords):
    '''
    Returns a connectivity matrix where the bond type is designed as
    ['', '1', 'Ar', '2', '3']. This is often just used as an intermidate
    calculation when dealing with connections.
    '''
    r = get_distance_matrix(coords, power=1)
    results = numpy.empty(r.shape, dtype=str)
    for bond_type in TYPE_ORDER:
        a = [BOND_LENGTHS[ele].get(bond_type, -1000) for ele in elements]
        limits = numpy.add.outer(a, a)
        results[r < limits] = bond_type
    return results


def get_all_length_types(base_types=BOND_LENGTHS.keys(), length=2):
    return list(product(base_types, repeat=length))


def get_all_bond_types():
    types = []
    for x, y in get_all_length_types():
        for bond_type in TYPE_ORDER:
            if bond_type in BOND_LENGTHS[x] and bond_type in BOND_LENGTHS[y]:
                types.append((x, y, bond_type))
    return types


def get_type_data(types):
    typemap = dict(zip(types, xrange(len(types))))
    counts = [0 for x in types]
    return typemap, counts


def get_atom_counts(elements, coords=None):
    types = sorted(BOND_LENGTHS.keys())
    typemap, counts = get_type_data(types)
    for ele in elements:
        counts[typemap[ele]] += 1
    return counts


def get_atom_env_counts(elements, coords=None):
    '''
    Return the atom env counts. This differs from just atom counts in that it
    takes into account the number of things that are connected to the atom. So
    instead of just a 'C' it could be a 'C1' or a 'C2' (being carbons attached
    to 1 thing or 2 things.)
    '''
    types = sorted(BOND_LENGTHS.keys())
    types2 = list(product(types, range(1,5)))
    typemap, counts = get_type_data(types2)

    mat = get_connectivity_matrix(elements, coords)
    mat = (mat != '').astype(int)
    sums = mat.sum(0).tolist()
    for i, ele in enumerate(elements):
        bal = sums[i] - 1
        bla = typemap[(ele, bal)]
        counts[bla] += 1
    return counts


def get_bond_type(element1, element2, dist):
    for key in TYPE_ORDER[::-1]:
        try:
            if dist < (BOND_LENGTHS[element1][key] + BOND_LENGTHS[element2][key]):
                return key
        except KeyError:
            continue


def get_bond_counts(elements, coords):
    '''
    Return the bond type counts and a list of all the bonds in the molecule.
    '''
    types = get_all_bond_types()
    typemap, counts = get_type_data(types)

    bonds = []
    dist_mat = get_distance_matrix(coords, power=1)
    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i + 1:]):
            j += i + 1
            dist = dist_mat[i, j]
            bond_type = get_bond_type(element1, element2, dist)
            if bond_type:
                if element1 > element2:
                    # Flip if they are not in alphabetical order
                    bonds.append((j, i, bond_type))
                    counts[typemap[element2, element1, bond_type]] += 1
                else:
                    bonds.append((i, j, bond_type))
                    counts[typemap[element1, element2, bond_type]] += 1
    return counts, bonds


def get_sum_bond_counts(elements, coords):
    types = get_all_bond_types()
    typemap, counts = get_type_data(types)

    bonds = []
    dist_mat = get_distance_matrix(coords, power=1)
    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i + 1:]):
            j += i + 1
            dist = dist_mat[i, j]
            bond_type = get_bond_type(element1, element2, dist)
            if bond_type:
                for t in TYPE_ORDER:
                    if element1 > element2:
                        # Flip if they are not in alphabetical order
                        bonds.append((j, i, bond_type))
                        counts[typemap[element2, element1, bond_type]] += 1
                    else:
                        bonds.append((i, j, bond_type))
                        counts[typemap[element1, element2, bond_type]] += 1
                    if bond_type == t:
                        break
    return counts, bonds


def get_fractional_bond_types(element1, element2, dist, slope=10.):
    values = []
    for key in TYPE_ORDER:
        try:
            a = BOND_LENGTHS[element1][key]
            b = BOND_LENGTHS[element2][key]
            r = (a + b) - dist
            values.append(sigmoid(slope * r))
        except KeyError:
            return values
    return values


def get_fractional_bond_counts(elements, coords, slope=10.):
    types = get_all_bond_types()
    typemap, counts = get_type_data(types)

    distances = get_distance_matrix(coords, power=1)
    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i + 1:]):
            j += i + 1
            r = distances[i, j]
            values = get_fractional_bond_types(element1, element2, r, slope=slope)
            for bond_type, value in zip(TYPE_ORDER, values):
                if element1 > element2:
                    # Flip if they are not in alphabetical order
                    counts[typemap[element2, element1, bond_type]] += value
                else:
                    counts[typemap[element1, element2, bond_type]] += value
    return counts


def get_depth_threshold_mask(mat, max_depth=1):
    '''
    Given a connectivity matrix (either strings or ints), return a mask that is
    True at [i,j] if there exists a path from i to j that is of length
    `max_depth` or fewer.

    This is done by repeated matrix multiplication of the connectivity matrix.

    If `max_depth` is less than 1, this will return all True matrix.
    '''
    if max_depth < 1:
        temp = numpy.ones(mat.shape).astype(bool)
        return numpy.matrix(temp)

    mask = (mat != '')
    if isinstance(mask, bool):
        mask = mat == 1

    d = numpy.matrix(mask).astype(int)
    acc = d.copy()
    for i in xrange(2, max_depth + 1):
        acc *= d
        mask |= (acc == 1)
    return mask


def get_encoded_lengths(elements, coords, segments=10, start=0.2, end=6., slope=1, max_depth=1, sigmoid="expit"):
    '''
    This is a function for encoding bond distances into a reasonable form. By
    reasonable, it means a form that is continous and of constant size relative
    to the whole molecule.

    `segments` defines the number of sigmoid functions to use. The interval they
    are spaced at is numpy.linspace(start, end, segments).

    `max_depth` defines the longest distance these bonds will be encoded for, as
    measured by the number of bonds between them.

    There are three possible kinds of sigmoids for this function. A normal CDF,
    zero/one, and the standard expit sigmoid. These can be selected with the
    `sgimoid` parameter as "norm_cdf", "zero_one", or "expit" respectively.

    '''
    sigmoid_options = {
        "norm_cdf": scipy.stats.norm.cdf,
        "zero_one": lambda x: (x > 0.) * 1.,
        "expit": expit,
        "tanh": lambda x: (numpy.tanh(x)+1) / 2,
    }
    sigmoid = sigmoid_options[sigmoid]

    ele_idx = {ele: i for i, ele in enumerate(ELE_TO_NUM)}
    vector = numpy.zeros((len(ELE_TO_NUM), len(ELE_TO_NUM), segments))
    theta = numpy.linspace(start, end, segments)

    distances = get_distance_matrix(coords, power=1)
    bonds = get_connectivity_matrix(elements, coords)
    bonds = get_depth_threshold_mask(bonds, max_depth=max_depth)

    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i + 1:]):
            j += i + 1
            if not bonds[i,j]: continue

            value = sigmoid(slope * (theta - distances[i, j]))
            if element1 < element2:
                vector[ele_idx[element1], ele_idx[element2]] += value
            else:
                vector[ele_idx[element2], ele_idx[element1]] += value
    return vector.flatten().tolist()


def get_encoded_angles(elements, coords, segments=10, sigma=1., sigma2=1.):
    ele_idx = {ele: i for i, ele in enumerate(ELE_TO_NUM)}
    vector = numpy.zeros((len(ELE_TO_NUM), len(ELE_TO_NUM), len(ELE_TO_NUM), segments))
    theta = numpy.linspace(0, numpy.pi, segments)

    for i, element1 in enumerate(elements):
        for j, element2 in enumerate(elements[i + 1:]):
            j += i + 1
            vec1 = coords[i] - coords[j]
            d1 = gauss_decay_function(norm(vec1), sigma2)
            for k, element3 in enumerate(elements[j + 1:]):
                k += j + 1
                vec2 = coords[k] - coords[j]
                d2 = gauss_decay_function(norm(vec2), sigma2)
                angle = get_angle_between(vec1, vec2)
                value = gauss_decay_function(theta - angle, sigma=sigma) #* d1 * d2
            if element1 < element3:
                vector[ele_idx[element1], ele_idx[element2], ele_idx[element3]] += value
            else:
                vector[ele_idx[element3], ele_idx[element2], ele_idx[element1]] += value
    return vector.flatten().tolist()


def get_angle_counts(elements, coords, bonds=None):
    if bonds is None:
        _, bonds = get_bond_counts(elements, coords)
    types = get_all_length_types(length=3)
    typemap, counts = get_type_data(types)

    angles = []
    for i, bond1 in enumerate(bonds):
        atoms1 = set(bond1[:2])
        for j, bond2 in enumerate(bonds[i + 1:]):
            atoms2 = set(bond2[:2])
            intersect = atoms1 & atoms2
            if intersect:
                idx1 = list(atoms1 - intersect)[0]
                idx2 = list(atoms2 - intersect)[0]
                idx3 = list(intersect)[0]
                element1 = elements[idx1]
                element2 = elements[idx2]
                element3 =  elements[idx3]
                if element1 > element2:
                    # Flip if they are not in alphabetical order
                    element1, element2 = element2, element1
                    idx1, idx2 = idx2, idx1
                counts[typemap[element1, element3, element2]] += 1
                angles.append((idx1, idx3, idx2))
    return counts, angles


def get_angle_bond_counts(elements, coords, bonds=None):
    if bonds is None:
        _, bonds = get_bond_counts(elements, coords)
    types = get_all_length_types(base_types=get_all_bond_types(), length=2)
    typemap, counts = get_type_data(types)
    angles = []
    for i, bond1 in enumerate(bonds):
        atoms1 = set(bond1[:2])
        for j, bond2 in enumerate(bonds[i + 1:]):
            atoms2 = set(bond2[:2])
            intersect = atoms1 & atoms2
            if intersect:
                ele_bond1 = tuple([elements[x] for x in bond1[:2]] + [bond1[2]])
                ele_bond2 = tuple([elements[x] for x in bond2[:2]] + [bond2[2]])
                if ele_bond1 > ele_bond2:
                    # Flip if they are not in alphabetical order
                    counts[typemap[ele_bond2, ele_bond1]] += 1
                    angles.append((bond2, bond1))
                else:
                    counts[typemap[ele_bond1, ele_bond2]] += 1
                    angles.append((bond1, bond2))
    return counts, angles


def get_dihedral_counts(elements, coords, angles=None, bonds=None):
    # TODO: Add switch to add back improper dihedrals
    if bonds is None:
        _, bonds = get_bond_counts(elements, coords)
        bonds_set = set([(x, y) if x < y else (y, x) for x,y,z in bonds])
    if angles is None:
        _, angles = get_angle_counts(elements, coords, bonds=bonds)
    types = get_all_length_types(length=4)
    typemap, counts = get_type_data(types)
    dihedrals = []
    for i, angle1 in enumerate(angles):
        atoms1 = set(angle1)
        for j, angle2 in enumerate(angles[i + 1:]):
            atoms2 = set(angle2)
            intersect = atoms1 & atoms2
            if len(intersect) == 2:
                idx1 = list(atoms1 - intersect)[0]
                idx2 = list(atoms2 - intersect)[0]
                idx3, idx4 = list(intersect)

                if tuple(sorted((idx2, idx4))) not in bonds_set:
                    continue
                if tuple(sorted((idx1, idx3))) not in bonds_set:
                    continue

                element1 = elements[idx1]
                element2 = elements[idx2]
                element3 =  elements[idx3]
                element4 = elements[idx4]

                if element3 > element4:
                    # Flip if they are not in alphabetical order
                    element3, element4 = element4, element3
                    idx3, idx4 = idx4, idx3

                    element1, element2 = element2, element1
                    idx1, idx2 = idx2, idx1

                counts[typemap[element1, element3, element4, element2]] += 1
                dihedrals.append((idx1, idx3, idx4, idx2))
    return counts, dihedrals


def get_dihedral_bond_counts(elements, coords, angles=None, bonds=None):
    # TODO: Add switch to add back improper dihedrals
    if bonds is None:
        _, bonds = get_bond_counts(elements, coords)
    bonds_dict = dict(((x, y), z) if x < y else ((y, x), z) for x, y, z in bonds)
    if angles is None:
        _, angles = get_angle_bond_counts(elements, coords, bonds=bonds)
    types = get_all_length_types(base_types=get_all_bond_types(), length=3)
    types = [(x,y,z) for (x,y,z) in types if x[1] == y[0] and y[1] == z[0]]
    # types = [(x,y,z) for (x,y,z) in types if any(a in y for a in x) and any(a in z for a in y)]
    typemap, counts = get_type_data(types)
    dihedrals = []
    for i, angle1 in enumerate(angles):
        atoms1 = set(angle1[0][:2] + angle1[1][:2])
        for j, angle2 in enumerate(angles[i + 1:]):
            atoms2 = set(angle2[0][:2] + angle2[1][:2])
            intersect = atoms1 & atoms2
            if len(intersect) == 2:
                idxs = [None, None, None, None]

                idxs[0] = list(atoms1 - intersect)[0]
                idxs[3] = list(atoms2 - intersect)[0]
                idxs[1] = [x for x in intersect if (idxs[0], x) in bonds_dict or (x, idxs[0]) in bonds_dict][0]
                idxs[2] = [x for x in intersect if (idxs[3], x) in bonds_dict or (x, idxs[3]) in bonds_dict][0]

                groups = [tuple(sorted((idx1, idx2))) for idx1, idx2 in zip(idxs, idxs[1:])]
                if not all(x in bonds_dict for x in groups):
                    continue

                eles = [elements[x] for x in idxs]

                if eles[1] > eles[2]:
                    # Flip if they are not in alphabetical order
                    eles[1], eles[2] = eles[2], eles[1]
                    idxs[1], idxs[2] = idxs[2], idxs[1]

                    eles[0], eles[3] = eles[3], eles[0]
                    idxs[0], idxs[3] = idxs[3], idxs[0]

                di_bonds = []
                for e1, e2, i1, i2 in zip(eles, eles[1:], idxs, idxs[1:]):
                    b_type = bonds_dict[i1, i2] if i1 < i2 else bonds_dict[i2, i1]
                    bond = (e1, e2, b_type)
                    di_bonds.append(bond)
                di_bonds = tuple(di_bonds)

                try:
                    counts[typemap[di_bonds]] += 1
                    dihedrals.append(di_bonds)
                except:
                    print di_bonds

    return counts, dihedrals


def get_trihedral_counts(elements, coords, dihedrals=None, angles=None, bonds=None):
    if dihedrals is None:
        _, dihedrals = get_dihedral_counts(elements, coords, bonds=bonds)
    types = get_all_length_types(length=5)
    typemap, counts = get_type_data(types)
    trihedrals = []
    for i, dihedral1 in enumerate(dihedrals):
        atoms1 = set(dihedral1)
        for j, dihedral2 in enumerate(dihedrals[i + 1:]):
            atoms2 = set(dihedral2)
            intersect = atoms1 & atoms2
            if len(intersect) == 3:
                idx1 = list(atoms1 - intersect)[0]
                idx2 = list(atoms2 - intersect)[0]
                temp_sorted = sorted([(dihedral1.index(x), x) for x in intersect])
                (_, idx3), (_, idx5), (_, idx4) = temp_sorted
                element1 = elements[idx1]
                element2 = elements[idx2]
                element3 =  elements[idx3]
                element4 = elements[idx4]
                element5 = elements[idx5]
                if element1 > element2:
                    # Flip if they are not in alphabetical order
                    element1, element2 = element2, element1
                    idx1, idx2 = idx2, idx1
                if element3 > element4:
                    # Flip if they are not in alphabetical order
                    element3, element4 = element4, element3
                    idx3, idx4 = idx4, idx3
                counts[typemap[element1, element3, element5, element4, element2]] += 1
                trihedrals.append((idx1, idx3, idx5, idx4, idx2))
    return counts, trihedrals


def get_angle_between(vector1, vector2):
    '''
    Calculate the angle between two vectors.
    '''
    unit_v1 = vector1 / norm(vector1)
    unit_v2 = vector2 / norm(vector2)
    angle = numpy.arccos(numpy.dot(unit_v1, unit_v2.T))
    if numpy.isnan(angle):
        if (unit_v1 == unit_v2).all():
            return 0.0
        else:
            return numpy.pi
    return angle[0,0]


def get_dihedral_angle(idxs, coords):
    '''
    Get the dihedral angle between the 4 atoms given in `idxs`.
    '''
    first_idxs = idxs[:2]
    second_idxs = idxs[2:]
    first = coords[first_idxs[0]] - coords[first_idxs[1]]
    second = coords[second_idxs[0]] - coords[second_idxs[1]]
    space = coords[first_idxs[1]] - coords[second_idxs[0]]
    first -= space
    return get_angle_between(first, second)


def get_angle_angle(idxs, coords):
    '''
    Get the angle between the 3 atoms given in `idxs`.
    '''
    first_idxs = idxs[:2]
    second_idxs = idxs[1:]
    first = coords[first_idxs[0]] - coords[first_idxs[1]]
    second = coords[second_idxs[0]] - coords[second_idxs[1]]
    return get_angle_between(first, second)


def get_bond_length(idxs, coords):
    '''
    Get the bond length between 2 atoms given in `idxs`.
    '''
    return norm(coords[idxs[0], :] - coords[idxs[1]])


def set_vector_length(vector, length, fill=0.0):
    '''
    Given the input `vector`, return that vector extended to length `length`
    and filled with `fill`.
    '''
    return vector + [fill] * (length - len(vector))


def construct_zmatrix_addition(elements, connections, values, idxs):
    vector = []
    for connection, value in zip(connections, values):
        if connection is not None:
            vector += sum([map_atom(elements[connection[i]]) for i in idxs], []) + [value]
        else:
            vector += sum([map_atom(None) for i in idxs], []) + [value]
    return vector
