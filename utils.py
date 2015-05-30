import re
from itertools import product

from scipy.spatial.distance import cdist
from scipy.special import erf
import numpy
from numpy.linalg import norm


def true_strip(string, left, right):
    if string.startswith(left):
        string = string[len(left):]
    if string.endswith(right):
        string = string[:-len(right)]
    return string


def read_file_data(path):
    elements = []
    numbers = []
    coords = []
    types = {'C': 6, 'H': 1, 'O': 8, 'N': 7, 'F': 9, 'S': 18}
    with open(path, 'r') as f:
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            elements.append(ele)
            numbers.append(types[ele])
            coords.append(point)
    return elements, numbers, numpy.matrix(coords)


def get_coulomb_matrix(numbers, coords):
    top = numpy.outer(numbers, numbers).astype(numpy.float64)
    r = get_distance_matrix(coords, power=1)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.divide(top, r, top)
    numpy.fill_diagonal(top, 0.5 * numpy.array(numbers) ** 2.4)
    top[top == numpy.Infinity] = 0
    top[numpy.isnan(top)] = 0
    return top


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


def get_thermometer_encoding(X, step=1):
    '''
    This is another method of encoding floating point values so that they work
    better with neural nets.

    This is based off the work in:
    Yunho Jeon and Chong-Ho Choi. IJCNN, (3) 1685-1690, 1999.
    and the recommendation from:
    Gregoire Montavon. On Layer-Wise Representations in Deep Neural Networks.
    '''
    b = numpy.arange(0, X.max() + step, step)
    temp = numpy.tanh(numpy.subtract.outer(X,b) / step)
    return temp.reshape(-1)


def get_eigenvalues(X):
    '''
    This returns the eigenvalues of a matrix in descending order.
    '''
    eigvals = numpy.linalg.eigvals(X)
    eigvals.sort()
    return numpy.real(eigvals[::-1])


ARYL = ['2', '3', '4', '6', '11', '12', '13']
ARYL0 = ['2', '3', '11']
RGROUPS = ['a', 'd', 'e', 'f', 'h', 'i', 'l']


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
    return (factor * (distance ** -H)) ** power


def gauss_decay_function(x, sigma=6):
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


def calculate_forces(clf, numbers, coords, meta=None, h=1e-6):
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

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = ax.matshow(results, extent=[-max_displacement,max_displacement,-max_displacement,max_displacement])
    fig.colorbar(m)
    plt.show()
    print results.max(), results.min(), results.std()
    return results


BOND_LENGTHS = {
    "C": {
        "3":   0.62,
        "2":   0.69,
        "Ar": 0.72,
        "1":   0.85,
    },
    "S": {
        "2":   0.905,
        "Ar": 0.945,
        "1":   1.07,
    },
    "O": {
        "3":   0.53,
        "2":   0.59,
        "Ar": 0.62,
        "1":   0.695,
    },
    "N": {
        "3":   0.565,
        "2":   0.63,
        "Ar": 0.655,
        "1":   0.74,
    },
    "H": {
        "1":   0.6,
    },
    "F": {
        "1":   1.23,
    },
}

TYPE_ORDER = ['1', 'Ar', '2', '3']


def get_connectivity_matrix(elements, coords):
    r = get_distance_matrix(coords, power=1)
    results = numpy.empty(r.shape, dtype=str)
    for bond_type in TYPE_ORDER:
        a = [BOND_LENGTHS[ele].get(bond_type, -1000) for ele in elements]
        limits = numpy.add.outer(a, a)
        results[r < limits] = bond_type
    return results


def map_atom(element):
    return [int(x == element) for x in BOND_LENGTHS]


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


def get_bond_type(element1, element2, dist):
    for key in TYPE_ORDER[::-1]:
        try:
            if dist < (BOND_LENGTHS[element1][key] + BOND_LENGTHS[element2][key]):
                return key
        except KeyError:
            continue


def get_bond_counts(elements, coords):
    types = get_all_bond_types()
    typemap, counts = get_type_data(types)

    bonds = []
    for i, (element1, xyz1) in enumerate(zip(elements, coords)):
        for j, (element2, xyz2) in enumerate(zip(elements, coords)[i + 1:]):
            j += i + 1
            dist = sum((x - y) ** 2 for (x, y) in zip(xyz1, xyz2)) ** 0.5
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


def get_angle_counts(elements, coords, bonds=None):
    if bonds is None:
        _, bonds = get_bond_counts(elements, coords)
    types = get_all_length_types(length=3)
    typemap, counts = get_type_data(types)

    angles = []
    for i, bond1 in enumerate(bonds):
        atoms1 = set(bond1[:2])
        for j, bond2 in enumerate(bonds[i + 1:]):
            j += i + 1
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
            j += i + 1
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
            j += i + 1
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


def get_trihedral_counts(elements, coords, dihedrals=None, angles=None, bonds=None):
    if dihedrals is None:
        _, dihedrals = get_dihedral_counts(elements, coords, bonds=bonds)
    types = get_all_length_types(length=5)
    typemap, counts = get_type_data(types)
    trihedrals = []
    for i, dihedral1 in enumerate(dihedrals):
        atoms1 = set(dihedral1)
        for j, dihedral2 in enumerate(dihedrals[i + 1:]):
            j += i + 1
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
    first_idxs = idxs[:2]
    second_idxs = idxs[2:]
    first = coords[first_idxs[0]] - coords[first_idxs[1]]
    second = coords[second_idxs[0]] - coords[second_idxs[1]]
    space = coords[first_idxs[1]] - coords[second_idxs[0]]
    first -= space
    return get_angle_between(first, second)


def get_angle_angle(idxs, coords):
    first_idxs = idxs[:2]
    second_idxs = idxs[1:]
    first = coords[first_idxs[0]] - coords[first_idxs[1]]
    second = coords[second_idxs[0]] - coords[second_idxs[1]]
    return get_angle_between(first, second)


def get_bond_length(idxs, coords):
    return norm(coords[idxs[0], :] - coords[idxs[1]])