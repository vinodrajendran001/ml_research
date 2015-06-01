import os
import cPickle

from utils import tokenize, ARYL, true_strip, erf_over_r, read_file_data, \
         map_atom


DATA_BASE_DIR = "data"


def load_mol_data(calc_set, opt_set, struct_set, prop_set=None):
    '''
    Load data from data sets and return lists of structure names, full paths
    to the geometry data, the properties, and the meta data.
    '''
    print "Dataset options used"
    print "\tCalculation methods:", calc_set
    print "\tOptimization methods:", opt_set
    print "\tStructure sets:", struct_set
    print "\tProperties:", prop_set
    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    for j, base_path in enumerate(opt_set):
        for i, file_path in enumerate(calc_set):
            for m, atom_set in enumerate(struct_set):
                path = os.path.join(DATA_BASE_DIR, "mol_data", base_path, atom_set, file_path)
                with open(path + ".txt", 'r') as f:
                    for line in f:
                        temp = line.split()
                        name, props = temp[0], temp[1:]

                        names.append(name)
                        datasets.append((base_path, file_path, atom_set))

                        geom_path = os.path.join(DATA_BASE_DIR, "mol_data", base_path, 'geoms', 'out', name + '.out')
                        geom_paths.append(geom_path)

                        properties.append([float(x) for x in props])

                        # Add part to feature vector to account for the 4 different data sets.
                        base_part = [i == k for k, x in enumerate(opt_set)]
                        # Add part to feature vector to account for the 3 different methods.
                        method_part = [j == k for k, x in enumerate(calc_set)]
                        # Add part to feature vector to account for the addition of N.
                        atom_part = [m == k for k, x in enumerate(struct_set)]
                        # Add bias feature
                        bias = [1]
                        meta.append(base_part + method_part + atom_part + bias)

                        tokens = tokenize(name, explicit_flips=True)
                        aryl_count = sum([1 for x in tokens if x in ARYL])
                        lengths.append(aryl_count)

    prop_desc = (("HOMO", "eV"), ("LUMO", "eV"), ("Excitation", "eV"))
    prop_vals = zip(*properties)
    prop_out = [(x, y, z) for ((x, y), z) in zip(prop_desc, prop_vals)]
    return names, datasets, geom_paths, prop_out, meta, lengths


def build_qm7_data():
    BHOR_TO_ANGSTROM = 0.529177249
    atoms = {1: 'H', 6: "C", 7: "N", 8: "O", 16: "S"}
    with open(os.path.join(DATA_BASE_DIR, "qm7.pkl"), "r") as f:
        temp = cPickle.load(f)
        X = temp['X'].reshape(7165, 23*23)
        Z = temp['Z']
        R = temp['R']
        T = temp['T']
        P = temp['P']

    for i, (zs, coords, t) in enumerate(zip(Z, R, T)):
        name = "qm-%04d" % i
        path = os.path.join(DATA_BASE_DIR, "qm", name + ".out")
        with open(path, "w") as f:
            for z, coord in zip(zs, coords):
                z = int(z)
                if z:
                    coord = [x * BHOR_TO_ANGSTROM for x in coord]
                    f.write("%s %.8f %.8f %.8f\n" % (atoms[z], coord[0], coord[1], coord[2]))


def load_qm7_data():
    if not os.path.isdir(os.path.join(DATA_BASE_DIR, "qm")) or not os.listdir(os.path.join(DATA_BASE_DIR, "qm")):
        build_qm7_data()

    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    with open(os.path.join(DATA_BASE_DIR, "qm7.pkl"), "r") as f:
        temp = cPickle.load(f)
        X = temp['X'].reshape(7165, 23*23)
        Z = temp['Z']
        R = temp['R']
        T = temp['T']
        P = temp['P']

    for i, (zs, coords, t) in enumerate(zip(Z, R, T)):
        name = "qm-%04d" % i
        path = os.path.join(DATA_BASE_DIR, "qm", name + ".out")
        names.append(name)
        datasets.append((1, ))
        geom_paths.append(path)
        properties.append([t])
        meta.append([1])
        lengths.append(1)

    prop_out = (("Atomization Energy", "kcal", zip(*properties)), )
    return names, datasets, geom_paths, prop_out, meta, lengths


def build_dave_data():
    atoms = {"1": 'H', "6": "C", "7": "N", "8": "O", "9": "F"}
    with open(os.path.join(DATA_BASE_DIR, "geom.txt"), 'r') as f:
        elements = []
        for line in f:
            numbers = line.strip().split()
            if len(numbers) == 2:
                try:
                    f2.close()
                except UnboundLocalError:
                    pass
                name = "dave-%04d" % (int(numbers[0]) - 1)
                path = os.path.join(DATA_BASE_DIR, "dave", name + ".out")
                f2  = open(path, "w")
            elif len(numbers) == 3:
                f2.write(" ".join([elements.pop()] + numbers) + "\n")
            else:
                elements = [atoms[x] for x in numbers][::-1]


def load_dave_data(add_extra=True):
    if not os.path.isdir(os.path.join(DATA_BASE_DIR, "dave")) or not os.listdir(os.path.join(DATA_BASE_DIR, "dave")):
        build_dave_data()

    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    with open(os.path.join(DATA_BASE_DIR, "data.txt"), "r") as f:
        for i, line in enumerate(f):
            if not i:
                continue
            igeom,atom1,atom2,r,KELL,S,bo,q1,q2,KEHL = line.strip().split(",")

            name = "dave-%04d" % (int(igeom) - 1)
            path = os.path.join(DATA_BASE_DIR, "dave", name + ".out")

            elements, numbers, coords = read_file_data(path)

            atom1 = int(atom1) - 1
            atom2 = int(atom2) - 1
            atom1_base = map_atom(elements[atom1])
            atom2_base = map_atom(elements[atom2])

            names.append('%d,%d,%d' % (atom1, atom2, i))
            datasets.append((1, ))
            geom_paths.append(path)
            properties.append((float(KEHL), ))

            if add_extra:
                meta.append(atom1_base + atom2_base + [float(x) for x in [r , KELL, S, bo, q1, q2, 1]])
            else:
                meta.append(atom1_base + atom2_base + [float(x) for x in [r , KELL, S, 1]])

            lengths.append(1)

    prop_desc = (("Kinetic Energy", "kcal"), )
    prop_vals = zip(*properties)
    prop_out = [(x, y, z) for ((x, y), z) in zip(prop_desc, prop_vals)]
    return names, datasets, geom_paths, prop_out, meta, lengths
