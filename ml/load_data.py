import os
import cPickle

import numpy

from utils import tokenize, true_strip, erf_over_r, read_file_data, \
         map_atom, mkdir_p
from constants import BHOR_TO_ANGSTROM, ARYL, NUM_TO_ELE, HARTREE_TO_KCAL
from features.utils import invert_coulomb_matrix

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


def calculate_atomization_energies(atom_counts, energies):
    # H C N O F
    atom_energies = numpy.matrix([
        [-0.497912],
        [-37.844411],
        [-54.581501],
        [-75.062219],
        [-99.716370]
    ])
    return energies - atom_counts * atom_energies


def build_gdb13_data():
    atom_idxs = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    base_path = os.path.join(DATA_BASE_DIR, "gdb13")
    mkdir_p(base_path)

    energies = []
    atom_counts = []
    for name in sorted(os.listdir(os.path.join(base_path, "xyz"))):
        xyz_path = os.path.join(base_path, "xyz", name)
        out_path = xyz_path.replace("xyz", "out")

        natoms = 0
        energy = None
        counts = [0 for _ in atom_idxs]
        with open(xyz_path, 'r') as xyz_f, open(out_path, 'w') as out_f:
            for i, line in enumerate(xyz_f):
                line = line.strip()
                if not i:
                    natoms = int(line)
                elif i == 1:
                    energy = float(line.split()[-3])
                elif i - 2 < natoms:
                    line = line.replace("*^", "e")
                    ele, x, y, z, _ = line.split()
                    counts[atom_idxs[ele]] += 1
                    out_f.write("%s %.8f %.8f %.8f\n" % (ele, float(x), float(y), float(z)))
        energies.append(energy)
        atom_counts.append(counts)
    atom_counts = numpy.matrix(atom_counts)
    atomization = calculate_atomization_energies(atom_counts, numpy.matrix(energies).T)
    atomization *= HARTREE_TO_KCAL
    numpy.savetxt(os.path.join(base_path, "energies.txt"), atomization)
    numpy.savetxt(os.path.join(base_path, "heavy_counts.txt"), atom_counts.sum(1))


def load_gdb13_data():
    base_path = os.path.join(DATA_BASE_DIR, "gdb13")
    if not os.path.isdir(base_path) or not os.listdir(base_path):
        build_gdb13_data()

    names = []
    datasets = []
    geom_paths = []
    meta = []
    lengths = []

    cutoff = 100000

    out_path = os.path.join(base_path, "out")
    for name in sorted(os.listdir(out_path))[:cutoff]:
        path = os.path.join(out_path, name)
        geom_paths.append(path)
        meta.append([1])
        datasets.append((1, ))

    lengths = numpy.loadtxt(os.path.join(base_path, "heavy_counts.txt")).astype(int).tolist()[:cutoff]
    names = ["test" if x > 7 else "train" for x in lengths]
    props = numpy.loadtxt(os.path.join(base_path, "energies.txt"))[:cutoff]
    prop_out = (("Atomization Energy", "kcal", [props.tolist()]), )
    return names, datasets, geom_paths, prop_out, meta, lengths


def build_qm7_data():
    mkdir_p(os.path.join(DATA_BASE_DIR, "qm7"))
    with open(os.path.join(DATA_BASE_DIR, "qm7.pkl"), "r") as f:
        temp = cPickle.load(f)
        X = temp['X'].reshape(7165, 23*23)
        Z = temp['Z']
        R = temp['R']
        T = temp['T']
        P = temp['P']

    for i, (zs, coords, t) in enumerate(zip(Z, R, T)):
        name = "qm-%04d" % i
        path = os.path.join(DATA_BASE_DIR, "qm7", name + ".out")
        with open(path, "w") as f:
            for z, coord in zip(zs, coords):
                z = int(z)
                if z:
                    coord = [x * BHOR_TO_ANGSTROM for x in coord]
                    f.write("%s %.8f %.8f %.8f\n" % (NUM_TO_ELE[z], coord[0], coord[1], coord[2]))


def load_qm7_data():
    base_path = os.path.join(DATA_BASE_DIR, "qm7")
    if not os.path.isdir(base_path) or not os.listdir(base_path):
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
    Psets = [set(x.tolist()) for x in P]

    for i, (zs, coords, t) in enumerate(zip(Z, R, T)):
        name = "qm-%04d" % i
        path = os.path.join(base_path, name + ".out")
        for j, value in enumerate(Psets):
            if i in value:
                name = str(j)
        names.append(name)
        datasets.append((1, ))
        geom_paths.append(path)
        properties.append([t])
        meta.append([1])
        lengths.append(1)

    prop_out = (("Atomization Energy", "kcal", zip(*properties)), )
    return names, datasets, geom_paths, prop_out, meta, lengths


def build_qm7b_data():
    mkdir_p(os.path.join(DATA_BASE_DIR, "qm7b"))
    with open(os.path.join(DATA_BASE_DIR, "qm7b.pkl"), "r") as f:
        temp = cPickle.load(f)
        X = temp['X']

    for i, coulomb_mat in enumerate(X):
        zs, coords = invert_coulomb_matrix(coulomb_mat)
        zs = zs.tolist()
        coords = coords.tolist()
        name = "qm-%04d" % i
        path = os.path.join(DATA_BASE_DIR, "qm7b", name + ".out")
        with open(path, "w") as f:
            for z, coord in zip(zs, coords):
                z = int(z)
                if z:
                    coord = [x * BHOR_TO_ANGSTROM for x in coord]
                    f.write("%s %.8f %.8f %.8f\n" % (NUM_TO_ELE[z], coord[0], coord[1], coord[2]))



def load_qm7b_data():
    base_path = os.path.join(DATA_BASE_DIR, "qm7b")
    if not os.path.isdir(base_path) or not os.listdir(base_path):
        build_qm7b_data()

    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    with open(os.path.join(DATA_BASE_DIR, "qm7b.pkl"), "r") as f:
        temp = cPickle.load(f)
        T = temp['T']

    for i, t in enumerate(T):
        name = "qm-%04d" % i
        path = os.path.join(base_path, name + ".out")
        names.append(name)
        datasets.append((1, ))
        geom_paths.append(path)
        properties.append(t)
        meta.append([1])
        lengths.append(1)

    prop_desc = [
        ('aepbe0', "kcal"),
        ('zindo-excitation-energy-with-the-most-absorption', "eV?"),
        ('zindo-highest-absorption', "eV?"),
        ('zindo-homo', "eV"),
        ('zindo-lumo', "eV"),
        ('zindo-1st-excitation-energy', "eV"),
        ('zindo-ionization-potential', "eV"),
        ('zindo-electron-affinity', "eV"),
        ('ks-homo', "eV"),
        ('ks-lumo', "eV"),
        ('gw-homo', "eV"),
        ('gw-lumo', "eV"),
        ('polarizability-pbe', "A^3"),
        ('polarizability-scs', "A^3"),
    ]
    prop_vals = zip(*properties)
    prop_out = [(x, y, z) for ((x, y), z) in zip(prop_desc, prop_vals) if x is not None]
    return names, datasets, geom_paths, prop_out, meta, lengths


def build_dave_data():
    with open(os.path.join(DATA_BASE_DIR, "geom.txt"), 'r') as f:
        elements = []
        for line in f:
            numbers = [int(x) for x in line.strip().split()]
            if len(numbers) == 2:
                try:
                    f2.close()
                except UnboundLocalError:
                    pass
                name = "dave-%04d" % (int(numbers[0]) - 1)
                path = os.path.join(DATA_BASE_DIR, "dave", name + ".out")
                f2 = open(path, "w")
            elif len(numbers) == 3:
                f2.write(" ".join([elements.pop()] + numbers) + "\n")
            else:
                elements = [NUM_TO_ELE[x] for x in numbers][::-1]


def load_dave_data(add_extra=True):
    base_path = os.path.join(DATA_BASE_DIR, "dave")
    if not os.path.isdir(base_path) or not os.listdir(base_path):
        build_dave_data()

    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    with open(os.path.join(DATA_BASE_DIR, "data.txt"), "r") as f:
        for i, line in enumerate(f):
            # Skip header line
            if not i:
                continue
            igeom,atom1,atom2,r,KELL,S,bo,q1,q2,KEHL = line.strip().split(",")

            name = "dave-%04d" % (int(igeom) - 1)
            path = os.path.join(base_path, name + ".out")

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


def load_quambo_data():
    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    with open(os.path.join(DATA_BASE_DIR, "quambo_data.txt"), "r") as f:
        for i, line in enumerate(f):
            # Skip header line
            if not i:
                continue
            values = [float(x) for x in line.strip().split(",")]

            names.append(str(i))
            datasets.append((1, ))
            geom_paths.append("")
            properties.append(values[31:])
            meta.append(values[:31] + [1.])
            lengths.append(1)

    prop_names = [
        "QU_CH_1s1s",
        "QU_CH_2s1s",
        "QU_CH_2px1s",
        "QU_CH_2py1s",
        "QU_CH_2pz1s",
    ]
    prop_desc = [(x, "kcal") for x in prop_names]
    prop_vals = zip(*properties)
    prop_out = [(x, y, z) for ((x, y), z) in zip(prop_desc, prop_vals)]
    return names, datasets, geom_paths, prop_out, meta, lengths
