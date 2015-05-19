import os
import cPickle

from utils import tokenize, ARYL, true_strip, erf_over_r, read_file_data, \
         map_atom


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
                path = os.path.join('../mol_data', base_path, atom_set, file_path)
                with open(path + ".txt", 'r') as f:
                    for line in f:
                        temp = line.split()
                        name, props = temp[0], temp[1:]

                        names.append(name)
                        datasets.append((base_path, file_path, atom_set))

                        geom_path = os.path.join('../mol_data', base_path, 'geoms', 'out', name + '.out')
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
    return names, datasets, geom_paths, zip(*properties), meta, lengths


def load_gdb7_data():
    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    atoms = {1: 'H', 6: "C", 7: "N", 8: "O", 16: "S"}
    with open("qm7.pkl", "r") as f:
        temp = cPickle.load(f)
        X = temp['X'].reshape(7165, 23*23)
        Z = temp['Z']
        R = temp['R']
        T = temp['T']
        P = temp['P']

    for i, (zs, coords, t) in enumerate(zip(Z, R, T)):
        # if 16 in zs.astype(int):
        #     continue

        name = "qm-%04d" % i
        path = "qm/" + name + ".out"
        # with open(path, "w") as f:
        #     for z, coord in zip(zs, coords):
        #         z = int(z)
        #         if z:
        #             f.write("%s %.8f %.8f %.8f\n" % (atoms[z], coord[0] * 0.529177249, coord[1] * 0.529177249, coord[2] * 0.529177249))
        names.append(name)
        datasets.append((1, ))
        geom_paths.append(path)
        properties.append([t])
        meta.append([])
        lengths.append(1)
    return names, datasets, geom_paths, zip(*properties), meta, lengths


def load_dave_data():
    names = []
    datasets = []
    geom_paths = []
    properties = []
    meta = []
    lengths = []

    atoms = {"1": 'H', "6": "C", "7": "N", "8": "O", "9": "F"}

    # with open("geom.txt", 'r') as f:
    #     elements = []
    #     for line in f:
    #         numbers = line.strip().split()
    #         if len(numbers) == 2:
    #             try:
    #                 f2.close()
    #             except UnboundLocalError:
    #                 pass
    #             name = "dave-%04d" % (int(numbers[0]) - 1)
    #             path = "dave/" + name + ".out"
    #             f2  = open(path, "w")
    #         elif len(numbers) == 3:
    #             f2.write(" ".join([elements.pop()] + numbers) + "\n")
    #         else:
    #             elements = [atoms[x] for x in numbers][::-1]


    with open("data/data.txt", "r") as f:
        for i, line in enumerate(f):
            if not i:
                continue
            # igeom,atom1,atom2,r,KELL,S,bo,q1,q2,KEHL
            igeom,atom1,atom2,r,KELL,S,bo,q1,q2,KEHL = line.strip().split(",")

            name = "dave-%04d" % (int(igeom) - 1)
            path = "data/dave/" + name + ".out"

            elements, numbers, coords = read_file_data(path)

            atom1 = int(atom1) - 1
            atom2 = int(atom2) - 1
            atom1_base = map_atom(elements[atom1])
            atom2_base = map_atom(elements[atom2])

            names.append('%d,%d,%d' % (atom1, atom2, i))
            datasets.append((1, ))
            geom_paths.append(path)
            properties.append((float(KEHL), ))

            # meta.append([float(r),float(KELL),float(S)])
            meta.append(atom1_base + atom2_base + [float(r),float(KELL),float(S),float(bo),float(q1),float(q2)])
            lengths.append(1)

    return names, datasets, geom_paths, zip(*properties), meta, lengths
