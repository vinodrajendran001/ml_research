'''
The simplest functions that rely on no information from the molecule. These
are purely for baseline comparisons only.
'''

import numpy


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