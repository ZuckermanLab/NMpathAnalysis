'''
Created on Aug 8, 2016

@author: ernesto
'''

from numbers import Number
from nmpath.auxfunctions import euclidean_distance


def rectilinear_mapper(bb):
    '''Returns a function:

        f : R^n --> N

    that maps the coordinates in R^n to an integer index
    using a rectilinear grid in R^n. Each rectangular cell
    will have its own index. The list bin_bounds stores for
    each dimension, the grid lines.
    '''
    bin_bounds = bb

    def r_mapper(coords):
        nonlocal bin_bounds
        # bin_bounds = b_bounds

        if isinstance(coords, Number):
            # if coords are a single number and not a list
            # we make a list with a single element
            coords = [coords]

        if all([isinstance(element, Number) for element in bin_bounds]):
            # it is a 1D variable
            bin_bounds = [bin_bounds]

        n_variables = len(coords)

        assert(n_variables == len(bin_bounds))

        n_partitions_in_dimension = []
        for bin_bounds_1D in bin_bounds:
            if any([bin_bounds_1D[i] >= bin_bounds_1D[i + 1] for i in range(len(bin_bounds_1D) - 1)]):
                raise Exception(
                    'For each dimension, the state bounds should increase monotonically')
            n_partitions_in_dimension.append(len(bin_bounds_1D) - 1)

        indexes_1D = [-1 for i in range(n_variables)]  # initializing
        for j in range(n_variables):
            mapped = False
            for k in range(n_partitions_in_dimension[j]):
                if bin_bounds[j][k] <= coords[j] < bin_bounds[j][k + 1]:
                    indexes_1D[j] = k
                    mapped = True
                    break
            if not mapped:
                raise Exception(
                    'A coord value could not be mapped, make sure the values/bin_bounds are correct')

        global_index = 0
        for i in range(n_variables):
            product = 1
            for j in range(i + 1, n_variables):
                product *= n_partitions_in_dimension[j]
            global_index += indexes_1D[i] * product
        return global_index

    return r_mapper


def voronoi_mapper(voronoi_centers):
    '''Returns a function

       f : R^n --> N,
    that maps the coordinates in R^n to an integer index,
    the index of the closest voronoi center listed in the
    variable voronoi_centers
    '''
    def v_mapper(coords):
        nonlocal voronoi_centers
        closest_center_index = 0
        min_dist = float('inf')

        for index, center in enumerate(voronoi_centers):
            distance = euclidean_distance(coords, center)
            if distance < min_dist:
                min_dist = distance
                closest_center_index = index

        return closest_center_index

    return v_mapper


def identity(x):
    '''Returns the same value guiven as input
    '''
    return x


## Testing ####################################
if __name__ == '__main__':
    bb = [[0, 2, 5, 7, 9], [1, 2, 3]]

    my_map = rectilinear_mapper(bb)
    coords = [5.5, 2]
    print(my_map(coords))

    voronoi_centers = [i * 0.5 for i in range(5)]
    print(voronoi_centers)

    my_v_map = voronoi_mapper(voronoi_centers)
    print(my_v_map(1.2))
