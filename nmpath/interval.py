'''
Created on Jul 28, 2016

@author: esuarez
'''
import numpy as np


class Interval:
    '''Intervals are in general defined as half-open interval [start,end),
    in any case, in each dimension the interval is specified using a list [a,b]
    where a < b

    - For 1D (single) interval a single list in the form [a,b] has to be given

    - The union of multiple (1D) intervals can be specified as:
        [[a,b],[c,d],...]

    - A list of lists [[a,b],[c,d],...] are used for a n-dimensional
        intervals, one for each dimension, i.e, len(inteval) = n_variables

    - A list-of-lists-of-lists for the mathematical union of n-dimensional
        intervals'

        [ [[a,b],[c,d],...],  [[e,f],[g,h],...], ... ]

    '''

    def __init__(self, interval_set, n_dim=None):
        self.interval_set = interval_set
        self.n_dim = n_dim
        if n_dim is None:
            raise Exception('The number of variables/dimensions has to be specified')

    def __contains__(self, item):

        shape = np.array(self.interval_set).shape

        len_shape = len(shape)

        if (self.n_dim == 1) and (len_shape == 1):  # single 1D interval
            return self.interval_set[0] <= item < self.interval_set[1]

        elif (self.n_dim == 1) and (len_shape == 2):  # union of multiple 1D intervals
            return any([(item in Interval(self.interval_set[i], 1)) for i in range(shape[0])])

        elif (self.n_dim > 1) and len_shape == 2:  # n-dimensional interval
            return all([(item[i] in Interval(self.interval_set[i], 1)) for i in range(shape[0])])

        elif len(shape) == 3:  # union of n-dimensional intervals
            return any([(item in Interval(self.interval_set[i], self.n_dim)) for i in range(shape[0])])
        else:
            raise Exception('The given interval has not the expected shape')

    def __iter__(self):
        return iter(self.interval_set)

    def __getitem__(self, arg):
        return self.interval_set[arg]


if __name__ == '__main__':

    # 1D single interval
    I = Interval([1, 2], 1)
    assert(0.5 not in I)
    assert(1.3 in I)

    I = Interval([[1, 2]], 1)  # should work too
    assert(0.5 not in I)
    assert(1.3 in I)

    # 1D union of intervals
    I = Interval([[1, 2], [3, 4]], 1)
    assert(0.5 not in I)
    assert(1.9 in I)
    assert(3.3 in I)

    # 3D single interval
    I = Interval([[1, 2], [0, 1], [0.1, 0.2]], 3)
    assert([1.5, 1.5, 0.15] not in I)
    assert([1.5, 0.5, 0.15] in I)

    # 3D union of intervals
    I = Interval([[[1, 2], [0, 1], [0.1, 0.2]], [[2, 3], [1, 2], [0.1, 0.2]]], 3)
    assert([1.5, 1.5, 0.15] not in I)
    assert([1.5, 0.5, 0.15] in I)
    assert([2.5, 1.5, 0.15] in I)
    assert([2.5, 1.5, 0.3] not in I)
