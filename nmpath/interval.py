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

    def __init__(self, interval_set, n_variables):
        self.interval_set = interval_set
        self.n_variables = n_variables

    def __contains__(self, item):

        shape = np.array(self.interval_set).shape

        len_shape = len(shape)

        if (self.n_variables == 1) and (len_shape == 1):  # single 1D interval
            return self.interval_set[0] <= item < self.interval_set[1]

        elif (self.n_variables == 1) and (len_shape == 2):  # union of multiple 1D intervals
            return any([(item in Interval(self.interval_set[i], 1)) for i in range(shape[0])])

        elif (self.n_variables > 1) and len_shape == 2:  # n-dimensional interval
            return all([(item[i] in Interval(self.interval_set[i], 1)) for i in range(shape[0])])

        elif len(shape) == 3:  # union of n-dimensional intervals
            return any([(item in Interval(self.interval_set[i], self.n_variables)) for i in range(shape[0])])
        else:
            raise Exception('The given interval has not the expected shape')
