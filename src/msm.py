'''
Created on Jul 28, 2016

'''

import numpy as np
from scipy import linalg, matrix

class MarkovStateModel:
    '''
    Basic Markov model. Computes basic properties from a Markov model
    constructed from a list of 1D sequences of integers
    
    For example:
    
    sequences = [ [1 , 2, 0, ...], [2, 2, 1, ...], [3, 1, 2, ...], ...]
    
    If only one sequence is given in sequences, the format has to be the same:
    
    sequences = [ [1 , 2, 0, ...] ]
    
    The first step should be to build the model, which creates the
    count matrix, from which any parameter can be estimated
    
    '''

    def __init__(self, lag_time = 1):
        '''
        sequence: 
            Sequence of integers. Is a 1D discrete ensemble or 
        '''
        self.count_matrix = None
        self.lag_time = lag_time
        
    def build(self, n_sates, sequences):
        '''
        Compute count matrix
        '''
        
        self.count_matrix = np.matrix((n_states, n_states))
        
        for sequence in sequences:
            previous_state = "Unknown"
            for state in sequence:
                if previous_state != "Unknown":
                    count_matrix[previous_state, state] += 1.0
            previous_state = current_state
        
        return count_matrix

def main():
    pass

if __name__ == '__main__':
    main()