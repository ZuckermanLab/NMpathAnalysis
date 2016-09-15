'''
Created on Jul 28, 2016

'''

import numpy as np

class MarkovStateModel:
    '''Markov State Model
    ----------------------
    **Based on MSMBuilder sofware (this is a simplified version)
    https://github.com/msmbuilder/msmbuilder

    Fits a regular Markov model from a list of 1D sequences of integers
    
    For example:
    
    sequences = [ [1 , 2, 0, ...], [2, 2, 1, ...], [3, 1, 2, ...], ...]
    
    If only one sequence is given in sequences, the format has to be the same:
    
    sequences = [ [1 , 2, 0, ...] ]

    Parameters
    ----------
    lag_time (integer, default: 1) 
        Lag time of the model.
    reversible_type: {None, 'transpose','mle'}
        Enforce the reversibility of the transition matrix.
        
        possible values:
        ---------------
        - None (Default): If selected, the reversibility is not enforced.
        - 'transpose': the count matrix is averaged with its transposed.
        - 'mle': Maximum likelihood estimator (J. Chem. Phys. 2011, 134, 174105)

    prior_counts (integer)
        Add prior counts (the same for all the elements of the count matrix).

    sliding_window (boolean)
        Use a sliding window of length lag_time to compute the count matrix

    Attributes
    ----------
    n_states : int

    count_matrix: array, with shape (n_states, n_states)
        Stores the number of transitions between states, the i,j element cij
        stores the number of transitions observed from i to j.

    populations: array, shape (n_states,)
        Equilibrium population, the steady state solution of of the 
        transition matrix
    '''
    

    def __init__(self, lag_time=1, reversible_type=None, prior_counts=0, sliding_window=True):
        '''
        sequence: 
            Sequence of integers. Is a 1D discrete ensembles or 
        '''
        self.reversible_type = reversible_type
        self.lag_time = lag_time
        self.sliding_window = sliding_window
        self.prior_counts = prior_counts


    def fit(sequence):
        '''Fits the model from a list of sequences
        
        '''
        pass


        
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