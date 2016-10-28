'''
Created on Jul 28, 2016

'''

import numpy as np

class MarkovStateModel:
    '''Markov State Model
    ----------------------
    **API Based on MSMBuilder sofware (this is a simplified version)
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
    reversible_type: {'transpose','mle','revmle'}
        Enforce the reversibility of the transition matrix.
        
        possible values:
        ---------------
        - 'mle': If selected, the reversibility is not enforced (regular MLE)
        - 'transpose': the count matrix is averaged with its transposed.
        - 'revmle': Maximum likelihood estimator for reversible 
                    matrix (J. Chem. Phys. 2011, 134, 174105)

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
    

    def __init__(self, lag_time=1, reversible_type=None, prior_counts=0, 
                sliding_window=True):
        self.reversible_type = reversible_type
        self.lag_time = lag_time
        self.sliding_window = sliding_window
        self.prior_counts = prior_counts
        self.transition_matrix = None
        self.count_matrix = None

        if reversible_type is None:
            serevesble_type = 'mle'
        else if reversible_type is not in ('mle','transpose','revmle'):
            raise ValueError('Reversible type: {} is not valid', reversible_type)
        
        self.reversible_type = reversible_type

        if (self.lag_time < 1) or (int(self.lag_time) != int(self.lag_time)):
            raise ValueError('The lag time should be an integer greater than 1')


    def fit(sequence):
        '''Fits the the markov model from a list of sequences
        
        '''
        ## maps the given sequence to an internal sequence
        #map = {}


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