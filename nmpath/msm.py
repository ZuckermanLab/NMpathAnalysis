'''
Created on Jul 28, 2016

'''

import numpy as np

class MarkovStateModel:
    '''Markov State Model
    ------------------
    Fits a regular Markov model from a list of 1D sequences of integers
    
    For example:
    
    sequences = [ [1 , 2, 0, ...], [2, 2, 1, ...], [3, 1, 2, ...], ...]
    
    If only one sequence is given in sequences, the format has to be the same:
    
    sequences = [ [1 , 2, 0, ...] ]

    Parameters
    ...........
    lag_time (integer, default: 1) 
        Lag time of the model.
    reversible (boolean, default: true) 
        Enforce the reversibility of the transition matrix.
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization, and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    prior_counts (integer)
        Add a number of "pseudo counts" to each entry in the counts matrix
        after ergodic trimming.  When prior_counts == 0 (default), the assigned
        transition probability between two states with no observed transitions
        will be zero, whereas when prior_counts > 0, even this unobserved
        transitions will be given nonzero probability.
    sliding_window (boolean)
        Count transitions using a window of length ``lag_time``, which is slid
        along the sequences 1 unit at a time, yielding transitions which
        contain more data but cannot be assumed to be statistically
        independent. Otherwise, the sequences are simply subsampled at an
        interval of ``lag_time``.
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