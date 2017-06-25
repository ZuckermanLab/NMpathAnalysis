'''
Created on Jul 28, 2016

'''

import numpy as np
from auxfunctions import map_to_integers


class NonMarkovModel:

    '''Non Markovian Model
    ----------------------

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

    def __init__(self, trajectories=None, stateA=None, stateB=None,
                 lag_time=1, clean_traj=False):
        self.lag_time = lag_time
        self.trajectories = trajectories
        self.stateA = stateA
        self.stateB = stateB

        if (self.lag_time < 1) or (int(self.lag_time) != int(self.lag_time)):
            raise ValueError('The lag time should be an integer \
            greater than 1')

        # Clean the sequences
        if clean_traj:
            seq_map = {}
            for seq in self.trajectories:
                newseq, m_dict = map_to_integers(seq, seq_map)
                seq = newseq
            self._stateA = [seq_map[i] for i in self.stateA]
            self._stateB = [seq_map[i] for i in self.stateB]

    def mfpt(self, sequences, stateA, stateB):
        '''Fits the the markov model from a list of sequences

        '''


def main():
    trajectories = [[3, 1, 5, 4]]

    model = NonMarkovModel(trajectories, stateA=[3], stateB=[4])
    print(model._stateA)
    print(model._stateB)


if __name__ == '__main__':
    main()
