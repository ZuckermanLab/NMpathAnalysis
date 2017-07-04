'''
Created on Jul 28, 2016

'''

import numpy as np
from auxfunctions import map_to_integers, normalize_markov_matrix


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
                 lag_time=1):
        self.lag_time = lag_time
        self.trajectories = trajectories
        self.stateA = stateA
        self.stateB = stateB

        if (self.lag_time < 1) or (int(self.lag_time) != int(self.lag_time)):
            raise ValueError('The lag time should be an integer \
            greater than 1')

        self.map_trajectories_to_integers()

    def map_trajectories_to_integers(self):
        # Clean the sequences
        seq_map = {}
        new_trajs = []
        for seq in self.trajectories:
            newseq, m_dict = map_to_integers(seq, seq_map)
            new_trajs.append(newseq)
        self._stateA = [seq_map[i] for i in self.stateA]
        self._stateB = [seq_map[i] for i in self.stateB]
        self.n_states = len(seq_map)
        self._trajectories = new_trajs
        self.seq_map = seq_map

        self.fit()

    def fit(self):
        '''Fits the the non markovian model from a list of sequences
        '''
        # Non-Markovian count matrix
        nm_cmatrix = np.zeros((2 * self.n_states, 2 * self.n_states))

        for traj in self._trajectories:

            prev_color = None
            for i, curr_microstate in enumerate(traj, start=self.lag_time):
                prev_microstate = traj[i - self.lag_time]
                # Macro state determination
                if curr_microstate in self._stateA:
                    state = "A"
                elif curr_microstate in self._stateB:
                    state = "B"
                else:
                    state = None

                # Color determination
                if state == "A":
                    color = "A"
                elif state == "B":
                    color = "B"
                else:
                    color = prev_color

                # Count matrix for the given lag time
                if prev_color == "A" and color == "B":
                    nm_cmatrix[2 * prev_microstate, 2 * curr_microstate + 1] += 1.0
                elif prev_color == "B" and color == "A":
                    nm_cmatrix[2 * prev_microstate + 1, 2 * curr_microstate] += 1.0
                elif prev_color == "A" and color == "A":
                    nm_cmatrix[2 * prev_microstate, 2 * curr_microstate] += 1.0
                elif prev_color == "B" and color == "B":
                    nm_cmatrix[2 * prev_microstate + 1, 2 * curr_microstate + 1] += 1.0

                prev_color = color

        nm_tmatrix = normalize_markov_matrix(nm_cmatrix)

        self.nm_tmatrix = nm_tmatrix
        self.nm_cmatrix = nm_cmatrix


if __name__ == '__main__':
    from random import random as rd
    from mfpt import non_markov_mfpts

    trajectories = [np.random.randint(0, 9, 500)]
    print(trajectories)

    model = NonMarkovModel(trajectories, stateA=[3], stateB=[4])
    print(model._stateA)
    print(model._stateB)
    print(model.n_states)

    for traj in model._trajectories:
        print(traj)

    print(non_markov_mfpts(model.nm_tmatrix, model._stateA, model._stateB))
