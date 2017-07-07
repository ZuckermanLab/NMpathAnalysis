'''
Created on Jul 28, 2016

'''

import numpy as np
from auxfunctions import map_to_integers, normalize_markov_matrix
from auxfunctions import pops_from_nm_tmatrix
from mfpt import direct_mfpts, non_markov_mfpts


class NonMarkovModel:

    '''Non Markovian Model
    ----------------------

    Fits a regular Markov model from a list of 1D trajectories of integers

    For example:

    trajectories = [ [1 , 2, 0, ...], [2, 2, 1, ...], [3, 1, 2, ...], ...]

    If only one sequence is given in trajectories, the format is the same:

    trajectories = [ [1 , 2, 0, ...] ]

    Parameters
    ----------
    lag_time (integer, default: 1)
        Lag time of the model.

    sliding_window (boolean)
        Use a sliding window of length lag_time to compute the count matrix

    stateA, stateB (python lists)
        Define the initial and final macrostates in form of python lists
        for example: stateA=[0,2,5], stateB = [1]

    Attributes
    ----------
    n_states : int

    nm_cmatrix: array, with shape (2 n_states, 2 n_states)
        Stores the number of transitions between states, the i,j element cij
        stores the number of transitions observed from i to j.

    populations: array, shape (n_states,)
        Equilibrium population, the steady state solution of of the
        transition matrix
    '''

    def __init__(self, trajectories, stateA, stateB,
                 lag_time=1, clean_traj=False, sliding_window=True, **kwargs):
        self.lag_time = lag_time
        self.trajectories = trajectories
        self.stateA = stateA
        self.stateB = stateB
        self.sliding_window = sliding_window

        if (self.lag_time < 1) or (int(self.lag_time) != int(self.lag_time)):
            raise ValueError('The lag time should be an integer \
            greater than 1')

        if clean_traj:
            print("WARNING: The trajectories are considered to be sequences "
                  "of integers in the interval [0, N-1] where N is de number"
                  "of microstates.")
            self.n_states = max([max(traj) for traj in self.trajectories]) + 1
        else:
            self.map_trajectories_to_integers()
            print("The trajectories are being mapped to a (new) "
                  "list of integers. See/print the attribute seq_map "
                  "for details")

        self.fit()

    def map_trajectories_to_integers(self):
        # Clean the sequences
        seq_map = {}
        new_trajs = []
        for seq in self.trajectories:
            newseq, m_dict = map_to_integers(seq, seq_map)
            new_trajs.append(newseq)
        self.stateA = [seq_map[i] for i in self.stateA]
        self.stateB = [seq_map[i] for i in self.stateB]
        self.n_states = len(seq_map)
        self.trajectories = new_trajs
        self.seq_map = seq_map

    def fit(self):
        '''Fits the the non markovian model from a list of sequences
        '''
        # Non-Markovian count matrix
        nm_cmatrix = np.zeros((2 * self.n_states, 2 * self.n_states))
        start = self.lag_time
        step = 1

        if not self.sliding_window:
            step = self.lag_time

        for traj in self.trajectories:

            prev_color = None

            for i in range(start, len(traj), step):
                curr_microstate = traj[i]
                prev_microstate = traj[i - self.lag_time]
                # Macro state determination
                if curr_microstate in self.stateA:
                    state = "A"
                elif curr_microstate in self.stateB:
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

    def mfpts(self):
        return non_markov_mfpts(self.nm_tmatrix, self.stateA, self.stateB)

    def empirical_mfpts(self):
        return direct_mfpts(self.trajectories, self.stateA, self.stateB)

    def populations(self):
        return pops_from_nm_tmatrix(self.nm_tmatrix)


class MarkovPlusColorModel(NonMarkovModel):

    def __init__(self, trajectories, stateA, stateB,
                 lag_time=1, clean_traj=False, sliding_window=True,
                 hist_length=None, **kwargs):
        super.__init__(trajectories, stateA, stateB,
                       lag_time, clean_traj, sliding_window, **kwargs)

    def fit(self):
        '''Fits the the non markovian model from a list of sequences
        '''
        # Non-Markovian count matrix
        nm_cmatrix = np.zeros((2 * self.n_states, 2 * self.n_states))
        start = self.lag_time
        step = 1

        if not self.sliding_window:
            step = self.lag_time

        for traj in self.trajectories:

            prev_color = None

            for i in range(start, len(traj), step):
                curr_microstate = traj[i]
                prev_microstate = traj[i - self.lag_time]
                # Macro state determination
                if curr_microstate in self.stateA:
                    state = "A"
                elif curr_microstate in self.stateB:
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
    # from test.tools_for_notebook import m
    np.random.seed(192348)
    trajectories = [np.random.randint(0, 3, 100000)]
    print(trajectories)

    model = NonMarkovModel(trajectories, stateA=[0], stateB=[2],
                           clean_traj=False, sliding_window=True, lag_time=100)
    print("Number of states: {}".format(model.n_states))
    print(model.stateA)
    print(model.stateB)
    print(model.n_states)
    # print(model.seq_map)

    print(model.nm_tmatrix)

    print(model.mfpts())
    print(model.empirical_mfpts())
    print(model.populations())
