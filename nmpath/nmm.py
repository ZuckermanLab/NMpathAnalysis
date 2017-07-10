'''
Created on Jul 28, 2016

'''

import numpy as np
from auxfunctions import map_to_integers, normalize_markov_matrix
from auxfunctions import pops_from_nm_tmatrix, pops_from_tmatrix
from auxfunctions import pseudo_nm_tmatrix
from mfpt import direct_mfpts, non_markov_mfpts, fpt_distribution
from mfpt import direct_fpts
from ensembles import DiscreteEnsemble


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
            # print("The trajectories are being mapped to a (new) "
            #       "list of integers. See/print the attribute seq_map "
            #       "for details")

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
        return non_markov_mfpts(self.nm_tmatrix, self.stateA, self.stateB,
                                lag_time=self.lag_time)

    def empirical_mfpts(self):
        return direct_mfpts(self.trajectories, self.stateA, self.stateB,
                            lag_time=self.lag_time)

    def empirical_fpts(self):
        return direct_fpts(self.trajectories, self.stateA, self.stateB,
                           lag_time=self.lag_time)

    def populations(self):
        return pops_from_nm_tmatrix(self.nm_tmatrix)

    def tmatrixAB(self):
        matrixAB = []
        for i in range(0, 2 * self.n_states, 2):
            for j in range(0, 2 * self.n_states, 2):
                if (i // 2 in self.stateB) and not(j // 2 in self.stateB):
                    matrixAB.append(0.0)
                elif (i // 2 in self.stateB) and (j // 2 in self.stateB):
                    if i // 2 == j // 2:
                        matrixAB.append(1.0)
                    else:
                        matrixAB.append(0.0)
                elif not(i // 2 in self.stateB) and (j // 2 in self.stateB):
                    matrixAB.append(self.nm_tmatrix[i, j + 1])
                else:
                    matrixAB.append(self.nm_tmatrix[i, j])
        matrixAB = np.array(matrixAB)
        matrixAB = matrixAB.reshape((self.n_states, self.n_states))
        return matrixAB

    def tmatrixBA(self):
        matrixBA = []
        for i in range(1, 2 * self.n_states + 1, 2):
            for j in range(1, 2 * self.n_states + 1, 2):
                if (i // 2 in self.stateA) and not(j // 2 in self.stateA):
                    matrixBA.append(0.0)
                elif (i // 2 in self.stateA) and (j // 2 in self.stateA):
                    if i // 2 == j // 2:
                        matrixBA.append(1.0)
                    else:
                        matrixBA.append(0.0)
                elif not(i // 2 in self.stateA) and (j // 2 in self.stateA):
                    matrixBA.append(self.nm_tmatrix[i, j - 1])
                else:
                    matrixBA.append(self.nm_tmatrix[i, j])
        matrixBA = np.array(matrixBA)
        matrixBA = matrixBA.reshape((self.n_states, self.n_states))
        return matrixBA

    def fluxAB_distribution_on_B(self):
        distrib_on_B = np.zeros(len(self.stateB))
        labeled_pops = pops_from_tmatrix(self.nm_tmatrix)
        for i in range(0, 2 * self.n_states, 2):
            for j in range(2 * self.n_states):
                if j // 2 in self.stateB:
                    distrib_on_B[self.stateB.index(j // 2)] += \
                        labeled_pops[i] * self.nm_tmatrix[i, j]
        return distrib_on_B

    def fluxBA_distribution_on_A(self):
        distrib_on_A = np.zeros(len(self.stateA))
        labeled_pops = pops_from_tmatrix(self.nm_tmatrix)
        for i in range(1, 2 * self.n_states + 1, 2):
            for j in range(2 * self.n_states):
                if j // 2 in self.stateA:
                    distrib_on_A[self.stateA.index(j // 2)] += \
                        labeled_pops[i] * self.nm_tmatrix[i, j]
        return distrib_on_A

    def fpt_distrib_AB(self, max_n_lags=1000):
        return fpt_distribution(self.tmatrixAB(), self.stateA,
                                self.stateB,
                                initial_distrib=self.fluxBA_distribution_on_A(),
                                max_n_lags=max_n_lags, lag_time=self.lag_time)

    def fpt_distrib_BA(self, max_n_lags=1000):
        return fpt_distribution(self.tmatrixBA(), self.stateB,
                                self.stateA,
                                initial_distrib=self.fluxAB_distribution_on_B(),
                                max_n_lags=max_n_lags, lag_time=self.lag_time)


class MarkovPlusColorModel(NonMarkovModel, DiscreteEnsemble):

    def __init__(self, trajectories, stateA, stateB,
                 lag_time=1, clean_traj=False, sliding_window=True,
                 hist_length=0, **kwargs):
        self.hist_length = hist_length
        super().__init__(trajectories, stateA, stateB,
                         lag_time, clean_traj, sliding_window, **kwargs)

    def fit(self):
        '''Fits the the markov plus color model from a list of sequences
        '''

        # Non-Markovian count matrix
        nm_tmatrix = np.zeros((2 * self.n_states, 2 * self.n_states))

        # Markovian transition matrix
        m_tmatrix = np.zeros((self.n_states, self.n_states))

        start = self.lag_time
        step = 1

        lag = self.lag_time
        hlength = self.hist_length

        if not self.sliding_window:
            step = lag

        # Markov first
        for traj in self.trajectories:
            for i in range(start, len(traj), step):
                m_tmatrix[traj[i - lag], traj[i]] += 1.0  # counting
        m_tmatrix = m_tmatrix + m_tmatrix.T
        m_tmatrix = normalize_markov_matrix(m_tmatrix)

        p_nm_tmatrix = pseudo_nm_tmatrix(m_tmatrix, self.stateA, self.stateB)
        pops = pops_from_tmatrix(p_nm_tmatrix)

        # Pseudo-Markov Flux matrix
        fmatrix = p_nm_tmatrix
        for i, _ in enumerate(fmatrix):
            fmatrix[i] *= pops[i]

        for traj in self.trajectories:
            for i in range(start, len(traj), step):

                # Previous color determination (index i - lag)
                prev_color = "U"
                for k in range(i - lag, max(i - lag - hlength, 0) - 1, - 1):
                    if traj[k] in self.stateA:
                        prev_color = "A"
                        break
                    elif traj[k] in self.stateB:
                        prev_color = "B"
                        break

                # Current Color (in index i)
                if traj[i] in self.stateA:
                    color = "A"
                elif traj[i] in self.stateB:
                    color = "B"
                else:
                    color = prev_color

                if prev_color == "A" and color == "B":
                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += 1.0
                elif prev_color == "B" and color == "A":
                    nm_tmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] += 1.0
                elif prev_color == "A" and color == "A":
                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i]] += 1.0
                elif prev_color == "B" and color == "B":
                    nm_tmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] += 1.0
                elif prev_color == "U" and color == "B":
                    temp_sum = fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] +\
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1]
                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += \
                        fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] += \
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] /\
                        temp_sum
                elif prev_color == "U" and color == "A":
                    temp_sum = (fmatrix[2 * traj[i - lag], 2 * traj[i]] +
                                fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]])
                    nm_tmatrix[2 * traj[i - lag]][2 * traj[i]] += \
                        fmatrix[2 * traj[i - lag], 2 * traj[i]] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1][2 * traj[i]] += \
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] / temp_sum

                elif prev_color == "U" and color == "U":
                    temp_sum = fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] +\
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] +\
                        fmatrix[2 * traj[i - lag], 2 * traj[i]] +\
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]]

                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += \
                        fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1][2 * traj[i] + 1] += \
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] /\
                        temp_sum
                    nm_tmatrix[2 * traj[i - lag]][2 * traj[i]] += \
                        fmatrix[2 * traj[i - lag], 2 * traj[i]] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1][2 * traj[i]] += \
                        fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] / temp_sum

        self.nm_cmatrix = nm_tmatrix  # not normalized, it is like count matrix

        nm_tmatrix = normalize_markov_matrix(nm_tmatrix)
        self.nm_tmatrix = nm_tmatrix

    def populations(self):
        return NotImplementedError("You should use a regular Markov model or "
                                   "a non-Markovian model for estimating "
                                   "populations")


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

    model = MarkovPlusColorModel(trajectories, stateA=[0], stateB=[2],
                                 clean_traj=False, sliding_window=True, lag_time=1)
