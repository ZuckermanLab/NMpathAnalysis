'''
Created on Jul 29, 2016

'''
import numpy as np
from copy import deepcopy
import auxfunctions as aux
from interval import Interval


def direct_mfpts(trajectories, stateA=None, stateB=None, discrete=True,
                 n_variables=None, lag_time=1):
    """Empirical mean first passage times (MFPTs) calculation (no model
    involved) by tracing the trajectories. Notice the diference between
    notation between FPTs and MFPTs.

    Parameters:
    -----------
    trajectories:   List of trajectories [traj1, traj2, traj4], each trajectory
                    can be a one dimensional array, e.g.,
                        [[1,2,1, ...], [0,1,1, ...], ... ]
                    or a mutidimensional array (matrix) where each column
                    represents the evolution of a variable.

                    Important: If a single trajectory is given as argument it
                    also has to be inside a list (e.g. [traj1])

    stateA, stateB: List of integers
                    If the trajectories are discrete (discrete = True), both
                    states are a list of indexes. However, if the trajectories
                    are not discrete, the states are "interval" objects
                    (see Interval class).

    lag_time:       integer
                    Lag time used, the trajectory is "observed" every lag_time
                    time steps

    discrete:       boolean
                    False when the trajectories are are not discrete. In that
                    case the macrostates stateA and stateB are considered
                    interval objects.

    n_variables:    integer
                    If the trajectory is space continuous,the number of
                    variables/dimensions is needed. In this case every
                    trajectory inside "trajectories" should have the same
                    number of dimensions.

    Returns
    -------
    A dictionary with the keys: 'mfptAB', 'std_err_mfptAB', 'mfptBA',
    'std_err_mfptBA' and the corresponding values. Those values are already
    multiplied by the lag_time used (not the physical units).
    """

    passage_timesAB, passage_timesBA = direct_fpts(trajectories, stateA,
                                                   stateB, discrete,
                                                   n_variables, lag_time)
    n_AB = len(passage_timesAB)
    n_BA = len(passage_timesBA)

    try:
        mfptAB = float(sum(passage_timesAB)) / n_AB
        std_err_mfptAB = np.std(passage_timesAB) / np.sqrt(n_AB)
    except:
        print('WARNING: No A->B events observed')
        mfptAB = 'NaN'
        std_err_mfptAB = 'NaN'

    try:
        mfptBA = float(sum(passage_timesBA)) / n_BA
        std_err_mfptBA = np.std(passage_timesBA) / np.sqrt(n_BA)
    except:
        print('WARNING: No B->A events observed')
        mfptBA = 'NaN'
        std_err_mfptBA = 'NaN'

    kinetics = {'mfptAB': mfptAB, 'std_err_mfptAB': std_err_mfptAB,
                'mfptBA': mfptBA, 'std_err_mfptBA': std_err_mfptBA}

    print('Number of A->B/B->A  events: {}/{}'.format(n_AB, n_BA))

    return kinetics


def direct_fpts(trajectories, stateA=None, stateB=None, discrete=True,
                n_variables=None, lag_time=1):
    """Empirical first passage times (FPTs) calculation (no model involved)
    by tracing the trajectories. IMPORTANT: Notice the diference in notation
    between FPTs and MFPTs.

    Parameters:
    -----------
    trajectories:   List of trajectories [traj1, traj2, traj4], each trajectory
                    can be a one dimensional array, e.g.,
                        [[1,2,1, ...], [0,1,1, ...], ... ]
                    or a mutidimensional array (matrix) where each column
                    represents the evolution of a variable.

                    Important: If a single trajectory is given as argument it
                    also has to be inside a list (e.g. [traj1])

    stateA, stateB: List of integers
                    If the trajectories are discrete (discrete = True), both
                    states are a list of indexes. However, if the trajectories
                    are not discrete, the states are "interval" objects
                    (see Interval class).

    lag_time:       integer
                    Lag time used, the trajectory is "observed" every lag_time
                    time steps

    discrete:       boolean
                    False when the trajectories are are not discrete. In that
                    case the macrostates stateA and stateB are considered
                    interval objects.

    n_variables:    integer
                    If the trajectory is space continuous,the number of
                    variables/dimensions is needed. In this case every
                    trajectory inside "trajectories" should have the same
                    number of dimensions.

    Returns
    -------
    A tuple of two 1D-ndarray (array1, array2), the first one contains the
    observed first passage times A->B and the second one the FPTs B->A. Those
    values are already multiplied by the lag_time used (not the physical units)
    """

    if (stateA is None) or (stateB is None):
        raise Exception('The final and initial states have '
                        'to be defined to compute the MFPT')

    if not discrete:
        '''
        The states are considered/transformed-to intervals if the Ensemble
        is a set of continuous trajectories
        '''
        if n_variables is None:
            raise Exception('In continuous trajectories the number of '
                            'variables is needed')

        stateA = Interval(stateA, n_variables)
        stateB = Interval(stateB, n_variables)

    passage_timesAB = []
    passage_timesBA = []
    fpt_counter = 0  # first passage time counter

    for traj in trajectories:
        previous_color = "Unknown"
        for i in range(0, len(traj), lag_time):
            snapshot = traj[i]
            # state and color determination
            if snapshot in stateA:
                color = "A"
            elif snapshot in stateB:
                color = "B"
            else:
                color = previous_color

            # passage times
            if (color == "A") or (color == "B"):
                fpt_counter += 1

            if previous_color == "A" and color == "B":
                passage_timesAB.append(fpt_counter)
                fpt_counter = 0
            elif previous_color == "B" and color == "A":
                passage_timesBA.append(fpt_counter)
                fpt_counter = 0
            elif previous_color == "Unknown" and (color == "A" or
                                                  color == "B"):
                fpt_counter = 0

            previous_color = color

    passage_timesAB = np.array(passage_timesAB) * lag_time
    passage_timesBA = np.array(passage_timesBA) * lag_time

    return passage_timesAB, passage_timesBA


def markov_mfpts(transition_matrix, stateA, stateB, lag_time=1):
    '''Computes mean first passage times in both directions A->B and B->A
    from a markov model. The mfpts computed this way are directly comparable
    with the values obtained by a long back and forth simulation between the
    tar
    '''
    transition_matrix = np.array(transition_matrix)

    n_states = len(transition_matrix)

    # pseudo non-markovian matrix (auxiliar_matrix)
    auxiliar_matrix = np.zeros((2 * n_states, 2 * n_states))

    for i in range(2 * n_states):
        for j in range(2 * n_states):
            auxiliar_matrix[i, j] = transition_matrix[int(i / 2), int(j / 2)]

    for i in range(n_states):
        for j in range(n_states):
            if (i in stateB) or (j in stateB):
                auxiliar_matrix[2 * i, 2 * j] = 0.0
            if (i in stateA) or (j in stateA):
                auxiliar_matrix[2 * i + 1, 2 * j + 1] = 0.0
            if (not (j in stateA)) or (i in stateA):
                auxiliar_matrix[2 * i + 1, 2 * j] = 0.0
            if (not (j in stateB)) or (i in stateB):
                auxiliar_matrix[2 * i, 2 * j + 1] = 0.0

    # Is going to return a MARKOVIAN mfpt since the auxiliar
    # matrix was build from a pure markovian matrix
    return non_markov_mfpts(auxiliar_matrix, stateA, stateB, lag_time)


def markov_commute_time(transition_matrix, stateA, stateB, lag_time=1):
    mfpts = markov_mfpts(transition_matrix, stateA, stateB, lag_time)
    return mfpts['mfptAB'] + mfpts['mfptBA']


def non_markov_mfpts(nm_transition_matrix, stateA, stateB, lag_time=1):
    '''Computes the mean first passage times A->B and B->A where
    from a non-markovian model.
    The shape of the transition matrix should be (2*n_states, 2*n_states)
    '''
    aux.check_tmatrix(nm_transition_matrix)

    labeled_pops = aux.pops_from_tmatrix(nm_transition_matrix)

    n_states = len(labeled_pops) // 2

    fluxAB = 0
    fluxBA = 0

    for i in range(0, 2 * n_states, 2):
        for j in range(2 * n_states):
            if int(j / 2) in stateB:
                fluxAB += labeled_pops[i] * nm_transition_matrix[i, j]

    for i in range(1, 2 * n_states + 1, 2):
        for j in range(2 * n_states):
            if int(j / 2) in stateA:
                fluxBA += labeled_pops[i] * nm_transition_matrix[i, j]

    pop_colorA = 0.0
    pop_colorB = 0.0

    for i in range(0, 2 * n_states, 2):
        pop_colorA += labeled_pops[i]

    for i in range(1, 2 * n_states + 1, 2):
        pop_colorB += labeled_pops[i]

    if fluxAB == 0:
        mfptAB = float('inf')
    else:
        mfptAB = pop_colorA / fluxAB

    if fluxBA == 0:
        mfptBA = float('inf')
    else:
        mfptBA = pop_colorB / fluxBA

    mfptAB *= lag_time
    mfptBA *= lag_time

    return dict(mfptAB=mfptAB, mfptBA=mfptBA)


def directional_mfpt(transition_matrix, stateA, stateB,
                     ini_probs=None, lag_time=1):
    '''Computes the mean-first passage in a single direction A->B
    using a recursive procedure. This method is useful when there is no
    B->A ensemble, for instance when B is absorbing.
    '''
    lenA = len(stateA)
    lenB = len(stateB)

    if ini_probs is None:
        ini_probs = [1. / lenA for i in range(lenA)]

    t_matrix = deepcopy(transition_matrix)

    ini_state = list(stateA)
    f_state = sorted(list(stateB))

    assert(lenA == len(ini_probs))

    for i in range(lenB - 1, -1, -1):
        t_matrix = np.delete(t_matrix, f_state[i], axis=1)
        t_matrix = np.delete(t_matrix, f_state[i], axis=0)
        for j in range(lenA):
            if f_state[i] < ini_state[j]:
                ini_state[j] = ini_state[j] - 1

    new_size = len(t_matrix)

    mfptAB = 0.0

    m = np.zeros(new_size)
    I = np.identity(new_size)
    c = np.array([1.0 for i in range(new_size)])

    m = np.dot(np.linalg.inv(I - t_matrix), c)

    for i in range(len(ini_state)):
        k = ini_state[i]
        mfptAB += ini_probs[i] * m[k]
    mfptAB = mfptAB / sum(ini_probs)

    return mfptAB * lag_time


def mfpts_to_target_microstate(transition_matrix, target, lag_time=1):
    '''Computes all the mean-first passage to a target microstate (k).
    Returns a list where the i-element is mfpt(i->k). This function is
    useful to compute the mfpt matrix.

    target: integer number that specifies the index of the state. The indexes
            should be consistent with the transition matrix and python
            (i.e. starting from 0)
    '''

    t_matrix = deepcopy(transition_matrix)

    t_matrix = np.delete(t_matrix, target, axis=1)
    t_matrix = np.delete(t_matrix, target, axis=0)

    new_size = len(t_matrix)

    m = np.zeros(new_size)
    I = np.identity(new_size)
    c = np.array([1.0 for i in range(new_size)])

    m = np.dot(np.linalg.inv(I - t_matrix), c)
    m = np.insert(m, target, 0.0)

    return m * lag_time


def mfpts_matrix(transition_matrix, lag_time=1):
    '''Returns the MFPT matrix, i.e., the matrix where the ij-element is the
    MFPT(i->j)
    '''

    size = len(transition_matrix)
    temp_values = []

    for i in range(size):
        temp_values.append(mfpts_to_target_microstate(transition_matrix,
                                                      i, lag_time))

    mfpt_m = np.array(temp_values).T  # to nummpy array and transposed
    return mfpt_m


def min_commute_time(matrix_of_mfpts):
    """Returns the min commuting time (round trip time) between all pairs
    of microstates from the matrix of mfpts. It also returns the indexes
    of the pair of microstates involved"""

    matrix_of_mfpts = np.array(matrix_of_mfpts)

    n_states = len(matrix_of_mfpts)
    assert(n_states == len(matrix_of_mfpts[0]) and n_states >= 2)

    # Initial values, arbitrary choice
    index_i = 0
    index_j = 1

    commute_times = matrix_of_mfpts + matrix_of_mfpts.T
    min_ct = commute_times[index_i, index_j]

    for i in range(n_states):
        for j in range(i + 1, n_states):
            if commute_times[i, j] < min_ct:
                min_ct = commute_times[i, j]
                index_i = i
                index_j = j

    return min_ct, index_i, index_j


def max_commute_time(matrix_of_mfpts):
    """Returns the max commuting time (round trip time) between all pairs
    of microstates from the matrix of mfpts. It also returns the indexes
    of the pair of microstates involved"""

    matrix_of_mfpts = np.array(matrix_of_mfpts)

    n_states = len(matrix_of_mfpts)
    assert(n_states == len(matrix_of_mfpts[0]) and n_states >= 2)

    # Initial values, arbitrary choice
    index_i = 0
    index_j = 1

    commute_times = matrix_of_mfpts + matrix_of_mfpts.T
    max_ct = commute_times[index_i, index_j]

    for i in range(n_states):
        for j in range(i + 1, n_states):
            if commute_times[i, j] > max_ct:
                max_ct = commute_times[i, j]
                index_i = i
                index_j = j

    return max_ct, index_i, index_j


def fpt_distribution(t_matrix, initial_state, final_state,
                     initial_distrib, max_n_lags=500, lag_time=1):

    # copy everything since they are going to be modified
    tmatrix = np.copy(t_matrix)
    ini_state = list(initial_state)
    f_state = sorted(list(final_state))

    assert(len(ini_state) == len(initial_distrib))

    tmatrix[:, f_state[0]] = np.sum(tmatrix[:, f_state], axis=1)

    for i in range(len(f_state) - 1, 0, -1):
        tmatrix = np.delete(tmatrix, f_state[i], axis=1)
        tmatrix = np.delete(tmatrix, f_state[i], axis=0)
        for j in range(len(ini_state)):
            if f_state[i] < ini_state[j]:
                ini_state[j] = ini_state[j] - 1

    f_state = f_state[0]
    new_n_states = len(tmatrix)
    list_of_pdfs = np.empty((len(ini_state), max_n_lags), dtype=np.float64)
    prevFmatrix = np.empty_like(tmatrix)

    for istateIndex in range(len(ini_state)):
        prevFmatrix = tmatrix.copy()
        # prevFmatrix[:] = tmatrix[:]
        Fmatrix = np.zeros((new_n_states, new_n_states))
        list_of_pdfs[istateIndex, 0] = tmatrix[ini_state[istateIndex], f_state]

        _calc_fmatrix(Fmatrix, tmatrix, prevFmatrix, list_of_pdfs,
                      max_n_lags, ini_state, istateIndex, f_state)

    sum_ = np.sum(initial_distrib)
    initial_distrib = np.array(initial_distrib)

    density = np.sum(initial_distrib[:, None] * list_of_pdfs, axis=0) / sum_

    return density / lag_time


def _calc_fmatrix(Fmatrix, tmatrix, prevFmatrix, list_of_pdfs,
                  max_n_lags, ini_state, istateIndex, f_state):
    for time in range(1, max_n_lags):
        Fmatrix = np.dot(tmatrix,
                         prevFmatrix - np.diag(np.diag(prevFmatrix)))

        list_of_pdfs[istateIndex, time] = \
            Fmatrix[ini_state[istateIndex], f_state]
        prevFmatrix = Fmatrix


if __name__ == '__main__':
    # k= np.array([[1,2],[2,3]])
    n_states = 5

    T = aux.random_markov_matrix(n_states, seed=1)

    pops = aux.pops_from_tmatrix(T)
    print(pops)
    print(markov_mfpts(T, [0], [4]))
    print(directional_mfpt(T, [0], [4], [1]))
    print(mfpts_to_target_microstate(T, 4))
    print()
    print(mfpts_matrix(T))
    print()
    print(min_commute_time(mfpts_matrix(T)))

    #sequence = [1, 'a', 1, 'b', 2.2, 3]

    #newseq, m_dict = aux.map_to_integers(sequence, {})

    # print(newseq)
    # print(m_dict)
