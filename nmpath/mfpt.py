'''
Created on Jul 29, 2016

'''
import numpy as np
from copy import deepcopy
import auxfunctions as aux


def markov_mfpts(transition_matrix, stateA, stateB):
    '''Computes the mean first passage times A->B and B->A
    from a markov model. The target state is not absorbing (no ss)
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
    return non_markov_mfpts(auxiliar_matrix, stateA, stateB)


def non_markov_mfpts(nm_transition_matrix, stateA, stateB):
    '''Computes the mean first passage times A->B and B->A where
    from a non-markovian model.
    The shape of the transition matrix should be (2*n_states, 2*n_states)
    '''
    labeled_pops = aux.pops_from_tmatrix(nm_transition_matrix)
    # labeled_pops = solveMarkovMatrix(nm_transition_matrix)

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

    return dict(mfptAB=mfptAB, mfptBA=mfptBA)


def directional_mfpt(transition_matrix, stateA, stateB, ini_probs=None):
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
    final_state = sorted(list(stateB))

    assert(lenA == len(ini_probs))

    for i in range(lenB - 1, -1, -1):
        t_matrix = np.delete(t_matrix, final_state[i], axis=1)
        t_matrix = np.delete(t_matrix, final_state[i], axis=0)
        for j in range(lenA):
            if final_state[i] < ini_state[j]:
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

    return mfptAB


if __name__ == '__main__':
    # k= np.array([[1,2],[2,3]])
    n_states = 5

    T = aux.random_markov_matrix(n_states)

    pops = aux.pops_from_tmatrix(T)
    print(pops)
    print(markov_mfpts(T, [0], [4]))
    print(directional_mfpt(T, [0], [4], [1]))

    sequence = [1, 'a', 1, 'b', 2.2, 3]

    newseq, m_dict = aux.map_to_integers(sequence, {})

    print(newseq)
    print(m_dict)
