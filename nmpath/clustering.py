'''
Created on June 29, 2017

@author: esuarez
'''
import numpy as np
from auxfunctions import pops_from_tmatrix, check_tmatrix, random_markov_matrix
from mfpt import mfpts_matrix, min_commute_time
import copy


def merge_microstates_in_tmatrix(transition_matrix, ms1, ms2):
    '''Merge two microstates (ms1 and ms2) in the transition matrix, i.e.,
    returns the transition matrix that we would obtain if the microstates where
    merged befored the estimation of the transition matrix. The transition
    matrix is expected to be a square numpy array'''

    check_tmatrix(transition_matrix)  # it is a valid t_matrix?

    p = pops_from_tmatrix(transition_matrix)
    size = len(transition_matrix)
    final_tmatrix = np.copy(transition_matrix)

    # sum of the columns with indexes ms1 and ms2
    # and saved in the index state1.
    for k in range(size):
        final_tmatrix[k, ms1] += final_tmatrix[k, ms2]

    # weighted sum of the rows
    for k in range(size):
        if (p[ms1] + p[ms2]) != 0.0:
            final_tmatrix[ms1, k] = (p[ms1] * final_tmatrix[ms1, k] +
                                     p[ms2] * final_tmatrix[ms2, k]) / \
                (p[ms1] + p[ms2])

    final_tmatrix = np.delete(final_tmatrix, ms2, axis=1)
    final_tmatrix = np.delete(final_tmatrix, ms2, axis=0)

    return final_tmatrix


def kinetic_clustering_from_tmatrix(transition_matrix, n_clusters=2,
                                    t_cut=float('inf'), ini_clusters=None):
    """Hierarchical agglomeratice kinetic clustering from the commute matrix
    (MFPTs in both directions). On each step, the matrix is recalculated.
    """
    # Check for consistency
    check_tmatrix(transition_matrix)  # it is a valid t_matrix?
    if n_clusters < 2:
        raise ValueError("The final number of clusters should be "
                         "greater than 2")

    n_states = len(transition_matrix)

    new_tmatrix = copy.copy(transition_matrix)

    if ini_clusters is None:
        clusters = [[i] for i in range(n_states)]
    else:
        clusters = copy.copy(ini_clusters)

    mfpt_M = mfpts_matrix(transition_matrix)
    min_t, index_i, index_j = min_commute_time(mfpt_M)

    print("Number of clusters: ", end=" ")
    while (min_t < t_cut) and (len(clusters) > n_clusters):
        # Merge clusters
        clusters[index_i] += clusters[index_j]
        del clusters[index_j]
        print(len(clusters), end=" ")

        # Merge states in the t_matrix
        new_tmatrix = merge_microstates_in_tmatrix(new_tmatrix,
                                                   index_i, index_j)

        # recalculate
        mfpt_M = mfpts_matrix(new_tmatrix)
        min_t, index_i, index_j = min_commute_time(mfpt_M)
    print()

    return clusters


if __name__ == "__main__":
    T = random_markov_matrix(20)

    clusters = kinetic_clustering_from_tmatrix(T, 5)
    print(clusters)
