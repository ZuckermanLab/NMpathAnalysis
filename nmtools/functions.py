'''
Created on Jul 29, 2016

'''
import numpy as np
from random import random

def weighted_choice(list, weights = None):
    '''
    Selects an element from a list given with probability given by
    the variable weights
    '''
    
    size = len(list)
    if weights is not None:
        assert(size == len(weights))
    
    if weights is None:
        probs = np.array([1/float(size) for i in range(size)])
    else:
        probs = np.array(weights)/sum(weights) # just in case they are not normalized
    
    rand = random()
    
    _sum = 0
    for i in range(size):
        if _sum <= rand < _sum + probs[i]:
            choice = i
            break
        else:
            _sum += probs[i]
    
    return list[choice]
        
def check_shape(trajectory):
     
    shape = trajectory.shape
    
    if len(shape) == 1:
        n_snapshots = shape[0] 
        n_variables =1
        if n_variables == 0:
            raise Exception('The shape {} of the trajectory/array given is not as expected'.format(shape))
    elif len(shape) == 2:
        n_snapshots = shape[0]
        n_variables = shape[1]
    else: 
        raise Exception('The shape {} of the trajectory/array given is not as expected'.format(shape))

    return n_snapshots, n_variables
    

# def markov_mfpt_from_fluxes(transition_matrix, ini_state, final_state):
#     '''
#     Computes the mfpt for systems with no absorbing states from 
#     the transtion matrix
# 
#     '''
#     transition_matrix = np.array(transition_matrix)
#     
#     n_states = len(transition_matrix)
#     
#     #pseudo non-markovian matrix (auxiliar_matrix)
#     auxiliar_matrix = np.zeros((2*n_states,2*n_states))
#     
#     for i in range(2*n_states):
#         for j in range(2*n_states):
#             auxiliar_matrix[i,j] = transition_matrix[int(i/2),int[j/2]]
#     
#     for i in range(n_states):
#         for j in range(n_states):
#             if (i in final_state) or (j in final_state):
#                 auxiliar_matrix[2*i,2*j] = 0.0
#             if (i in ini_state) or (j in ini_state):
#                 auxiliar_matrix[2*i+1,2*j+1] = 0.0
#             if (not (j in ini_state)) or (i in ini_state):
#                 auxiliar_matrix[2*i+1,2*j] = 0.0
#             if (not (j in final_state)) or (i in final_state):
#                 auxiliar_matrix[2*i,2*j+1] = 0.0
#                  
#     # the following line is going to return a MARKOVIAN mfpt since the auxiliar
#     # matrix was build from a markovian matrix 
#     return nonmarkov_mfpt_from_fluxes(auxiliar_matrix, ini_state, final_state)
# 
# 
# def nonmarkov_mfpt_from_fluxes(nm_transition_matrix, ini_state, final_state):
#     
#     labeled_pops = pops_from_markov(nm_transition_matrix)
#     
#     flux = 0
#     
#     for i in range(0, 2*n_states, 2):
#         for j in range(2*n_states):
#             if int(j/2) in final_state:
#                 flux += labeled_pops[i] * nm_transition_matrix[i,j]
#     
# 
# def pops_from_markov(rate_matrix):
#     "This a fake function"
#     return [1/float(len(rate_matrix)) for i in range(len(rate_matrix))]

## To test the code    
if __name__ ==  '__main__':
    pass
#     n_var = 2
#     n_snapshots  = 100
#     n_trajs = 1
#     #
#     t = np.array([ [random() for i in range(n_var)] for j in range(n_snapshots)])
#     s = [t for i in range(n_trajs)]
#     s = np.array(s)
#     s = np.array([])
#     print('input shape:', s.shape)
#     
#     shape2 = get_sequence_shape(s, n_var)
#     
#     print("\n n_trajs = {}, n_snapshots = {} and n_var = {}".format(shape2[0],shape2[1],n_var))

    
    
    