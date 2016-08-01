'''
Created on Jul 29, 2016

'''
import numpy as np
from random import random as r

def get_sequence_shape(sequence, n_variables):
    '''
    "Automatically" figures out if the ensemble given is 
    a set of trajectories or a single ensemble
    '''
    shape_error_message = '\nThe shape of the array given is not as expected, the shape should be:\n' + \
                      'For list of trajectories -> (n_trajectories, n_snapshots, n_variables) if n_variables > 1, or \n' +\
                      'For list of trajectories -> (n_trajectories, n_snapshots) if n_variables = 1, or\n' + \
                      'For a ensemble -> (n_snapshots, n_variables) if n_trajectories = 1 and n_variables > 1, or\n' + \
                      'For a ensemble -> (n_snapshots,) if n_trajectories = 1 and n_variables = 1 \n'
    
    shape = sequence.shape
    
    is_list_of_trajs = False
    
    if len(shape) > 3:
        raise Exception(shape_error_message)
    
    if n_variables == 1:
        if len(shape) == 2:
            n_trajs = shape[0]
            n_snapshots = shape[1]
            is_list_of_trajs = True
            return n_trajs, n_snapshots, is_list_of_trajs
        elif len(shape) == 1:
            n_trajs = 1
            n_snapshots = shape[0]
            return n_trajs, n_snapshots, is_list_of_trajs
        else:
            raise Exception(shape_error_message)
    elif n_variables > 1:
        if len(shape) == 3:
            n_trajs = shape[0]
            n_snapshots = shape[1]
            assert(n_variables == shape[2])
            is_list_of_trajs = True
            return n_trajs, n_snapshots, is_list_of_trajs
        elif len(shape) == 2:
            n_trajs = 1
            n_snapshots = shape[0]
            assert(n_variables == shape[1])
            return n_trajs, n_snapshots, is_list_of_trajs
    

## To test the code    
if __name__ ==  '__main__':
    n_var = 2
    n_snapshots  = 100
    n_trajs = 1
    #
    t = np.array([ [r() for i in range(n_var)] for j in range(n_snapshots)])
    s = [t for i in range(n_trajs)]
    s = np.array(s)
    s = np.array([])
    print('input shape:', s.shape)
    
    shape2 = get_sequence_shape(s, n_var)
    
    print("\n n_trajs = {}, n_snapshots = {} and n_var = {}".format(shape2[0],shape2[1],n_var))
    
    
    