#!/usr/bin/env python

import numpy as np
import random

from NMpathAnalysis.nmtools.interval import Interval
from NMpathAnalysis.nmtools.functions import get_sequence_shape

class Ensemble:
    '''
    Stores a array of space-continuous trajectories (Ensemble).
    If a single trajectory is given, is going to transform it to a list/array of trajs
    with only one element/trajectory.
    '''
    
    def __init__(self, sequence, n_variables = 1):
        
        self.sequence = np.array(sequence) # a list of arrays is saved
        
        if self.sequence.shape[0] == 0:
            # the sequence is an empty list
            self.n_variables = 0
        else:
            n_trajectories, n_snapshots, is_list_of_trajs = get_sequence_shape(self.sequence, n_variables)
            self.n_variables = n_variables
            
            if not is_list_of_trajs:
                self.sequence = np.array([self.sequence])  # make it a list/array of trajs
      
    def __len__(self):
        #returns the number of trajectories in the ensemble
        return len(self.sequence)
    
    def __add__(self,other):
        if (self.n_variables != other.n_variables) and self.n_variables != 0 and other.n_variables != 0:
            raise Exception('To add two not-empty trajectories, the number of variables should be the same')
        
        if (self.sequence.shape[0] == 0) and (other.sequence.shape[0] == 0): # both arrays are empty
            return Ensemble(self.sequence)
        elif (self.sequence.shape[0] == 0) and (other.sequence.shape[0] != 0): # only other is empty
            return Ensemble(other.sequence)
        elif (self.sequence.shape[0] != 0) and (other.sequence.shape[0] == 0):
            return Ensemble(self.sequence)
        else:
            sum_seq = np.append(self.sequence, other.sequence, axis = 0)
            return Ensemble(sum_seq)
        
    def __iadd__(self,other):
        return self.__add__(other)
    
    def mfpts(self, stateA = None, stateB = None): 
        #mean first passage times calculation
        if (stateA is None) or (stateB is None):
            raise Exception('The final and initial states have to be defined to compute the MFPT')
        
        if self.__class__.__name__ == 'Ensemble':
            '''
            The states are considered/transformed-to intervals if the Ensemble
            is a set of continuous trajectories 
            '''
            stateA = Interval(stateA)
            stateB = Interval(stateB)
        
        passageTimeAB=[]
        passageTimeBA=[]
        fpt_counter = 0  # first passage time counter
        
        for traj in self.sequence:
            previous_color = "Unknown"
            state = "Unknown"
            for snapshot in traj:
                #state and color determination
                if snapshot in stateA:
                    state = "A"; color = "A"
                elif snapshot in stateB:
                    state = "B"; color = "B"
                else:
                    state = "Unknown"; color = previous_color
                    
                #passage times
                if (color == "A") or (color == "B"):
                    fpt_counter += 1
                
                if previous_color == "A" and color == "B":
                    passageTimeAB.append(fpt_counter)
                    fpt_counter = 0
                elif previous_color=="B" and color=="A":
                    passageTimeBA.append(fpt_counter)
                    fpt_counter = 0
                elif previous_color == "Unknown" and (color == "A" or color == "B"):
                    fpt_counter = 0
                
                previous_color = color
                
        try:
            mfptAB = float(sum(passageTimeAB))/len(passageTimeAB)
            std_err_mfptAB = np.std(passageTimeAB)/np.sqrt(len(passageTimeAB))
        except:
            print('WARNING: No A->B events observed')
            mfptAB = 'NaN'
            std_err_mfptAB = 'NaN'

        try:
            mfptBA = float(sum(passageTimeBA))/len(passageTimeBA)
            std_err_mfptBA = np.std(passageTimeBA)/np.sqrt(len(passageTimeBA))
        except:
            print('WARNING: No B->A events observed')
            mfptBA = 'NaN'
            std_err_mfptBA = 'NaN'
        
        kinetics = {'mfptAB': mfptAB, 'std_err_mfptAB': std_err_mfptAB, 'mfptBA': mfptBA, 'std_err_mfptBA': std_err_mfptBA}
        
        return kinetics            

    #def populations(self,binbounds):
        
    def _count_matrix(self, n_states, map_function = None):
        if (map_function is None) or (n_states is None):
            raise ValueError('The number of states and a map function have to be given as argument')

        count_matrix = np.zeros((n_states, n_states))
        
        for traj in self.sequence:
            previous_state = "Unknown" 
            for snapshot in traj:
                current_state = map_function(snapshot)
                if previous_state != "Unknown":
                    count_matrix[previous_state, current_state] += 1.0
                previous_state = current_state
        
        return count_matrix
    
    def _mle_transition_matrix(self, n_states, map_function):
        
        count_matrix = self._count_matrix(n_states, map_function)
        transition_matrix = count_matrix.copy()
        
        # Transforming the count matrix to a transition matrix
        for i in range(n_states):
            row_sum = sum(transition_matrix[i,:])
            if row_sum != 0.0:
                transition_matrix[i,:] = transition_matrix[i,:]/row_sum
                
        return transition_matrix
        
class DiscreteEnsemble(Ensemble):
    '''
    Discrete trajectory
    '''
    def __init__(self, sequence):
        n_variables = 1 #the state of a discrete trajectories must have only one index
        self.sequence = np.array(sequence,dtype = np.int32) # a list of arrays is saved
        n_trajectories, n_snapshots, is_list_of_trajs = get_sequence_shape(self.sequence, n_variables)
        
        if not is_list_of_trajs:
            self.sequence = np.array([self.sequence]) # make it a list
        self.n_variables = n_variables
    
    @classmethod
    def from_ensemble(cls, ens, map_function = None):
        '''
        Build a DiscreteEnsemble from an ensemble object or a single trajectory
        '''
        if map_function is None:
            raise Exception('A map function has to be given as argument')
        
        discrete_sequence = [] # list of ensembles
        
        if isinstance(ens, Ensemble):
            #it is an Ensemble object
            for seq in ens.sequence:
                temp_traj = np.array([], dtype=np.int32)
                for snapshot in seq:
                    temp_traj = np.append(temp_traj, [map_function(snapshot)], axis = 0)
                discrete_sequence += [temp_traj]
        else:
            # it is a single trajectory or array
            for snapshot in ens:
                discrete_sequence += [map_function(snapshot)]

        return cls(discrete_sequence)
    
    @classmethod
    def from_transition_matrix(cls, transition_matrix, sim_length = None, initial_state = 0):
        '''
        Generates a trajectory from the transition matrix
        '''
        if sim_length is None:
            raise Exception('The simulation length must be given')
         
        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)
        
        n_states = len(transition_matrix)
        assert(len(transition_matrix) == len(transition_matrix[0]))
        
        current_state = initial_state
        discrete_traj = np.array([initial_state],dtype=np.int32)
        
        for i in range(sim_length):
            rand_num = random.random()
            partial_sum = 0.0
            
            for j in range(n_states):
                if partial_sum <= rand_num < ( partial_sum + transition_matrix[current_state,j] ):
                    current_state = j
                    break
                else:
                    partial_sum += transition_matrix[current_state,j]
                    
            discrete_traj = np.append(discrete_traj, [current_state], axis = 0)
        
        return cls(discrete_traj)


    def _count_matrix(self, n_states):

        count_matrix = np.zeros((n_states, n_states))
        
        for traj in self.sequence:
            for i in range(len(traj)-1):
                count_matrix[traj[i], traj[i+1]] += 1.0

        return count_matrix
    
    
    def _mle_transition_matrix(self, n_states):

        count_matrix = self._count_matrix(n_states)

        transition_matrix = count_matrix.copy()
        
        # Transforming the count matrix to a transition matrix
        for i in range(n_states):
            row_sum = sum(transition_matrix[i,:])
            if row_sum != 0.0:
                transition_matrix[i,:] = transition_matrix[i,:]/row_sum
                
        return transition_matrix

class PathEnsemble(Ensemble):
    pass
    

'''
------------------------------------------------------
EVERYTHING FROM HERE IS JUST TO TEST THE CODE ABOVE
--------------------------------------------------------
'''
#just to test the code
def mc_simulation(numsteps):
    x = 5
    I = Interval([0,100])
    mc_traj = []
    
    for i in range(numsteps):
        dx = random.uniform(-10,10)
        if (x + dx) in I:
            x = x + dx
        mc_traj.append(x)
    return np.array(mc_traj)
        
def simple_mapping(x):
    return int(x/5)

def simple_mapping2(x):
    return int(x/10)
    

def main():
    #test_trajectory = [random.random()*100 for i in range(100000)]
    test_trajectory0 = mc_simulation(10000)
    test_trajectory1 = mc_simulation(10000)
    test_trajectory2 = mc_simulation(10000)
    #print(test_trajectory)
    #################################################
    stateA = [0,10]
    stateB = [90,100]
    ensemble0 = Ensemble(test_trajectory0)
    print('\nmfpts from the continuous simulation: given t0')
    print(ensemble0.mfpts(stateA, stateB))

    ##################################################
    list_of_trajs = [test_trajectory1, test_trajectory2]
    ensemble_tot = Ensemble(list_of_trajs)
    #transition_matrix = ensemble_tot._mle_transition_matrix(10,simple_mapping2)
    print('\nmfpts from the continuous simulation: given [t1,t2]')
    print(ensemble_tot.mfpts(stateA, stateB))
    
    #######################################
    e1 = Ensemble(test_trajectory1)
    e2 = Ensemble(test_trajectory2)
    
    ensemble_tot = e1 + e2
    
    print('\nmfpts from the continuous simulation: given Ensemble(t1) + Ensemble(t2)')
    print(ensemble_tot.mfpts(stateA, stateB))
    
    ##########################################
    d_ensemble_0 = DiscreteEnsemble.from_ensemble(test_trajectory0, simple_mapping2)
    
    stateA = [0]
    stateB = [9]
    
    print('\nmfpts from the discrete simulation, optained from t0')
    print(d_ensemble_0.mfpts(stateA, stateB))
    
    
    #############################################

    
    
    #d_from_T = DiscreteEnsemble.from_transition_matrix(T_matrix, 10000)
    

    

    
    #print('\nmfpts from the simulation -> discrete_sim -> transition_matrix -> discrete simulation')
    #print(d_from_T.mfpts(stateA,stateB))

    
if __name__ == '__main__':
    from interval import Interval
    main()
        
    
        
    
