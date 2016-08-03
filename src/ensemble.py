#!/usr/bin/env python

import numpy as np
import random
from copy import deepcopy
import networkx as nx

from NMpathAnalysis.nmtools.interval import Interval
from NMpathAnalysis.nmtools.functions import check_shape, weighted_choice

class Ensemble:
    '''
    Stores a list of space-continuous trajectories.
    '''
    
    def __init__(self, trajectory = None, verbose = False, dtype ='float32'):
        '''
        trajectory:      is a single trajectory that can be used to instantiate      
        '''
        if trajectory is None:
            self.trajectories = []
            self.n_variables = 0
            if verbose: print('\nEmpty ensemble generated')
        else:
            trajectory = np.array(trajectory, dtype = dtype)
            n_snapshots, n_variables = check_shape(trajectory)
            self.n_variables = n_variables
            self.trajectories = [ trajectory ]
            
            if verbose:
                print('\nSingle trajectory was read with shape {}'.format(trajectory.shape) )
                print('n_snapshots = {}, n_variables = {}'.format(n_snapshots, self.n_variables))
    
    def add_trajectory(self, trajectory):
        '''
        Add a trajectory to the ensemble
        '''
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)

        _n_snapshots, _n_variables = check_shape(trajectory)

        if self.n_variables == 0: # Empty ensemble
            self.trajectories = [trajectory] 
        else:
            if self.n_variables != _n_variables:
                raise Exception('All the trajectories in the same ensemble must have the same number of variables')
            else:
                self.trajectories.append(trajectory)
                      
    def __len__(self):
        #returns the number of trajectories in the ensemble
        return len(self.trajectories)
    
    def __add__(self, other):
        ensemble_sum = deepcopy(self)
        
        for traj in other.trajectories:
            ensemble_sum.add_trajectory(traj)
            
        return ensemble_sum
         
    def __iadd__(self, other):
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
        
        for traj in self.trajectories:
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
        
    def _count_matrix(self, n_states = None, map_function = None):
        if (map_function is None) or (n_states is None):
            raise Exception('The number of states and a map function have to be given as argument')

        count_matrix = np.zeros((n_states, n_states))
        
        for traj in self.trajectories:
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
    def __init__(self, trajectory = None, verbose = False, dtype = 'int32'):
        super().__init__(trajectory, verbose, dtype)
        if (self.n_variables !=1) and (self.n_variables != 0): # In this case is just an index
            raise Exception('A discrete trajectory must have a one-dimensional index/variable unless is empty')
    
    @classmethod
    def from_ensemble(cls, ens, map_function = None , dtype = 'int32'):
        '''
        Build a DiscreteEnsemble from an ensemble object or a single trajectory
        '''
        if map_function is None:
            raise Exception('A map function has to be given as argument')
        
        discrete_ensemble = cls()
        
        if isinstance(ens, Ensemble):
            #it is an Ensemble object
            for traj in ens.trajectories:
                d_traj = np.array([], dtype = dtype)
                for snapshot in traj:
                    d_traj = np.append(d_traj, [map_function(snapshot)], axis = 0)
                discrete_ensemble.add_trajectory(d_traj)
        else:
            # it is a single trajectory or array
            d_traj = []
            for snapshot in ens:
                d_traj += [map_function(snapshot)]
            d_traj = np.array(d_traj, dtype = dtype)
            discrete_ensemble.add_trajectory(d_traj)
            

        return discrete_ensemble
    
    @classmethod
    def from_transition_matrix(cls, transition_matrix, sim_length = None, initial_state = 0, dtype = 'int32'):
        '''
        Generates a discrete ensemble from the transition matrix
        '''
        if sim_length is None:
            raise Exception('The simulation length must be given')
         
        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)
        
        n_states = len(transition_matrix)
        assert(n_states == len(transition_matrix[0]))
        
        current_state = initial_state
        discrete_traj = np.array([initial_state],dtype=np.int32)
        
        for i in range(sim_length):
            next_state = weighted_choice([k for k in range(n_states)],transition_matrix[current_state,:])
            discrete_traj = np.append(discrete_traj, [next_state], axis = 0)
            current_state = next_state
            
        return cls(discrete_traj)


    def _count_matrix(self, n_states):

        count_matrix = np.zeros((n_states, n_states))
        
        for traj in self.trajectories:
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

class DiscretePathEnsemble(DiscreteEnsemble):

    @classmethod
    def from_transition_matrix(cls, transition_matrix, ini_state, final_state, n_paths = 1000, ini_pops = None,  max_iters = 1000000000, dtype = 'int32'):
        '''
        Construct a path ensemble from a transition matrix
        
        ini_state:        list, intitial state
        
        final_state:      list, final state
        
        ini_pops:         list or label, probability distribution over the 
                          initial state used to generate the path
                          
                          possible values:
                          ----------------
                          a)  None
                              Use a uniform distribution over the states in ini_state
                              
                          b) 'ss'
                              Use the steady state solution (first eigenvector) of
                              the transtion matrix
                              
                          c) list
                              A list with the explicit values of the populations in ini_state 
                              that should be used to generate the ensemble
        '''
        
        if ini_pops is None:
            ini_pops = [1/float(len(ini_pops)) for i in range(len(ini_pops))]
        elif ini_pops == 'ss':
            raise NotImplementedError('Sorry: not yet implemented')
        
        
        cls.transtion_matrix = transition_matrix
        cls.ini_state = ini_state
        cls.final_state = final_state
        cls.ini_pops = ini_pops
        cls.max_iters = max_iters
        cls.n_paths = n_paths

        n_states = len(transition_matrix)
        assert(n_states == len(transition_matrix[0]))
        
        d_trajectories = []
        
        for i in range(n_paths):
            current_state = weighted_choice(ini_state, ini_pops) #Initial state
            path = [current_state]
            
            for j in range(max_iters):
                next_state = weighted_choice([k for k in range(n_states)],transition_matrix[current_state,:])
                path += [ next_state ]
                current_state = next_state
                if j+1 == max_iters:
                    print('\nWARNING: max iteration reached when generating the path ensemble, consider to increase max_iters')
                if (current_state in final_state): break
                
            path = np.array(path,dtype = dtype)
            d_trajectories += [path]
        
        cls.trajectories = d_trajectories
        
        return cls

#     def fundamental_sequences(self, _transition_matrix = None):
#         
#         if _transition_matrix is None:
#             transtion_matrix = self.transition_matrix
#         
#         for path in self.trajectories:
            
        
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
        dx = np.random.uniform(-10,10)
        if (x + dx) in I:
            x = x + dx
        mc_traj.append(x)
    return np.array(mc_traj)
        
def simple_mapping(x):
    return int(x/5)

def simple_mapping2(x):
    return int(x/10)
    

def main():

    test_trajectory0 = mc_simulation(10000)
    test_trajectory1 = mc_simulation(10000)
    test_trajectory2 = mc_simulation(10000)

    stateA = [0,10]
    stateB = [90,100]
    ensemble0 = Ensemble(test_trajectory0)
    print('\nmfpts from the continuous simulation: given t0')
    print(ensemble0.mfpts(stateA, stateB))
    print('\nNum of trajs: ',len(ensemble0))
    
    ensemble0.add_trajectory(test_trajectory1)
    print('\nNum of trajs: ',len(ensemble0))
    
    ensemble2 = Ensemble(test_trajectory2,verbose=True)  
    
    ensemble_tot = ensemble0 + ensemble2
    print('\nNum of trajs: ',len(ensemble_tot))
    print(np.array(ensemble_tot.trajectories).shape)
    
    
    

if __name__ == '__main__':
    main()
        
    
        
    
