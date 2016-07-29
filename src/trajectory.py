#!/usr/bin/env python

import numpy as np
import random

from NMpathAnalysis.nmtools.interval import Interval

class Trajectory:
    '''
    Continuous sequence of the selected collective variables
    Right now is only designed for trajectories projected in one dimension
    '''
    def __init__(self, sequence):
        '''
        A sequence of trajectories is expected
        '''
        self.sequence = np.array(sequence) # a list of arrays is saved
    
    def __len__(self):
        return len(self.sequence)
    
    def mfpts(self, stateA = None, stateB = None): 
        #mean first passage times calculation
        if (stateA is None) or (stateB is None):
            raise Exception('The final and initial states have to be defined to compute the MFPT')
        
        if self.__class__.__name__ == 'Trajectory':
            stateA = Interval(stateA)
            stateB = Interval(stateB)
        else:
            #Other wise stateA and B are consider sets
            pass
        
        previous_color = "Unknown"
        state = "Unknown"
        passageTimeAB=[]
        passageTimeBA=[]
        fpt_counter = 0  # first passage time counter
        
        for snapshot in self.sequence:
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
        
    def count_matrix(self, n_states, map_function = None):
        if (map_function is None) or (n_states is None):
            raise ValueError('The number of states and a map function have to be given as argument')

        previous_state = "Unknown" #Previous state is unknown
        count_matrix = np.zeros((n_states, n_states))
        
        for snapshot in self.sequence:
            current_state = map_function(snapshot)
            if previous_state != "Unknown":
                count_matrix[previous_state, current_state] += 1.0
            previous_state = current_state
        
        return count_matrix
    
    def _mle_transition_matrix(self, n_states, map_function):
        
        count_matrix = self.count_matrix(n_states, map_function)
        transition_matrix = count_matrix.copy()
        
        # Transforming the count matrix to a transition matrix
        for i in range(n_states):
            row_sum = sum(transition_matrix[i,:])
            if row_sum != 0.0:
                transition_matrix[i,:] = transition_matrix[i,:]/row_sum
                
        return transition_matrix
        
class DiscreteTrajectory(Trajectory):
    '''
    Discrete trajectory
    '''
    @classmethod
    def from_continuous_traj(cls, traj, map_function = None):
        if map_function is None:
            raise ValueError('A map function has to be given as argument')
        
        discrete_traj = np.array([],dtype=np.int32)
        
        if isinstance(traj, Trajectory):   
            for snapshot in traj.sequence:
                discrete_traj = np.append(discrete_traj, [map_function(snapshot)], axis = 0)  
        else:
            for snapshot in traj:
                discrete_traj = np.append(discrete_traj, [map_function(snapshot)], axis = 0)

        return cls(discrete_traj)
    
    @classmethod
    def from_transition_matrix(cls, transition_matrix, sim_length = None, initial_state = 0):
        '''
        Generates a trajectory from the transition matrix
        '''
        if sim_length is None:
            raise ValueError('The simulation length must be given')
         
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


    def count_matrix(self, n_states):

        count_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(self.sequence)-1):
            count_matrix[self.sequence[i], self.sequence[i+1]] += 1.0

        return count_matrix
    
    
    def _mle_transition_matrix(self, n_states):

        count_matrix = self.count_matrix(n_states)

        transition_matrix = count_matrix.copy()
        
        # Transforming the count matrix to a transition matrix
        for i in range(n_states):
            row_sum = sum(transition_matrix[i,:])
            if row_sum != 0.0:
                transition_matrix[i,:] = transition_matrix[i,:]/row_sum
                
        return transition_matrix
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
    test_trajectory = mc_simulation(10000)
    #print(test_trajectory)
    
    stateA = [0,10]
    stateB = [90,100]
    
    t = Trajectory(test_trajectory)
    
    T_matrix = t._mle_transition_matrix(10,simple_mapping2)
    #print(T_matrix)

    #histogram,edges = np.histogram(test_trajectory, [i*5 for i in range(20)], density = True)
    print('\nmfpts from the continuous simulation')
    print(t.mfpts(stateA, stateB))
     
    d = DiscreteTrajectory.from_continuous_traj(t, simple_mapping2)
    
    d_from_T = DiscreteTrajectory.from_transition_matrix(T_matrix, 10000)
    
    stateA = [0]
    stateB = [9]
    
    print('\nmfpts from the discrete simulation')
    print(d.mfpts(stateA, stateB))
    
    print('\nmfpts from the simulation -> discrete_sim -> transition_matrix -> discrete simulation')
    print(d_from_T.mfpts(stateA,stateB))

    
if __name__ == '__main__':
    from interval import Interval
    main()
        
    
        
    
