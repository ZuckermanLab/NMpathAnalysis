#!/usr/bin/env python

import numpy as np
import random


class Interval:
    '''
    Interval defined as half-open interval [start,end)
    '''
    def __init__(self,start,end):
        self.start = start
        self.end = end

    def __contains__(self,item):
        return self.start <= item and item < self.end


class Trajectory:
    '''
    Continuous trajectory as a numpy array, also stores the states A and B as intervals
    Right now is only designed for trajectories projected in one dimension
    '''
    def __init__(self, trajectory):
        self.trajectory = np.array(trajectory)
    
    def mfpts(self, stateA = None, stateB = None): 
        #mean first passage times calculation
        if (stateA is None) or (stateB is None):
            raise Exception('The final and initial states have to be defined to compute the MFPT')
        
        previous_color = "Unknown"
        state = "Unknown"
        passageTimeAB=[]
        passageTimeBA=[]
        fpt_counter = 0  # first passage time counter
        
        for snapshot in self.trajectory:
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
            mfptAB = 'NaN'
            std_err_mfptAB = 'NaN'

        try:
            mfptBA = float(sum(passageTimeBA))/len(passageTimeBA)
            std_err_mfptBA = np.std(passageTimeBA)/np.sqrt(len(passageTimeBA))
        except:
            mfptBA = 'NaN'
            std_err_mfptBA = 'NaN'
        
        return mfptAB, std_err_mfptAB, mfptBA, std_err_mfptBA
            
    #def populations(self,binbounds):
        
        
    
def main():
    test_trajectory = [random.random()*100 for i in range(100000)]
    #print(test_trajectory)
    
    stateA = Interval(0,5)
    stateB = Interval(95,100)
    
    t = Trajectory(test_trajectory)
    
    histogram,edges = np.histogram(test_trajectory, [i*5 for i in range(20)], density = True)
    
    
    print(t.mfpts(stateA, stateB))
    
    print(histogram.sum())

    #print(np.array(map(square,test_trajectory)))
    
if __name__ == '__main__':
    main()
        
    
        
    