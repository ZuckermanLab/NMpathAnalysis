'''
Created on Jul 28, 2016

@author: esuarez
'''
import numpy as np

class Interval:
    '''
    Interval are in general defined as half-open interval [start,end),
    in any case, in each dimension the interval is specified using a list [a,b]
    where a < b
    
    - For 1D interval a single list in the for [a,b] has to be given 
    
    - A list of lists [[a,b],[c,d],...] are used for a n-dimensional 
        intervals, one for each dimension, i.e, len(inteval) = n_variables 
        
    - A list-of-lists-of-lists for the union of n-dimensional intervals' 
    
    '''
    
    def __init__(self, range):
        self.range = range
            

    def __contains__(self,item):
        
        shape = np.array(self.range).shape
        
        if len(shape) == 1: # 1D interval
            if self.range[0] != self.range[1]:
                try:
                    return self.range[0] <= item and item < self.range[1]
                except: 
                    raise Exception('The given interval has not the expected shape')
            else:
                return item == self.range[0]
            
        elif len(shape) == 2: # n-dimensional interval
            if len(self.range) != len(item):
               raise Exception('The given interval has not the expected shape')
            else:
                if all([  (item[i] in Interval(self.range[i])) for i in range(len(item))   ]):
                    return True  
                
        elif len(shape) == 3: # union of n-dimensional intervals
            if any( [(item in Interval(self.range[i])) for i in range(len(self.range))]   ):
                return True
            else:
                return False
        else:
            raise Exception('The given interval has not the expected shape')
        

if __name__ == '__main__':
    I = Interval([[2,3],[1,2]])
    I = Interval([ [[2,3],[1,2]] ,[[1,2],[0,1]] ])
    #I = Interval((1,2))
    
    if [1.5,2] in I:
        print('yes')
    else:
        print('no')