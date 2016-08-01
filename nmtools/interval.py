'''
Created on Jul 28, 2016

@author: esuarez
'''

class Interval:
    '''
    Interval defined as half-open interval [start,end)
    '''
    def __init__(self, range):
        self.range = range

    def __contains__(self,item):
        if self.range[0] != self.range[1]:
            return self.range[0] <= item and item < self.range[1]
        else:
            return item == self.range[0]