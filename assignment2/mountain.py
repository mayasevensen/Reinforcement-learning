
import numpy as np
from typing import Tuple
import os


script_dir = os.path.dirname(__file__)



class Mountain():
    def __init__(self):
        
        self.actions = ["upforward", "forward", "downforward"]
        self.actions_to_dir = {"upforward":(-1,1), "forward":(0,1),  "downforward":(1,1)} # in terms of (i,j)/(y,x)
        
    def get_map(self)->np.ndarray:
        """
        return: map, a 100 by 31 matrix containing the terrain roughness at each position.  
        """
        return self.map

    def get_direction(self,action:str)->Tuple[int,int]:
        """
        input: 
            action: "upforward", "forward" or "downforward"
        return: (delta i, delta j) or equivalently (delta y, delta x)
        """
        self._check_action(action)
        return self.actions_to_dir[action]
        
        
    def _check_state(self,state:Tuple[int,int]):
        i,j =state
        if (i<0 or i>30):
            raise ValueError(f'Row position, {i} is beyond the gridworld')
        if (j<0 or j>99):
            raise ValueError(f'Column position, {j}  is beyond the gridworld')

    def _check_action(self,action:str):
        if action not in self.actions:
            raise ValueError('Action is wrongly specified. Choose from {0}'.format(self.actions))
    
    def next_state(self,state:Tuple[int,int],action:str):
        self._check_state(state)
        self._check_action(action)

        if action == "forward":
            if(state[1] <= 98): return (state[0],state[1]+1)
            else: return state
        if action == "upforward":
            if(state[0]>0 and state[1] <= 98): return (state[0]-1,state[1]+1)
            elif (state[0]==0 and state[1] <= 98): return (state[0],state[1]+1)
            else: return state
        if action == "downforward":
            if(state[0]<30 and state[1] <= 98): return (state[0]+1,state[1]+1)
            elif (state[0]==30 and state[1] <= 98): return (state[0],state[1]+1)
            else: return state

    def get_time(self,state:Tuple[int,int])->float:
        """
        We want to minimize accumulated time getting accross the terrain. The time it takes the sled to cover one unit of distance on the hill depends directly on how rough
        roughness is equivalent to the time.  

        input: 
            state: position (i,j) or equivalently (y,x)
        return: time, the roughness for the position
        """
        self._check_state(state)
        i,j =state
        if(j <= 98): return self.map[i,j]
        else: return 0

    def get_reward(self,state:Tuple[int,int],action:str)->float:
        """
        We want to minimize accumulated time  getting accross the terrain. The time it takes the sled to cover one unit of distance on the hill depends directly on how rough
        roughness is equivalent to the time. This is equivalent to maximizing the accumulated negative time

        input: 
            state: position (i,j) or equivalently (y,x)
            action: "upforward", "forward" or "downforward"
        return: reward, the negative roughness for the position
        """
        self._check_state(state)
        self._check_action(action)
        next_state = self.next_state(state,action)
        return -self.get_time(next_state)

class Mountain_one(Mountain):
    def __init__(self):
        super().__init__()
        file_path = os.path.join(script_dir, "the_hill.txt")
        self.map = np.genfromtxt(file_path, dtype=float) 

class Mountain_two(Mountain):
    def __init__(self):
        super().__init__()
        
        file_path = os.path.join(script_dir, "the_hill2.txt")
        self.map = np.genfromtxt(file_path, dtype=float) 



print("adad", os.path.dirname(__file__))