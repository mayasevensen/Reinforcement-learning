
import numpy as np

def sample_new_means():
    return np.random.randint(0,6,size=3)

class Bandits():
    def __init__(self,k,mu,sigma):   
        self.k = k
        self.means = np.random.normal(mu,sigma,k)
        self.variances = np.ones(k)
        self._step = 0
        self.state = None

    def reset(self):
        """
        return (observation, reward, terminated, truncated, info)
        """
        self.state = None
        return self._get_obs(), 0, False,False,  self._get_info(), # observation, reward, terminated, truncated, info

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {"steps": self._step}
    
    def get_optimal_action(self):
        return np.argmax(self.means) 
      
    def step(self,action:int):
        """
        input: action
        return (observation, reward, terminated, truncated, info)
        """     
        self._step +=1 
        reward = np.random.normal(self.means[action],self.variances[action])
        return self._get_obs(), reward, True, False, self._get_info()
    


class Bandits_final(Bandits):
    def __init__(self, gene:int = 0):
        self.k = 3
        self.means = sample_new_means()
        self.variances = np.array([1., 1., 1.])
        self._step = 0
        self.state = None

    def step(self,action:int):
        """
        input: action
        return (observation, reward, terminated, truncated, info)
        """
        if(self._step % 200 == 0):
            if(np.random.rand() > .5):
                self.means = sample_new_means()
        self._step +=1 
        reward = np.random.normal(self.means[action],self.variances[action])
        return self._get_obs(), reward, True, False, self._get_info()
    
        