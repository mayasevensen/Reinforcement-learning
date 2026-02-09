# Code for problem assignment 2

The model for the problem assignment is provided in the file *mountain.py*. 

## Instantiating the models
Once you import the file, you can instantiate the model:
- **Mountain_one()**
Notice that the model does NOT follow the standard interface of *Gymnasium*. Since we will NOT *run the environment* but *interact with the model* of the environment, you will not need to actually run your agent through episodes in the environment. The model can be instantiated by calling the constructor above.

## Interacting with the models
The main methods the model provide for your task are:
- *get_map()*: returning a map of the mountain and its roughness in the shape of a matrix. You can print it using *pyplot.imshow()*.
- *get_reward(state,action)*: returning the reward from a given state taking a certain action.

The model follows the following encoding:
- *states* are represented as a tuple *(i,j)*, where the first index encodes the row and the second index encodes the column. Becuase of the shape of the gridworld, *i* must be greater or equal to *0* and smaller or equal to *30*; *j* must be greater or equal to *0* and smaller or equal to *99*. 
- *actions* are encoded by the strings *'forward', 'upforward', 'downforward'*.

**IMPORTANT:** Beware of the ordering of rows and columns!

