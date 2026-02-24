
# Code for problem assignment 3

The code for problem assignment 3 is made up by two main files:
- *trajectories.pickle*: a file collecting a set of trajectories.
- *mountain.zip*: a zip file containing a gym environment.

## Loading the trajectories
The file *trajectories.pickle* contains trajectories generated on the *mountain* environment using a fixed policy. You can load the trajectories using the commands:
```
import pickle

fd = open('trajectories.pickle','r')
D = pickle.load(fd)
```

## Instantiating the environment
The environment **mountain** provides the environment together with rendering functions relying on the *pygame* library which you can use to visualize the simulation.

### Instantiating mountain
In order to setup the *mountain* environment, you will have to perform the following steps:
1. Unzip *mountain.zip*
2. Move into the directory containing the *mountain/* folder
3. From the prompt, run the command  ```pip install -e mountain```

You will then be able to instantiate the environment as:
```
import gymnasium
import mountain

env = gymnasium.make('mountain/GridWorld-v0')
```
You can then interact with the environment *env* as a standard *gymnasium* environment.




