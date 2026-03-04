
# Code for problem assignment 4

The code for problem assignment 4 relies both on custom code and *gymnasium* environments.

The environments have been tested and run on different architectures, but if you find problem with rendering, please get in touch with the TA Morten Blørstad.


## Instantiating the *MountainCart* environment

In order to setup the *MountainCart* environment, you will have to load it from the *gymnasium* library:

```
import gymnasium

env = gymnasium.make('MountainCar-v0', max_episode_steps = 1000)
```
Please, notice that you are requested to set the maximum number of steps to 1000. You can then interact with the environment *env* as a standard *gymnasium* environment.

## Instantiating the *skyscraper* environment

In order to setup the *skyscraper* environment,  you will have to perform the following steps:
1. Unzip *skyscraper.zip*
2. Move into the directory containing the *skyscraper/* folder
3. From the prompt, run the command  ```pip install -e skyscraper```

You will then be able to instantiate the environment as:
```
import gymnasium
import skyscraper

env = gymnasium.make('skyscraper/GridWorld-v0')
```
You can then interact with the environment *env* as a standard *gymnasium* environment.

