
# Code for problem assignment 5

The code for problem assignment 5 relies on custom code. The environment has been tested and run on different architectures, but if you find problem with rendering, please get in touch with the TA Morten Blørstad.


## Instantiating the *cookiedisaster* environment

In order to setup the *cookiedisaster* environment,  you will have to perform the following steps:
1. Unzip *cookiedisaster.zip*
2. Move into the directory containing the *cookiedisaster/* folder
3. From the prompt, run the command  ```pip install -e cookiedisaster```

You will then be able to instantiate the environment as:
```
import gymnasium
import cookiedisaster

env = gymnasium.make('cookiedisaster/GridWorld-v0')
```
You can then interact with the environment *env* as a standard *gymnasium* environment.