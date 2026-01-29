# Code for problem assignment 1

The environments for the problem assignment are provided in the file *bandits.py*. 

## Instantiating environments
Once you import the file, you can instantiate four environments:
- **Bandits_one()**
- **Bandits_two()**
- **Bandits_three()**
- **Bandits_four()**
These environments follows the standard interface of *Gymnasium*, but they do not need to be instantiated through the *gym.make()* command. It is sufficient to call the constructors above to instantiate and interact with the bandit environment.

## Interacting with the environment
As in the standard *Gymnasium* interface, each environment has the following methods:
- *reset()*: restarting the bandit and returning the outputs *observation, reward, terminated, truncated, info*
- *step(action)*: receiving an action, processing it, and returning the outputs *observation, reward, terminated, truncated, info*

The bandit environment also exposes another method:
- *get_optimal_action()*: returning the true optimal action. This method should not be used during learning, as the agent is not supposed to have knowledge of the true optimal action. This method should be used when computing the regret of your agent.

## Training the bandit agents
Your task is to develop agents that can interact with the bandits environment and solve them.
