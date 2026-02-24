import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple
import os


class MountainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, backwards = False):
        self.window_size = 1200  # The size of the PyGame window
  
        self.MAP = np.genfromtxt(os.path.join('mountain','envs','the_hill2_modified.txt'),float)
        self.height = self.MAP.shape[0]  # The height of the grid
        self.width = self.MAP.shape[1]  # The width of the grid
        self.graphics = np.expand_dims(self.MAP, axis = 2)
        self.graphics = np.repeat(self.graphics, 4, axis=2)
        self.graphics*=255
        self._cummlative_reward = 0 

        self._step = 0
        self.graphics[:,:,3] = 1-self.graphics[:,:,3]

        self.start_position = np.array([15, 0])
        #self.graphics[15,0,:] = (173, 216, 230,0.1) #start

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Dict(
                        {"pos": spaces.Box(
                            low=np.array([0, 0]), 
                            high=np.array([self.width - 1, self.height - 1]), 
                            dtype=np.int32
                        )
                          }
                ) 
            }
        )
        
        self.backwards = backwards    
        self.set_action_space(backwards)
       
        
       

        

        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    

    def set_action_space(self, backwards=True):
        """Sets the action space based on the chosen version."""
        if not backwards:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Discrete(8)
        self.action_to_direction = {
            0: np.array([0, 1]),   # forward
            1: np.array([-1, 0]),  # left 
            2: np.array([-1, 1]),  # forward left  
            3: np.array([1, 0]),   # right 
            4: np.array([1, 1]),   # forward right 
            5: np.array([-1, -1]), # backward left
            6: np.array([1, -1]),  # backward right
            7: np.array([0, -1]),  # backward
        }

        

    
    def set_start_position(self,position:Tuple[int,int] = (15, 0)):
        self.start_position = np.array([position[0], position[1]])

    def _get_obs(self):
        return {"agent": {"pos": self._agent_location}}


    def _get_info(self):
        return {"distance": abs(self._agent_location[1] - self._target_location[1]), "steps": self._step}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = self.start_position
        self._step = 0
        self._cummlative_reward = 0

        self._target_location = np.array([15,100])
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    
    def _transition(self, action):
        r = np.random.uniform()
        if action in [5,6,7]: #backwards, 100%
            return self.action_to_direction[action]
        if  action in [1,3]: #sideways
            if r<1/16:
                return self.action_to_direction[action+1] # sideways forward
            else:
                return self.action_to_direction[action] # sideways 

        if  action in [2,4]: #forward sideways
            if r<1/16:
                return self.action_to_direction[action-1] # sideways 
            elif r<1/8:
                return self.action_to_direction[0] # forward
            else:
                return self.action_to_direction[action] # sideways forward
            
        return self.action_to_direction[action]


    
    def step(self, action):
        # Map the action (element of {0,1,2}) to the direction to move in
        
        direction = self._transition(action)
        
        # We use `np.clip` to make sure we don't leave the grid
        next_pos = self._agent_location + direction
        next_pos[0] = np.clip(self._agent_location[0] + direction[0],a_min=0, a_max=30)
        next_pos[1] = np.clip(self._agent_location[1] + direction[1],a_min=0, a_max=100)
    

         
       
        
        self._agent_location = next_pos
        reward = -(self.MAP[self._agent_location[0], self._agent_location[1]])
        self._step -= reward   
        # An episode is done if the agent has reached the target
        reached_goal = self._agent_location[1] >= 99   # np.array_equal(self._agent_location, self._target_location)
        time_out = (self._step >= 80)
        terminated = reached_goal or time_out
        if time_out:
            reward = self._agent_location[1] - 99
        if reached_goal:
            reward = 0
        
            
        self._cummlative_reward += reward
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, time_out, info
    
    def render(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size/3))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size+300, self.window_size/3))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.width
        
    

        
        for row in range(self.height):
            for col in range(self.width):
                color = self.graphics[row, col]
                pygame.draw.rect(
                    canvas,
                    color[:3],  # Use the first three values (RGB) from the color
                    (col * pix_square_size, row* pix_square_size, pix_square_size, pix_square_size),
                )

        # First we draw the target
        # target_rect = pygame.Rect(
        #     self.window_size-pix_square_size,
        #     0,
        #     pix_square_size,
        #     self.window_size//3
        # )
        #pygame.draw.rect(canvas, (0, 255, 0), target_rect)    
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Use the first three values (RGB) from the color
            (self._target_location[1] * pix_square_size, self._target_location[0]* pix_square_size, pix_square_size, pix_square_size),
        )
        
        
        # Now we draw the agent
        agent_center = (
            int((self._agent_location[1] + 0.5) * pix_square_size),
            int((self._agent_location[0] + 0.5) * pix_square_size)
        )
        pygame.draw.circle(canvas, (151,87,43), agent_center, pix_square_size // 3)

        pygame.font.init()
        font = pygame.font.Font(None, 32)
        
        
        score_text = font.render(f'Steps: {self._step}',True, (0, 0, 0))
        pos_text = font.render(f'position: ({self._agent_location[0]}, {self._agent_location[1]})',True, (0, 0, 0))
        cum_reward_text = font.render(f'time: {abs(self._cummlative_reward):.2f}',True, (0, 0, 0))

        canvas.blit(score_text, (self.window_size+5, 10))
        canvas.blit(pos_text, (self.window_size+5, 90))
        canvas.blit(cum_reward_text, (self.window_size+5, 170))


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        