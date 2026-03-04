import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple
import os
import math


def draw_arrow(screen, color, start, end, width, head_width, head_length,flip_direction= False):
        
        if flip_direction:
            start, end = end, start  # Swap start and end points
        # Draw the shaft of the arrow
        pygame.draw.line(screen, color, start, end, width)
        
        # Calculate the direction of the arrow
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)
        
        # Calculate points for the arrowhead
        right_side = (end[0] - head_length * math.cos(angle - math.pi / 6),
                    end[1] - head_length * math.sin(angle - math.pi / 6))
        left_side = (end[0] - head_length * math.cos(angle + math.pi / 6),
                    end[1] - head_length * math.sin(angle + math.pi / 6))
        
        # Draw the arrowhead
        pygame.draw.polygon(screen, color, [end, right_side, left_side])


def get_wind_color(wind_speed):
    # Define wind speed thresholds (in units per your dataset, e.g., meters per second)
    thresholds = [1,2,3]  # Example thresholds
    # Define corresponding colors from light blue to dark blue
    colors = [
        #(173, 216, 230),  # Light blue
        (135, 206, 250),  # Sky blue
        #(0, 191, 255),    # Deep sky blue
        (30, 144, 255),   # Dodger blue
        (0, 0, 255),      # Blue
        #(0, 0, 139)      # Dark blue
        #(25, 25, 112)     # Midnight blue
    ]
    
    # Assign a color based on wind speed
    for i, threshold in enumerate(thresholds):
        if wind_speed <= threshold:
            return colors[i]
    return colors[-1]

class SkyscraperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 1200  # The size of the PyGame window
  
        self.MAP = np.genfromtxt(os.path.join('skyscraper','envs','skyline.txt'),float)
        self.vertical_wind = np.genfromtxt(os.path.join('skyscraper','envs','vertical_wind.txt'),float)
        self.horizontal_wind = np.genfromtxt(os.path.join('skyscraper','envs','horizontal_wind.txt'),float)
 
       
        self.height = self.MAP.shape[0]  # The height of the grid
        self.width = self.MAP.shape[1]  # The width of the grid
        self.graphics = np.expand_dims(self.MAP, axis = 2)
        self.graphics = np.repeat(self.graphics, 4, axis=2)
        self.graphics*=255
        self._cummlative_reward = 0 

        self._step = 0
        self.graphics[:,:,3] = 1-self.graphics[:,:,3]
        


        self.start_position = np.array([13, 5],dtype=int)
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

        # We have 3 actions, corresponding to "leftforward", "straigth", "left", "rightforward"
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_to_direction = {
            0: np.array([0, -3]),#left
            1: np.array([0, 3]) #right 
        }
       

        

        

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
    


    def _get_obs(self):
        return {"agent": {"pos": self._agent_location}}


    def _get_info(self):
        return {"distance": abs(self._agent_location[1] - self._target_location[1]), "steps": self._step}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = self.start_position
        self._agent_location.dtype = int
        self._step = 0
        self._cummlative_reward = 0

        self._target_location = np.array([14,54])
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _crashed(self,curr_i:int ,curr_j:int, next_i:int,next_j:int):
        delta_i = abs(next_i - curr_i)
        delta_j = abs(next_j - curr_j)
        n = delta_i + delta_j + 1

        if next_i > curr_i:
            di = 1
        else:
            di = -1

        if next_j > curr_j:
            dj = 1
        else:
            dj = -1

        epsilon = delta_i - delta_j
        delta_i *= 2
        delta_j *= 2

        while n > 0:
            if self.MAP[curr_i, curr_j] == 0:
                # Uncomment the line below to print current coordinates and map value
                # print(curr_i, curr_j, MAP[curr_i, curr_j])
                return True

            if epsilon > 0:
                curr_i += di
                epsilon -= delta_j
                n -= 1
            elif epsilon < 0:
                curr_j += dj
                epsilon += delta_i
                n -= 1
            elif epsilon == 0:
                curr_i += di
                curr_j += dj
                epsilon += delta_i - delta_j
                n -= 2

        return False


    
    def step(self, action):
        crashed =reached_goal = False
        
        if self._agent_location[0] < 0 or self._agent_location[0] > 31 or self._agent_location[1] < 0 or self._agent_location[1] > 63:
            crashed = True
        
        if not crashed:
            direction = self.action_to_direction[action]
            i= int(self._agent_location[0]); j = int(self._agent_location[1])
            g = 1 #gravity
            v_wind = self.vertical_wind[i,j ] 
            h_wind = self.horizontal_wind[i,j ]
            
            # We use `np.clip` to make sure we don't leave the grid
            next_pos = self._agent_location + direction + np.array([v_wind+g,h_wind])
            
        

            self._step += 1    
            reward = 0

            if next_pos[0] <0 or  next_pos[0]> 31 or next_pos[1]<0 or next_pos[1]>63:
                crashed = True
            else:
                crashed = self._crashed(i,j,int(next_pos[0]),int(next_pos[1]))

        if not crashed:
            self._agent_location = next_pos
            # An episode is done if the agent has reached the target
            reached_goal= np.array_equal(self._agent_location, self._target_location)


        terminated = reached_goal
        if crashed:
            self._agent_location = self.start_position
            reward = 0

        if reached_goal:
            reward =1
        self._cummlative_reward +=reward
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        if reward >0:
            print(observation, reward)
        return observation, reward, terminated, False, info
    
    def render(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()
    

    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size/2))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size+300, self.window_size/2))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size // self.width
        
    
        draw_arrow(canvas,
                    get_wind_color(int(abs(self.vertical_wind[22,8]))),
                    (8 * pix_square_size + pix_square_size/10, #start
                    22 * pix_square_size + pix_square_size//2 ), 
                    (8 * pix_square_size + pix_square_size - pix_square_size/10 , # end 
                    22*pix_square_size + pix_square_size//2),
                    width=1, # int(abs(self.vertical_wind[row,col])),
                    head_width = 2,
                    head_length= 4,#2.5*int(abs(self.vertical_wind[row,col]))+2
                    flip_direction=False)
        for row in range(self.height):
            for col in range(self.width):
                color = self.graphics[row, col]
                pygame.draw.rect(
                    canvas,
                    color[:3],  # Use the first three values (RGB) from the color
                    (col * pix_square_size, row* pix_square_size, pix_square_size, pix_square_size),
                )
                

                
                if self.vertical_wind[row,col] != 0:
                    flip = 0>np.sign(self.vertical_wind[row,col])
                    draw_arrow(canvas,
                        get_wind_color(int(abs(self.horizontal_wind[row,col]))),
                        (col * pix_square_size + pix_square_size//2 , #start
                        row * pix_square_size + pix_square_size/10), 
                        (col * pix_square_size + pix_square_size//2  , # end 
                        row*pix_square_size + pix_square_size - pix_square_size/10),
                        width=1, #int(abs(self.horizontal_wind[row,col])),
                        head_width = 2,
                        head_length= 4,
                        flip_direction=flip)
                    
                    
                if self.horizontal_wind[row,col] != 0:
                    flip = 0>np.sign(self.horizontal_wind[row,col])
                    
                    draw_arrow(canvas,
                                get_wind_color(int(abs(self.vertical_wind[row,col]))),
                                (col * pix_square_size + pix_square_size/10, #start
                                row * pix_square_size + pix_square_size//2 ), 
                                (col * pix_square_size + pix_square_size - pix_square_size/10 , # end 
                                row*pix_square_size + pix_square_size//2),
                                width=1, # int(abs(self.vertical_wind[row,col])),
                                head_width = 2,
                                head_length= 4,#2.5*int(abs(self.vertical_wind[row,col]))+2
                                flip_direction=flip)
                    
         
               
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
        pygame.draw.circle(canvas, (151,87,43), agent_center, pix_square_size // 2)

        pygame.font.init()
        font = pygame.font.Font(None, 32)
        
        
        score_text = font.render(f'Steps: {self._step}',True, (0, 0, 0))
        pos_text = font.render(f'position: ({self._agent_location[0]}, {self._agent_location[1]})',True, (0, 0, 0))
        cum_reward_text = font.render(f'Cumulative return: {abs(self._cummlative_reward):.2f}',True, (0, 0, 0))

        pygame.draw.rect(
            canvas,
            (0, 0, 0),  # Use the first three values (RGB) from the color
            (64 * pix_square_size , # left
            0 , # top
            pix_square_size, # width
              32*pix_square_size), # heigth
        ) 
        #(col * pix_square_size, row* pix_square_size, pix_square_size, pix_square_size)
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
        