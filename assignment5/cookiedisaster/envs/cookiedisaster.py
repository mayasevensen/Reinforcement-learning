import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple
import os
import math

np.random.seed(0)

class CookieDisasterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = (1200, 600)  # Width and height of the PyGame window
        self.width = 10  # Width of the environment grid
        self.height = 5  # Height of the environment grid
        self.scale_x = self.window_size[0] /  self.width
        self.scale_y = self.window_size[1] / self.height

        image_path = os.path.join('cookiedisaster', 'envs', 'cookie.png')
        self.cookie_image = pygame.image.load(image_path)
        self.cookie_image = pygame.transform.scale( self.cookie_image, (20, 20))
        self.target_image = pygame.transform.scale( self.cookie_image, (35, 35))
        
        self._cumulative_reward = 0 

        self._step = 0
        self.time = 0.0
        self.delta_time= 0.2 
        self.cookie_time = 5
        self.human_action = None
        

        self.cookie_positions = []
        for i in np.linspace(0.1, 9.9, 20):
            self.cookie_positions.append(( (min(np.random.normal(i,0.02), 9.8) * self.scale_x )*0.95- 20/2, np.random.normal(self.window_size[1]/1.8,0.1)  - 20/2))
            self.cookie_positions.append(((min(np.random.normal(i,0.01), 9.8)* self.scale_x - 20/2 , np.random.normal(self.window_size[1]/2,0.1) - 20/2)))
            self.cookie_positions.append( (((min(np.random.normal(i,0.02), 9.8)* self.scale_x*0.95 )- 20/2,  np.random.normal(self.window_size[1]/2.2,0.1) - 20/2)))


       
        

        
        

     
  
        self.start_position = 4
        
        


        self.observation_space = spaces.Box(low=np.array([0,-np.inf,0]),
                               high=np.array([10, np.inf,10]),
                               dtype=np.float32)
                               
        

        # We have 3 actions, corresponding to "accelerate left, no accelerate, accelerate right "
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.thrust = {
            0: -5,  # accelerate left
            1: 0,   # no accelerate
            2: 5    # accelerate right
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
    

    def _friction(self,vel:float):
        return - abs(vel)*vel*0.05
    
    def _collides(self, x:float):
        if x<=0 or x>=10:
            return True
        return False
    
    def _getReward(self, x1:float, x2:float, vel:float, x_c :float):
        if (x1< x_c and x_c <x2) or (x2< x_c and x_c <x1):
     
            if abs(vel) > 4:
                return -1,True
            else:
                return 1,True
        
        return 0,False

    def _get_obs(self):
        return {"agent": {"pos": self._agent_location, "vel": self._agent_velocity},
                 "cookie": {"pos":self._target_location, "time": self.cookie_time },
                }


    def _get_info(self):
        return {"distance": abs(self._agent_location - self._target_location), "steps": self._step}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = self.start_position
        self._agent_velocity = 0.0
        self._step = 0
        self._cummlative_reward = 0
        self.cookie_time = 5

        self._target_location = np.random.uniform(0.001,0.9999)
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def conveyor(self, gotCookie:bool):
        if self._step ==0:
            return 

        if (np.round(self.cookie_time,2)==0 and not gotCookie) or gotCookie:
            self._target_location = np.random.uniform(0,10)
            self.cookie_time = 5
        

    
    def step(self, action):
        

        friction = self._friction(self._agent_velocity)
        self._agent_velocity = self._agent_velocity +  (self.thrust[action] + friction)*self.delta_time

        new_x = self._agent_location + self._agent_velocity*self.delta_time


        
        reward, gotCookie = self._getReward(self._agent_location,new_x,self._agent_velocity, self._target_location  )
        
    
        if self._collides(new_x):
            reward -= (self._agent_velocity**2)*0.1
            new_x= np.clip(new_x,0.00001,9.999999)
            self._agent_velocity = 0.0
        
        self._agent_location = new_x
        if self.cookie_time<=0:
            reward -= 0.5
        self.conveyor(gotCookie)
       
       
   
    

        self._step += 1    

        
    
        # An episode is done if the agent has reached the target
        reached_goal= np.array_equal(self._agent_location, self._target_location)


        terminated = reached_goal
       
        

        for i, pos in enumerate(self.cookie_positions):
            x_pos = pos[0]+2
            if x_pos > self.window_size[0]:
                x_pos = 0

            self.cookie_positions[i] = (x_pos  , pos[1])
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        
        self._cummlative_reward +=reward
        self.time+=self.delta_time
        self.cookie_time -= self.delta_time

        return observation, reward, False, False, info
    
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        



        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))  # White background

        
        # Example drawing code
        pix_square_size = self.window_size[0] // self.width
        
        # Draw the conveyor belt
        conveyor_belt_top_left = (0, 2 * self.scale_y)
        conveyor_belt_width_height = (self.window_size[0], self.scale_y)
        pygame.draw.rect(canvas, (245, 222, 179), (conveyor_belt_top_left, conveyor_belt_width_height))

        # roof 
        roof_belt_top_left = (0, 0.0)
        roof_belt_width_height = (self.window_size[0], 50)
        pygame.draw.rect(canvas, (50, 40, 25), (roof_belt_top_left, roof_belt_width_height))

        # floor
        floor_belt_top_left = (0, self.window_size[1]-50)
        floor_belt_width_height = (self.window_size[0], 50)
        pygame.draw.rect(canvas, (50, 40, 25), (floor_belt_top_left, floor_belt_width_height))


        # Draw lights on the ceiling
        for i in range(1, 10, 2):
            pygame.draw.circle(canvas, (255, 255, 0), (i * self.scale_x, 52), 10)

        for pos in self.cookie_positions:
            canvas.blit(self.cookie_image, (pos[0], pos[1]))
        

        #draw target
        canvas.blit(self.target_image, (self._target_location * self.scale_x - 35/2, self.window_size[1]-60))

        #draw the agent
        agent_center = (
            self._agent_location * self.scale_x,
            self.window_size[1]-60
        )
        pygame.draw.circle(canvas, (151,87,43), agent_center, 20)

        
        pygame.font.init()
        font = pygame.font.Font(None, 28)
        
        
        score_text = font.render(f'Steps: {self._step}',True, (0, 0, 0))
        pos_text = font.render(f'Position: ({self._agent_location:.2f})',True, (0, 0, 0))
        vel_text = font.render(f'Velocity: {self._agent_velocity:.2f}',True, (0, 0, 0))
        cum_reward_text = font.render(f'Cumulative return: {self._cummlative_reward:.2f}',True, (0, 0, 0))
        cookie_time_text = font.render(f'Cookie timer: {self.cookie_time:.2f}',True, (0, 0, 0))

        canvas.blit(score_text, (10, 60))
        canvas.blit(pos_text, (10, 100))
        canvas.blit(vel_text, (10, 140))
        canvas.blit(cum_reward_text, (10, 180))
        canvas.blit(cookie_time_text, (10, 220))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

