import gym
import numpy as np
from gym import spaces

count = 0
class UAV2UERelativePosition(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.metadata = env.metadata
        assert ('region' in self.metadata and 'nUAV' in self.metadata and 'nUser' in self.metadata)

        region = np.array(self.metadata['region'])
        nUAV = self.metadata['nUAV']
        self.nUser = self.metadata['nUser']

        uav_lower_bound = np.concatenate([np.full((1, nUAV), region[0,0]), 
                                          np.full((1, nUAV), region[0,1]), 
                                          np.full((1, nUAV), region[0,2])])
        
        uav_upper_bound = np.concatenate([np.full((1, nUAV), region[1,0]), 
                                          np.full((1, nUAV), region[1,1]), 
                                          np.full((1, nUAV), region[1,2])])
        
        user_lower_bound = np.concatenate([np.full((1, self.nUser), region[0,0]-region[1,0]), 
                                           np.full((1, self.nUser), region[0,1]-region[1,1]), 
                                           np.full((1, self.nUser), region[0,2]-region[1,2])])
        
        user_upper_bound = np.concatenate([np.full((1, self.nUser), region[1,0]-region[0,0]),
                                           np.full((1, self.nUser), region[1,1]-region[0,1]),
                                           np.full((1, self.nUser), region[1,2]-region[0,2])])
        
        association_lower_bound = np.ones(self.nUser)
        association_upper_bound = np.full(self.nUser, nUAV)


        self.observation_space = spaces.Dict({
                "uavPosition": spaces.Box(uav_lower_bound, uav_upper_bound),
                "userPosition": spaces.Box(user_lower_bound, user_upper_bound),
                "userAssociation": spaces.Box(association_lower_bound, association_upper_bound, dtype=int)
            })

    def observation(self, observation):
        uav_pos = np.array(observation["uavPosition"])
        user_pos = np.array(observation["userPosition"])
        association = np.array(observation["userAssociation"], int)
        
        user_relative_pos = np.zeros(user_pos.shape)
        for user in range(self.nUser):
            user_relative_pos[:, user] = user_pos[:, user] - uav_pos[:, association[user]]
        observation["userPosition"] = user_relative_pos

        return observation

class UnitSquareScaling(gym.ObservationWrapper):
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    z_min, z_max =  0, 1

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.metadata = env.metadata
        assert ('region' in self.metadata and 'nUAV' in self.metadata and 'nUser' in self.metadata)

        self.metadata['scaled_region'] = np.array([[self.x_min, self.y_min, self.z_min],
                                                   [self.x_max, self.y_max, self.z_max]])

        uav_lower_bound = np.concatenate([np.full((1, self.metadata['nUAV']), self.x_min), 
                                          np.full((1, self.metadata['nUAV']), self.y_min), 
                                          np.full((1, self.metadata['nUAV']), self.z_min)])
        
        uav_upper_bound = np.concatenate([np.full((1, self.metadata['nUAV']), self.x_max), 
                                          np.full((1, self.metadata['nUAV']), self.y_max), 
                                          np.full((1, self.metadata['nUAV']), self.z_max)])
        
        user_lower_bound = np.concatenate([np.full((1, self.metadata['nUser']), self.x_min), 
                                           np.full((1, self.metadata['nUser']), self.y_min), 
                                           np.full((1, self.metadata['nUser']), self.z_min)])
        
        user_upper_bound = np.concatenate([np.full((1, self.metadata['nUser']), self.x_max),
                                           np.full((1, self.metadata['nUser']), self.y_max),
                                           np.full((1, self.metadata['nUser']), self.z_max)])
        
        association_lower_bound = np.ones(self.metadata['nUser'])
        association_upper_bound = np.full(self.metadata['nUser'], self.metadata['nUAV'])
        
        self.observation_space = spaces.Dict({
                "uavPosition": spaces.Box(uav_lower_bound, uav_upper_bound),
                "userPosition": spaces.Box(user_lower_bound, user_upper_bound),
                "userAssociation": spaces.Box(association_lower_bound, association_upper_bound, dtype=int)
            })

    def observation(self, observation):
        uav_pos = np.array(observation["uavPosition"])
        user_pos = np.array(observation["userPosition"])

        # scale x-axis
        scaled_uav_x = (self.x_max-self.x_min) * (uav_pos[0]-self.metadata['region'][0,0]) / (self.metadata['region'][1,0]-self.metadata['region'][0,0]) + self.x_min
        scaled_user_x = (self.x_max-self.x_min) * (user_pos[0]-self.metadata['region'][0,0]) / (self.metadata['region'][1,0]-self.metadata['region'][0,0]) + self.x_min
        # scale y-axis
        scaled_uav_y = (self.y_max-self.y_min) * (uav_pos[1]-self.metadata['region'][0,1]) / (self.metadata['region'][1,1]-self.metadata['region'][0,1]) + self.y_min
        scaled_user_y = (self.y_max-self.y_min) * (user_pos[1]-self.metadata['region'][0,1]) / (self.metadata['region'][1,1]-self.metadata['region'][0,1]) + self.y_min
        # scale z-axis
        scaled_uav_z = (self.z_max-self.z_min) * (uav_pos[2]-self.metadata['region'][0,2]) / (self.metadata['region'][1,2]-self.metadata['region'][0,2]) + self.z_min
        scaled_user_z =(self.z_max-self.z_min) * (user_pos[2]-self.metadata['region'][0,2]) / (self.metadata['region'][1,2]-self.metadata['region'][0,2]) + self.z_min
        # reconstruct
        scaled_uav_pos = np.array([scaled_uav_x, scaled_uav_y, scaled_uav_z])
        scaled_user_pos = np.array([scaled_user_x, scaled_user_y, scaled_user_z])

        observation["uavPosition"] = scaled_uav_pos
        observation["userPosition"] = scaled_user_pos

        return observation
    
class SingleUAV2D(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.metadata = env.metadata
        assert ('region' in self.metadata and 'nUAV' in self.metadata and 'nUser' in self.metadata)

        region = np.array(self.metadata.get('scaled_region', self.metadata['region']))
        assert (region.shape == (2, 3)) and (self.metadata['nUAV'] == 1)
        x_min, y_min, _ = region[0]
        x_max, y_max, _ = region[1]

        uav_lower_bound = np.concatenate([np.full((1, self.metadata['nUAV']), x_min), 
                                          np.full((1, self.metadata['nUAV']), y_min)])
        
        uav_upper_bound = np.concatenate([np.full((1, self.metadata['nUAV']), x_max), 
                                          np.full((1, self.metadata['nUAV']), y_max)])
        
        user_lower_bound = np.concatenate([np.full((1, self.metadata['nUser']), x_min), 
                                           np.full((1, self.metadata['nUser']), y_min)])
        
        user_upper_bound = np.concatenate([np.full((1, self.metadata['nUser']), x_max),
                                           np.full((1, self.metadata['nUser']), y_max)])
                
        self.observation_space = spaces.Dict({
                "uavPosition": spaces.Box(uav_lower_bound, uav_upper_bound),
                "userPosition": spaces.Box(user_lower_bound, user_upper_bound),
            }) 

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if type(obs) is dict:
            obs.pop('userAssociation', None)
            uav_pos = obs.get('uavPosition', None)
            if uav_pos is not None:
                obs['uavPosition'] = np.array(obs['uavPosition'])[:2]
            user_pos = obs.get('userPosition', None)
            if user_pos is not None:
                obs['userPosition'] = np.array(obs['userPosition'])[:2]

        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if type(obs) is dict:
            obs.pop('userAssociation', None)
            uav_pos = obs.get('uavPosition', None)
            if uav_pos is not None:
                obs['uavPosition'] = np.array(obs['uavPosition'])[:2]
            user_pos = obs.get('userPosition', None)
            if user_pos is not None:
                obs['userPosition'] = np.array(obs['userPosition'])[:2]

        return obs, reward, done, truncated, info
    
class DiscreteReward(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.metadata = env.metadata
        assert ('region' in self.metadata and 'nUAV' in self.metadata and 'nUser' in self.metadata)

        region = np.array(self.metadata.get('scaled_region', self.metadata['region']))
        assert (region.shape == (2, 3)) and (self.metadata['nUAV'] == 1)
        self.x_min, self.y_min, _ = region[0]
        self.x_max, self.y_max, _ = region[1]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_sum_rate = info["initial_reward"]
        return obs, info

    def step(self, action):
        obs, self.sum_rate, done, truncated, info = self.env.step(action)

        if truncated:
            reward = -5
        elif self.sum_rate > self.prev_sum_rate:
            reward = 1
        elif self.sum_rate < self.prev_sum_rate:
            reward = -1
        else:
            reward = -0.2

        self.prev_sum_rate = self.sum_rate

        return obs, reward, done, truncated, info

#not use class
class MixedRateBoundaryDiscreteReward(gym.Wrapper):

    def __init__(self, env: gym.Env, rate_factor: float = 1, boundary_factor: float = 1):
        super().__init__(env)
        self.metadata = env.metadata
        assert ('region' in self.metadata and 'nUAV' in self.metadata and 'nUser' in self.metadata)

        region = np.array(self.metadata.get('scaled_region', self.metadata['region']))
        assert (region.shape == (2, 3)) and (self.metadata['nUAV'] == 1)
        self.x_min, self.y_min, _ = region[0]
        self.x_max, self.y_max, _ = region[1]

        self.rate_factor = rate_factor
        self.boundary_factor = boundary_factor

    def reset(self, **kwargs):
        print("reset")
        obs, info = self.env.reset(**kwargs)
        self.prev_sum_rate = self.min_sum_rate = self.max_sum_rate = info["initial_reward"]
        print(f"initial sum_rate = {self.prev_sum_rate}")
        return obs, info

    def step(self, action):
        obs, self.sum_rate, done, truncated, info = self.env.step(action)
        print(f"sum_rate = {self.sum_rate}")

        if truncated:
            rate_reward = 0
        elif self.sum_rate >= self.max_sum_rate:
            rate_reward = 1
        elif self.sum_rate <= self.min_sum_rate:
            rate_reward = -1
        elif self.sum_rate > self.prev_sum_rate:
            rate_reward = 0.1
        elif self.sum_rate < self.prev_sum_rate:
            rate_reward = -0.1
        else:
            rate_reward = -0.05

        if self.sum_rate > self.max_sum_rate:
            self.max_sum_rate = self.sum_rate
        if self.sum_rate < self.min_sum_rate:
            self.min_sum_rate = self.sum_rate
        self.prev_sum_rate = self.sum_rate

        uavPosition = np.array(obs["uavPosition"])

        if truncated:                                 #超出邊界，結束episode
            boundary_penalty = -4
        elif self._check_boundary(uavPosition) == -3: #超出邊界
            boundary_penalty = -4
        elif self._check_boundary(uavPosition) == -2: #距離邊界一步
            boundary_penalty = -2
        elif self._check_boundary(uavPosition) == -1: #距離邊界兩步
            boundary_penalty = -1
        else:                                         #其他
            boundary_penalty = 0
        
        # 兩個factor皆設為1
        reward = self.rate_factor*rate_reward + self.boundary_factor*boundary_penalty
        return obs, reward, done, truncated, info

class NSDiscreteReward(gym.Wrapper): #google search gym wrapper

    def __init__(self, env: gym.Env, rate_factor: float = 1, endpoint_factor: float = 1, distance_factor: float = 1):
        
        super().__init__(env)
        self.metadata = env.metadata
        assert ('region' in self.metadata and 'nUAV' in self.metadata and 'nUser' in self.metadata)

        region = np.array(self.metadata.get('scaled_region', self.metadata['region']))
        assert (region.shape == (2, 3)) and (self.metadata['nUAV'] == 1)
        self.x_min, self.y_min, _ = region[0]
        self.x_max, self.y_max, _ = region[1]

        self.rate_factor = rate_factor
        #self.endpoint_factor = endpoint_factor
        self.distance_factor = distance_factor

    def reset(self, **kwargs):

        print("reset")
        obs, info = self.env.reset(**kwargs)
        self.prev_sum_rate = self.min_sum_rate = self.max_sum_rate = info["initial_reward"]
        print(f"initial sum_rate = {self.prev_sum_rate}")
        self.prev_distance = 3
        return obs, info

    def step(self, action):
        
        global count
        obs, self.sum_rate, done, truncated, info = self.env.step(action)
        print(f"sum_rate = {self.sum_rate}")
        print(f"count = {count}")

        uavPosition = np.array(obs["uavPosition"])
        print(f"uav position = {uavPosition}")

        if self.sum_rate > self.prev_sum_rate:
            rate_reward = 0.1
        elif self.sum_rate == self.prev_sum_rate:
            rate_reward = 0
        elif self.sum_rate < self.prev_sum_rate:
            rate_reward = -0.1

        distance = self._check_distance(uavPosition)
        if distance < self.prev_distance:
            distance_reward = 0.1
        elif distance == self.prev_distance:
            distance_reward = 0
        else:
            distance_reward = -0.1

        self.prev_distance = distance
        self.prev_sum_rate = self.sum_rate
        
        reward = self.rate_factor*rate_reward + 1*self.sum_rate*0.005 #+ distance_reward
        return obs, reward, done, truncated, info
    
    def _check_boundary(self, pos)->int:
        pos = np.squeeze(pos)
        
        if pos[0] < self.x_min or pos[0] > self.x_max \
            or pos[1] < self.y_min or pos[1] > self.y_max:
            return -3
        elif pos[0] < 0.95*self.x_min or pos[0] > 0.95*self.x_max \
            or pos[1] < 0.95*self.y_min or pos[1] > 0.95*self.y_max:
            return -2
        elif pos[0] < 0.9*self.x_min or pos[0] > 0.9*self.x_max \
            or pos[1] < 0.9*self.y_min or pos[1] > 0.9*self.y_max:
            return -1
        
        return 0

    #not use class
    def _check_arrive_endpoint(self, pos)->int:
        pos = np.squeeze(pos)

        if pos[0] > 0.95*self.x_max and pos[1] > 0.95*self.y_max:
            return 1
        else:
            return -1

        return 0
        
    def _check_distance(self, pos)->float:
        pos = np.squeeze(pos)
        centroid = [0.48,0.4215]

        distance = np.sqrt((pos[0] - centroid[0])**2 + (pos[1] - centroid[1])**2)
        

        return distance