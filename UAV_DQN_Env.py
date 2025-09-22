import os
import sys

# Interface using file
PYTHON_RL_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SIMU_ROOT_PATH = os.path.dirname(PYTHON_RL_ROOT_PATH)
sys.path.insert(1, SIMU_ROOT_PATH)

SAVE_PATH = os.path.join(SIMU_ROOT_PATH, 'training_result')
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

readFileDir = os.path.join(SIMU_ROOT_PATH, 'read_write_file_interface')
readFile = os.path.join(readFileDir, 'out.json')
writeFileDir = readFileDir
writeFile = os.path.join(writeFileDir, '23.json')

import numpy as np
import gym
from gym import spaces
from matplotlib import pyplot, animation
from mpl_toolkits import mplot3d

from python_RL.Interface import file_interface

class UAVDQNEnv(gym.Env):
    metadata = {"render_modes": ["animation", "log"]}

    save_episode = False

    def __init__(self, render_mode=None, time_str: str = 'Non-Specific_Name', step_per_episode: int = 1, debug=False):
        self.UAV_DQN_ENV_DEBUG = debug
        if self.UAV_DQN_ENV_DEBUG: print("UAVDQNEnv initialization")

        self.time_str = time_str
        self.root_save_path = os.path.join(SAVE_PATH, self.time_str)
        self.scene_2d_save_path = os.path.join(self.root_save_path, 'save_scene', '2d')
        self.scene_3d_save_path = os.path.join(self.root_save_path, 'save_scene', '3d')
        self.log_path = os.path.join(self.root_save_path, 'save_log')
        print(f'\nenvironment save_scene path:\n{self.scene_2d_save_path}\n{self.scene_3d_save_path}\n')
        if not os.path.isdir(self.scene_2d_save_path):
            os.makedirs(self.scene_2d_save_path)
        if not os.path.isdir(self.scene_3d_save_path):
            os.makedirs(self.scene_3d_save_path)
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        self.episode_count: int = -1
        self.max_step = step_per_episode
        
        # send initialization request to simulator
        # receive observation space
        # Region=((x1, y1, z1), (x2, y2, z2)), nUAV, nUser, uavSpeed

        self.region, self.nUAV, self.nUser, self.uavSpeed = self._init_simu()
        print(f'\nUAVDQNEnv settings:\nregion = {self.region}\nnUAV = {self.nUAV}\nnUser = {self.nUser}\nuavSpeed = {self.uavSpeed}\n')

        self.metadata['region'] = self.region
        self.metadata['nUAV'] = self.nUAV
        self.metadata['nUser'] = self.nUser

        self._action_to_movement = {
            0: np.zeros(3),
            1: self.uavSpeed * self._direction(degree=0, height=0),
            2: self.uavSpeed * self._direction(90, 0),
            3: self.uavSpeed * self._direction(180, 0),
            4: self.uavSpeed * self._direction(270, 0),
            5: self.uavSpeed * self._direction(45, 0),
            6: self.uavSpeed * self._direction(135, 0),
            7: self.uavSpeed * self._direction(225, 0),
            8: self.uavSpeed * self._direction(315, 0),
            9: self.uavSpeed * self._direction(0, 90),
            10: self.uavSpeed * self._direction(90, 1),
            11: self.uavSpeed * self._direction(180, 1),
            12: self.uavSpeed * self._direction(270, 1),
            13: self.uavSpeed * self._direction(45, 1),
            14: self.uavSpeed * self._direction(135, 1),
            15: self.uavSpeed * self._direction(225, 1),
            16: self.uavSpeed * self._direction(315, 1),
            17: self.uavSpeed * self._direction(0, -1),
            18: self.uavSpeed * self._direction(90, -1),
            19: self.uavSpeed * self._direction(180, -1),
            20: self.uavSpeed * self._direction(270, -1),
            21: self.uavSpeed * self._direction(45, -1),
            22: self.uavSpeed * self._direction(135, -1),
            23: self.uavSpeed * self._direction(225, -1),
            24: self.uavSpeed * self._direction(315, -1)
            
        }
        
        uav_lower_bound = np.concatenate([np.full((1, self.nUAV), self.region[0,0]), 
                                         np.full((1, self.nUAV), self.region[0,1]), 
                                         np.full((1, self.nUAV), self.region[0,2])])
        
        uav_upper_bound = np.concatenate([np.full((1, self.nUAV), self.region[1,0]), 
                                         np.full((1, self.nUAV), self.region[1,1]), 
                                         np.full((1, self.nUAV), self.region[1,2])])
        
        user_lower_bound = np.concatenate([np.full((1, self.nUser), self.region[0,0]), 
                                          np.full((1, self.nUser), self.region[0,1]), 
                                          np.full((1, self.nUser), self.region[0,2])])
        
        user_upper_bound = np.concatenate([np.full((1, self.nUser), self.region[1,0]),
                                          np.full((1, self.nUser), self.region[1,1]),
                                          np.full((1, self.nUser), self.region[1,2])])
        
        association_lower_bound = np.ones(self.nUser)
        association_upper_bound = np.full(self.nUser, self.nUAV)
        
        self.observation_space = spaces.Dict({
                "uavPosition": spaces.Box(uav_lower_bound, uav_upper_bound),
                "userPosition": spaces.Box(user_lower_bound, user_upper_bound),
                "userAssociation": spaces.Box(association_lower_bound, association_upper_bound, dtype=int)
            }           
        )
        self.action_size = 25
        self.action_space = spaces.MultiDiscrete(np.full(self.nUAV, self.action_size))

        self.observation = None
        self.reward_record: np.ndarray = None   # numpy array shape: (num_uav, max_step_per_episode+1)
        self.uavPosTensor: np.ndarray = None    # numpy array shape: (num_frames, 3, nUAV)
        self.userPosTensor: np.ndarray = None   # numpy array shape: (num_frames, 3, nUser)

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        
        self.container_2d = []  # elements are collection of artists for each frame
        self.container_3d = []  # elements are collection of artists for each frame

        if self.render_mode == "animation":

            self.figure_2d = pyplot.figure("2d")
            self.axes_2d = self.figure_2d.add_subplot()
            self.animation_2d = None

            self.figure_3d = pyplot.figure("3d")
            self.axes_3d = self.figure_3d.add_subplot(projection='3d')
            self.animation_3d = None

    def reset(self):
        if self.UAV_DQN_ENV_DEBUG: print("UAVDQNEnv reset")

        self.episode_count += 1

        if self.save_episode:
            self._save_episode(f'epi_{self.episode_count}')
        
        self.save_episode = True

        self.step_count: int = 0
        self.observation, initial_reward = self._reset_simu()
        self.reward_record = initial_reward.reshape([1, initial_reward.shape[0]])
        
        uavPosition = np.array(self.observation['uavPosition'])
        self.uavPosTensor = uavPosition.reshape([1, uavPosition.shape[0], uavPosition.shape[1]])

        userPosition = np.array(self.observation['userPosition'])
        self.userPosTensor = userPosition.reshape([1, userPosition.shape[0], userPosition.shape[1]])

        valid_actions = self._validate_actions()

        if self.render_mode == "animation":
            for l_2d, l_3d in zip(self.container_2d, self.container_3d):
                l_2d.clear()
                l_3d.clear()
            self.container_2d.clear()
            self.container_3d.clear()
            
            self.axes_2d.clear()
            self.axes_3d.clear()

            self.axes_2d.set_xlim(1.5*self.region[0,0], 1.5*self.region[1,0])
            self.axes_2d.set_xlabel('X')
            self.axes_2d.set_ylim(1.5*self.region[0,1], 1.5*self.region[1,1])
            self.axes_2d.set_ylabel('Y')

            self.axes_3d.set(xlim3d=(1.5*self.region[0,0], 1.5*self.region[1,0]), xlabel='X')
            self.axes_3d.set(ylim3d=(1.5*self.region[0,1], 1.5*self.region[1,1]), xlabel='Y')
            self.axes_3d.set(zlim3d=(self.region[0,2], self.region[1,2]), xlabel='Z')

            self.render()

        return self.observation, {"initial_reward": initial_reward, "region": self.region, "valid_actions": valid_actions}

    def step(self, action):
        if self.UAV_DQN_ENV_DEBUG: print("UAVDQNEnv step")
        # print('[REMINDER] argument `action` of function `step` should be iterable, e.g. list')
        
        self.step_count += 1

        action = self._action_decoder(action) # A 3-by-nUAV matrix indicates movement
        self.observation, reward = self._step_with_action(action)
        reshaped_reward = reward.reshape([1, reward.shape[0]])
        self.reward_record = np.concatenate([self.reward_record, reshaped_reward])

        uavPosition = np.array(self.observation['uavPosition'])
        uavAppendedTensor = uavPosition.reshape([1, uavPosition.shape[0], uavPosition.shape[1]])
        self.uavPosTensor = np.concatenate([self.uavPosTensor, uavAppendedTensor])

        userPosition = np.array(self.observation['userPosition'])
        userAppendedTensor = userPosition.reshape([1, userPosition.shape[0], userPosition.shape[1]])
        self.userPosTensor = np.concatenate([self.userPosTensor, userAppendedTensor])

        # check done
        done = (self.step_count >= self.max_step)

        # check truncated
        for i in range(uavPosition.shape[1]):
            # TODO: only 1 uav for now, modify to fit multi uav
            truncated = self._check_truncation(uavPosition[:,i])
            break

        valid_actions = self._validate_actions()

        if self.render_mode == "animation":
            self.render()

        return self.observation, reward, done, truncated, {"region": self.region, "valid_actions": valid_actions}
    
    def render(self):
        if self.UAV_DQN_ENV_DEBUG: print('render')

        if self.observation is None:
            return
        
        if self.render_mode != "animation":
            return
        
        frame_2d, frame_3d = [], []
        uavTransPosTensor = np.transpose(self.uavPosTensor, [1, 0, 2])
        userTransPosTensor = np.transpose(self.userPosTensor, [1, 0, 2])

        # plot region of interest
        region_bound_x = [self.region[0,0], self.region[1,0], self.region[1,0], self.region[0,0], self.region[0,0]]
        region_bound_y = [self.region[0,1], self.region[0,1], self.region[1,1], self.region[1,1], self.region[0,1]]
        # plot 2D
        line2d, = self.axes_2d.plot(region_bound_x, region_bound_y, color='black')
        frame_2d.append(line2d)
        # plot 3D
        line3d, = self.axes_3d.plot(region_bound_x, region_bound_y, np.zeros(5), color='black')
        frame_3d.append(line3d)
        
        
        # plot users' trajectories and association line

        # assoc_array = self.observation['userAssociation']
        trail = np.max([-20, -userTransPosTensor.shape[1]])
        for user in range(self.nUser):
            # plot trajectories in 2D
            line2d, = self.axes_2d.plot(userTransPosTensor[0, trail:, user],
                                        userTransPosTensor[1, trail:, user],
                                        color='gold',
                                        linestyle='-.')
            frame_2d.append(line2d)
            # plot trajectories in 3D
            line3d, = self.axes_3d.plot(userTransPosTensor[0, trail:, user],
                                        userTransPosTensor[1, trail:, user],
                                        userTransPosTensor[2, trail:, user],
                                        color='gold',
                                        linestyle='-.')
            frame_3d.append(line3d)

        # plot UAVs' trajectories
        for uav in range(self.nUAV):
            # plot trajectories in 2D
            line2d, = self.axes_2d.plot(uavTransPosTensor[0, :, uav],
                                        uavTransPosTensor[1, :, uav],
                                        color='limegreen',
                                        linestyle='-.')
            frame_2d.append(line2d)
            # plot trajectories in 3D
            line3d, = self.axes_3d.plot(uavTransPosTensor[0, :, uav],
                                        uavTransPosTensor[1, :, uav],
                                        uavTransPosTensor[2, :, uav],
                                        color='limegreen',
                                        linestyle='-.')
            frame_3d.append(line3d)

            # plot 3D positioning line
            position_line, = self.axes_3d.plot([uavTransPosTensor[0, -1, uav], uavTransPosTensor[0, -1, uav]],
                                               [uavTransPosTensor[1, -1, uav], uavTransPosTensor[1, -1, uav]],
                                               [uavTransPosTensor[2, -1, uav], 0],
                                               color='orange')
            frame_3d.append(position_line)


        # # plot association in 2D
        # assoc_uav = int(assoc_array[user])
        # if assoc_uav < 0 or assoc_uav >= self.nUAV: 
        #     # check if invalid association occur. If yes, ignore to plot
        #     continue

        # association_2d, = self.axes_2d.plot([userTransPosTensor[0, -1, user], uavTransPosTensor[0, -1, assoc_uav]],
        #                                     [userTransPosTensor[1, -1, user], uavTransPosTensor[1, -1, assoc_uav]],
        #                                     color='black',
        #                                     linestyle='--')
        # frame_2d.append(association_2d)
        # # plot association in 3D
        # association_3d, = self.axes_3d.plot([userTransPosTensor[0, -1, user], uavTransPosTensor[0, -1, assoc_uav]],
        #                                     [userTransPosTensor[1, -1, user], uavTransPosTensor[1, -1, assoc_uav]],
        #                                     [userTransPosTensor[2, -1, user], uavTransPosTensor[2, -1, assoc_uav]],
        #                                     color='black',
        #                                     linestyle='--')
        # frame_3d.append(association_3d)

        # plot users' current points
        user_point_2d = self.axes_2d.scatter(userTransPosTensor[0, -1, :],
                                             userTransPosTensor[1, -1, :],
                                             color='royalblue')
        user_point_3d = self.axes_3d.scatter(userTransPosTensor[0, -1, :],
                                             userTransPosTensor[1, -1, :],
                                             userTransPosTensor[2, -1, :],
                                             color='royalblue')
        # plot UAVs' current points
        uav_point_2d = self.axes_2d.scatter(uavTransPosTensor[0, -1, :],
                                            uavTransPosTensor[1, -1, :],
                                            color='red')
        uav_point_3d = self.axes_3d.scatter(uavTransPosTensor[0, -1, :],
                                            uavTransPosTensor[1, -1, :],
                                            uavTransPosTensor[2, -1, :],
                                            color='red')

        # add points to frames
        frame_2d.append(uav_point_2d)
        frame_2d.append(user_point_2d)

        frame_3d.append(uav_point_3d)
        frame_3d.append(user_point_3d)

        # add frames to containers
        self.container_2d.append(frame_2d)
        self.container_3d.append(frame_3d)

    def close(self):
        file_interface.send(writeFile, "end")
        if self.save_episode:
            self._save_episode(f'epi_{self.episode_count+1}')

    def _save_episode(self, filename: str = 'NoName'):
        if self.UAV_DQN_ENV_DEBUG: print('save episode')
        if self.render_mode == "animation":
            self.animation_2d = animation.ArtistAnimation(self.figure_2d, self.container_2d, blit=True, interval=50, repeat=True, repeat_delay=250)
            self.animation_3d = animation.ArtistAnimation(self.figure_3d, self.container_3d, blit=True, interval=50, repeat=True, repeat_delay=250)
            self.animation_2d.save(os.path.join(self.scene_2d_save_path, 'single_uav_dqn_'+filename+'.gif'), writer=animation.PillowWriter(fps=20))
            self.animation_3d.save(os.path.join(self.scene_3d_save_path, 'single_uav_dqn_'+filename+'.gif'), writer=animation.PillowWriter(fps=20))
        elif self.render_mode == "log":
            # print("save log")
            np.savez_compressed(os.path.join(self.log_path, 'single_uav_dqn_'+filename), region=self.region, uav=self.uavPosTensor, user=self.userPosTensor, sum_rate=self.reward_record)

    def _init_simu(self):
        if self.UAV_DQN_ENV_DEBUG: print("UAVDQNEnv _init_simu")
        while True:
            file_interface.send(writeFile, "init")
            response_code, region, _, nUAV, nUser, uavSpeed = file_interface.receive_init(readFile)
            if response_code == "init":
                break
            else:
                print(f"[Warning] Receive response_code '{response_code}' when requesting 'INIT'")
        
        return region, nUAV, nUser, uavSpeed

    def _reset_simu(self):
        if self.UAV_DQN_ENV_DEBUG: print("UAVDQNEnv _reset_simu")
        while True:
            file_interface.send(writeFile, "reset")
            response_code, uavPosition, userMatrix, initial_reward = file_interface.receive_reset_step(readFile) 
            if response_code == "reset":
                break
            else:
                print(f"[Warning] Receive response_code '{response_code}' when requesting 'RESET'")
        
        observation = {"uavPosition": uavPosition, "userPosition": userMatrix[:3], "userAssociation": userMatrix[3]}
        return observation, initial_reward

    def _step_with_action(self, action):
        if self.UAV_DQN_ENV_DEBUG: print("UAVDQNEnv _step_with_action")
        while True:
            file_interface.send(writeFile, "step", action) 
            response_code, uavPosition, userMatrix, reward = file_interface.receive_reset_step(readFile)
            if response_code == "step":
                break
            else:
                print(f"[Warning] Receive response_code '{response_code}' when requesting 'STEP'")
        
        next_observation = {"uavPosition": uavPosition, "userPosition": userMatrix[:3], "userAssociation": userMatrix[3]}
        return next_observation, reward

    def _direction(self, degree:float, height:float):
        return np.array([np.cos(np.radians(degree)), np.sin(np.radians(degree)), height])
        
    def _action_decoder(self, actions):
        return np.concatenate([ np.array(self._action_to_movement[action]).reshape([3, 1]) for action in actions ], 1)

    def _check_truncation(self, uav_pos: np.ndarray = None):
        if uav_pos is None:
            return True

        #print(f'check trunction:\nregion=\n{self.region}\nuav_pos={uav_pos}')
        if (uav_pos[0] < self.region[0,0]) or (uav_pos[0] > self.region[1,0]) \
            or (uav_pos[1] < self.region[0,1]) or (uav_pos[1] > self.region[1,1]) \
            or (uav_pos[2] < self.region[0,2]) or (uav_pos[2] > self.region[1,2]):
            return True
        #Obstacle avoidance
        if (uav_pos[0] > -100) and (uav_pos[0] < -50) and (uav_pos[1] > -100) and (uav_pos[1] < -50) \
            or (uav_pos[0] > 50) and (uav_pos[0] < 100) and (uav_pos[1] > 50) and (uav_pos[1] < 100) \
            or (uav_pos[0] > -100) and (uav_pos[0] < -50) and (uav_pos[1] > 50) and (uav_pos[1] < 100) \
            or (uav_pos[0] > 50) and (uav_pos[0] < 100) and (uav_pos[1] > -100) and (uav_pos[1] < -50):
            return True
        
        return False
        
    def _validate_actions(self) -> np.ndarray:
        uav_pos = self.observation["uavPosition"]
        valid_actions = np.ones((self.nUAV, self.action_size), int)
        for k, v in self._action_to_movement.items():
            for uav in range(self.nUAV):
                test_move = uav_pos[:, uav] + v
                if self._check_truncation(test_move):
                    valid_actions[uav, k] = 0

        return valid_actions