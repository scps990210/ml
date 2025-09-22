import os
import sys
import time

# set up file interface
PYTHON_RL_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SIMU_ROOT_PATH = os.path.dirname(PYTHON_RL_ROOT_PATH)
sys.path.insert(1, SIMU_ROOT_PATH)

import gym
from gym.wrappers.flatten_observation import FlattenObservation
from matplotlib import pyplot
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from python_RL.Environment.wrapper.wrapper_pool import UnitSquareScaling, SingleUAV2D, MixedRateBoundaryDiscreteReward, NSDiscreteReward

# DQN Agent for UAV optimal position control
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size, model_file: str = None):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.95
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        self.batch_size = 256
        self.train_start = 512
        # create replay memory using deque
        self.memory = deque(maxlen=100000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model and model_file is not None:
            self.model.load_weights(model_file)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state, constraint: np.ndarray[int] = None):
        # TODO: 依據state所處位置，以及constraint
        if constraint is None:
            if np.random.rand() <= self.epsilon:
                res = random.randrange(self.action_size)
            else:
                q_value = self.model.predict(state)
                print(f'q_value = {q_value[0]}')
                res = np.argmax(q_value[0])
        else:
            if np.random.rand() <= self.epsilon:
                valid_actions = np.arange(self.action_size)[constraint == 1]
                res = random.choice(valid_actions)
            else:
                q_value = self.model.predict(state)
                print(f'q_value = {q_value[0]}')
                q_value = np.array(q_value[0])
                q_value[constraint == 0] = -np.inf
                res = np.argmax(q_value)
        
        return [res]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


'''
    start of program 
'''

if __name__ == "__main__":
    PROGRAM_START_TIME = time.time()

    DEBUG = True
    EPISODES = 600
    STEP_PER_EPISODE = 200
    TARGET_UPDATE_INTERVAL = 500

    if DEBUG:
        PREVIOUS_RECORDED_TIME = PROGRAM_START_TIME
        PREVIOUS_RESET_TIME = PROGRAM_START_TIME

    time_struct = time.localtime(PROGRAM_START_TIME)
    time_str = f'{time_struct.tm_year}-{time_struct.tm_mon}-{time_struct.tm_mday}_{time_struct.tm_hour}-{time_struct.tm_min}-{time_struct.tm_sec}'
    RESULT_PATH = os.path.join(SIMU_ROOT_PATH, 'training_result', time_str)
    SCORE_PATH = os.path.join(RESULT_PATH, 'save_score')
    SCORE_FILE = os.path.join(SCORE_PATH, 'single_uav_dqn.png')
    MODEL_PATH = os.path.join(RESULT_PATH, 'save_model')
    FINAL_MODEL_FILE = os.path.join(MODEL_PATH, '(final)single_uav_dqn.h5')

    print(f'DQN path:\n{SCORE_FILE}\n{FINAL_MODEL_FILE}\n')
    if not os.path.isdir(SCORE_PATH):
        os.makedirs(SCORE_PATH)
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    env = gym.make('UAV_5G_Sim/UAVDQNEnv-v0',
                   render_mode='log',
                   time_str=time_str,
                   step_per_episode=STEP_PER_EPISODE,
                   debug=False)

    env = UnitSquareScaling(env)
    env = SingleUAV2D(env) # check whether only 1 UAV is considered
    env = NSDiscreteReward(env, rate_factor=1, endpoint_factor=1, distance_factor=1)

    flatten_env = FlattenObservation(env)
    state_size = flatten_env.observation_space.shape[0]
    action_size = np.sum(flatten_env.action_space.nvec)
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    target_update_count = 0

    score_episode_fig = pyplot.figure('score-episode')

    for epi in range(EPISODES):
        print("==================================================")
        print(f"\n\t\t*** EPISODE {epi} START ***\n")
        print("==================================================")
        if DEBUG:
            CURRENT_TIME = time.time()
            print(f'\n******************************\nElapsed time since previous reset until reset of episode {epi} (NOW): {CURRENT_TIME-PREVIOUS_RESET_TIME}\n******************************\n')
            PREVIOUS_RECORDED_TIME = CURRENT_TIME
            PREVIOUS_RESET_TIME = CURRENT_TIME
        
        step = 0
        done = False
        truncated = False
        score = 0

        state, info = flatten_env.reset() # state is a tuple with (array)
        prev_sum_rate = info["initial_reward"]
        min_sum_rate = prev_sum_rate
        max_sum_rate = prev_sum_rate

        state = np.reshape(state, [1, state_size])

        while not done and not truncated:# and (step < STEP_PER_EPISODE if STEP_PER_EPISODE is not None and 0 < STEP_PER_EPISODE else True):
            if DEBUG:
                CURRENT_TIME = time.time()
                print(f'\n******************************\nElapsed time since previous observation until NOW ( episode {epi} , step {step} ): {CURRENT_TIME-PREVIOUS_RECORDED_TIME}\n******************************\n')
                PREVIOUS_RECORDED_TIME = CURRENT_TIME

            # get action for the current state and go one step in environment
            constraints = info["valid_actions"]
            action = agent.get_action(state, constraints[0, :])
            print("action = ",action)
            next_state, reward, done, truncated, info = flatten_env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            if DEBUG:
                print(f'\n<<<<<<<<<<<<<<<<<<<<\nepisode = {epi}')
                print(f'step = {step}')
                print(f'reward = {reward}\n<<<<<<<<<<<<<<<<<<<<\n')

            # save the sample <s, a, r, s'> to the replay memory
            # done flag is always set to False due to the nature of infinite mission time
            agent.append_sample(state, action, reward, next_state, False)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done or truncated:
                scores.append(score)
                episodes.append(epi)
                
                ax = score_episode_fig.gca()
                ax.plot(episodes, scores, 'b')
                ax.set_xlabel('Episode number')
                ax.set_ylabel('Accumulated reward')
                score_episode_fig.savefig(SCORE_FILE)
                print("episode:", epi, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
            
            target_update_count += 1 
            if target_update_count >= TARGET_UPDATE_INTERVAL:
                # update target network after fix step interval
                agent.update_target_model()
                target_update_count = 0
            
            step += 1

        # save the model
        if epi % 5 == 0:
            EPI_MODEL_PATH = os.path.join(MODEL_PATH, f'single_uav_dqn_epi_{epi}.h5')
            agent.model.save_weights(EPI_MODEL_PATH)

    
    agent.model.save_weights(FINAL_MODEL_FILE)
    flatten_env.close()
    print(f'\n\nTotal elapsed time: {time.time()-PROGRAM_START_TIME}\n')