import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import math
from RacingTrack import Env, beziers

EPISODES   = 300
load_model = False

class Agent:
    def __init__(self, state_size, action_size):
        if load_model:
            self.state_size = state_size #Get size of the state
            self.action_size = action_size #Get size of the action

            #Hyperparameters
            self.discount_factor = 0.99 #Disocunt Factor
            self.learning_rate = 0.000001 #Learning Rate

            #Hyperparameters to adjust the Exploitation-Explore tradeoff
            self.epsilon = 0.2 #Setting the epislon (0= Explore, 1= Exploit)
            self.epsilon_decay = 0.999999 #Adjusting how our epsilon will decay
            self.epsilon_min = 0.2 #Min Epsilon

            self.batch_size = 64 #Batch Size for training the neural network
            self.train_start = 1000 #If Agent's memory is less, no training is done

        else:
            self.state_size = state_size #Get size of the state
            self.action_size = action_size #Get size of the action

            #Hyperparameters
            self.discount_factor = 0.99 #Disocunt Factor
            self.learning_rate = 0.001 #Learning Rate

            #Hyperparameters to adjust the Exploitation-Explore tradeoff
            self.epsilon = 1.0 #Setting the epislon (0= Explore, 1= Exploit)
            self.epsilon_decay = 0.999 #Adjusting how our epsilon will decay
            self.epsilon_min = 0.1 #Min Epsilon

            self.batch_size = 64 #Batch Size for training the neural network
            self.train_start = 1000 #If Agent's memory is less, no training is done

        # create main replay memory for the agent using deque
        self.memory = deque(maxlen=2000)

        # create main model
        self.model = self.build_model()

        #Loading weights if load_model=True
        if load_model:
            self.model.load_weights("./RacingTrack.h5")

    def step(self, env, action, action_size): # add reward to each action
        action_holder = [0 for _ in range(action_size)]
        action_holder[action] = 1
        action = np.array(action_holder)
        vision_distances, t, speed, done = env.step(action)
        vision_distances = list(vision_distances)
        ### need to calculate reward here depending on the state
        reward = 0 
        # for i in range(len(vision_distances)):
        #     reward -= abs(vision_distances[i]-vision_distances[len(vision_distances)-i-1])
        reward += t
        reward += speed/100
        next_state = vision_distances[:] + [t, speed]
        return next_state, reward, done

    # approximate Q function using Neural Network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))#State is input
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))#Q_Value of each action is Output
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state) # [[q_value_0, q_value_1, q_value_2, q_value_3]]
            return np.argmax(q_value[0])
    
    # save sample <state,action,reward,next_state,done> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # decay epsilon

    # take a random sample from replay memory and train the neural network
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            q_update = reward
            if not done:
                q_update = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

if __name__ == '__main__':
    env = Env(20, 9, math.pi/2, 500, math.pi, 20, beziers[0], 2, 0)    
    env.reset()

    state_size = 11
    action_size = 4

    agent = Agent(state_size,action_size)

    scores, episodes = [], []
    
    for e in range(EPISODES):
        done = False
        score = 0
        state = [22.,  23.,  31.,  48., 101., 48., 31.,  23.,  21., 0, 1.5]
        env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            env.render()
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = agent.step(env, action, action_size)
            reward = reward if not done else -100 # if car hits the wall, reward = -10000
            next_state = np.reshape(next_state, [1, state_size])
            # save the sample <s, a, r, s',d> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()

            state = next_state
            
            score += reward
        print("Episode: {}, Cumulative reward: {:0.2f}".format(
            e, score))

        if (e % 50 == 0) & (load_model==False):
            agent.model.save_weights("RacingTrack.h5")

            



