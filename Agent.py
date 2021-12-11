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

    def step(self, env, action): # add reward to each action
        # next_state, reward, done
        vision_distances, t, speed, done = env.step(action[0], action[1])
        vision_distances = list(vision_distances)
        ### need to calculate reward here depending on the state
        reward = 0 
        for i in range(len(vision_distances)):
            reward -= abs(vision_distances[i]-vision_distances[len(vision_distances)-i-1])
        reward += 1000 * t
        reward += speed
        next_state = vision_distances[:] + [t, speed]
        return next_state, reward, done

    # approximate Q function using Neural Network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))#State is input
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))#Q_Value of each action is Output
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        q_value = self.model.predict(state) # [[q_value_0, q_value_1]]
        # return np.argmax(q_value[0])
        action = q_value[0]
        return np.interp(action, (action.min(), action.max()), (0, 1)) # squeeze into range [0,1]
    
    # save sample <state,action,reward,next_state,done> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # decay epsilon

    # take a random sample from replay memory and train the neural network
    def train_model(self):
        # print("Memory\n", self.memory)
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        state_input = np.zeros((batch_size, self.state_size))
        next_state_input = np.zeros((batch_size, self.state_size))
        action_input, reward_input, done_input = [], [], []

        for i in range(self.batch_size):
            state_input[i] = mini_batch[i][0]
            action_input.append(mini_batch[i][1])
            reward_input.append(mini_batch[i][2])
            next_state_input[i] = mini_batch[i][3]
            done_input.append(mini_batch[i][4])
    
        print("state_input \n", state_input)
        print("action_input\n", action_input)
        print("reward_input\n", reward_input)
        print("done_input\n", done_input)

        state_qvals = self.model.predict(state_input)
        print("state_qvals \n", state_qvals)

        print("next_state_input \n", next_state_input)
        next_state_qvals = self.model.predict(next_state_input)
        print("next_state_predicted \n", next_state_qvals)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done_input[i]:
                state_qvals[i][action_input[i]] = reward_input[i]
            else:
                state_qvals[i][action_input[i]] = reward_input[i] + self.discount_factor * (
                    np.amax(next_state_qvals[i]))

        # and do the model fit
        self.model.fit(state_input, state_predicted, batch_size=self.batch_size,
                       epochs=1, verbose=0)

if __name__ == '__main__':
    env = Env(20, 9, math.pi/2, 25, math.pi/2.5, 20, beziers[2], 2)
    env.reset()

    state_size = 11
    action_size = 2

    agent = Agent(state_size,action_size)

    scores, episodes = [], []
    
    for e in range(EPISODES):
        done = False
        score = 0
        state = [400, 400, 400, 400, 400, 400, 400, 400, 400, 0, 1.25]
        env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            env.render()
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = agent.step(env, action)
            next_state = np.reshape(next_state, [1, state_size])
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()

            state = next_state
            reward = reward if not done else -1000 # if car hits the wall, reward = -1000
            score += reward

        if (e % 50 == 0) & (load_model==False):
            agent.model.save_weights("RacingTrack.h5")

            



