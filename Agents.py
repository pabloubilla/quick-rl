import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_probability as tfp
import tensorflow as tf
import torch
import torch.nn as nn

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def get_model(input_shape, output, lr):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(12, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(output, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return model


def get_model_value(input_shape, lr):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(12, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))  # Single unit for numeric output, linear activation by default
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')  # Use mean squared error for regression
    return model


'''This one is not so good, check what it does'''
class MonteCarloG:

    def __init__(self, state_shape, n_actions, learning_rate, gamma):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = get_model(state_shape, n_actions, learning_rate)

    def select_action(self, s):
        probs = self.model.predict(np.array([s]), verbose=0)[0]
        if np.isnan(probs).any():
            print('FOUND NAN')
            print(probs)
        a = np.random.choice(self.n_actions, p=probs)
        return a, probs

    def update(self, r, a, s):
        G = 0
        G_list = []
        for t in range(len(r) - 2, -1, -1):
            G = r[t+1] + G * self.gamma
            G_list.insert(0, G)
        for t in range(len(r)-1):
            with tf.GradientTape() as tape:
                p = self.model(np.array([s[t]]))
                loss = -self.gamma**t * G_list[t] * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, name):
        self.model.save(f'saved_models/{name}')

'''This one is not so good, check what it does'''
class MonteCarloG_baseline:

    def __init__(self, state_shape, n_actions, learning_rate, gamma):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_policy = get_model(state_shape, n_actions, learning_rate)
        self.model_value =  get_model_value(state_shape, learning_rate)

    def select_action(self, s):
        probs = self.model_policy.predict(np.array([s]), verbose=0)[0]
        if np.isnan(probs).any():
            print('FOUND NAN')
            print(probs)
        a = np.random.choice(self.n_actions, p=probs)
        return a, probs
    
    def get_value(self, s):
        return self.model_value.predict(s, verbose=0)[0]

    def update(self, r, a, s):
        G = 0
        G_list = []
        for t in range(len(r) - 2, -1, -1):
            G = r[t+1] + G * self.gamma
            G_list.insert(0, G)
        for t in range(len(r)-1):
            v = self.get_value(np.array([s[t]]))
            delta_t = G_list[t] - v
            with tf.GradientTape() as tape:
                p = self.model_policy(np.array([s[t]]))
                loss = -self.gamma**t * delta_t * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            grads = tape.gradient(loss, self.model_policy.trainable_variables)
            self.model_policy.optimizer.apply_gradients(zip(grads, self.model_policy.trainable_variables))
            with tf.GradientTape() as tape:
                v = self.model_value(np.array([s[t]]))
                loss = -delta_t * v
            grads = tape.gradient(loss, self.model_value.trainable_variables)
            self.model_value.optimizer.apply_gradients(zip(grads, self.model_value.trainable_variables))
    def save_model(self, name):
        self.model_policy.save(f'saved_models/{name}')


def get_model_statevalue_v2(input_shape, output, lr):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return model

'''This one is not so good, check what it does'''
class MonteCarloG_statevalue:

    def __init__(self, state_shape, n_actions, learning_rate, gamma, epsilon):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_value =  get_model_statevalue_v2(state_shape, n_actions, learning_rate)
        self.epsilon = epsilon

    def select_action(self, s):
        get_value = self.get_value(np.array([s]))
        if np.random.random() < self.epsilon:
            a = np.random.choice(self.n_actions)
        else:
            # one of the actions with the highest value
            a = np.argmax(get_value)

        # probs = self.model_policy.predict(np.array([s]), verbose=0)[0]
        # if np.isnan(probs).any():
        #     print('FOUND NAN')
        #     print(probs)
        # a = np.random.choice(self.n_actions, p=probs)
        return a, np.nan
    
    def get_value(self, s):
        return self.model_value.predict(s, verbose=0)[0]

    def update(self, r, a, s):
        G = 0
        G_list = []
        for t in range(len(r) - 2, -1, -1):
            G = r[t+1] + G * self.gamma
            G_list.insert(0, G)
        for t in range(len(r)-1):
            v = self.get_value(np.array([s[t]]))[a[t]]
            delta_t = G_list[t] - v
            # with tf.GradientTape() as tape:
            #     p = self.model_policy(np.array([s[t]]))
            #     loss = -self.gamma**t * delta_t * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            # grads = tape.gradient(loss, self.model_policy.trainable_variables)
            # self.model_policy.optimizer.apply_gradients(zip(grads, self.model_policy.trainable_variables))
            with tf.GradientTape() as tape:
                v = self.model_value(np.array([s[t]]))[0, a[t]]
                # update loss using the value function with the gr
                loss = tf.reduce_mean(G_list[t] - v)
            grads = tape.gradient(loss, self.model_value.trainable_variables)
            self.model_value.optimizer.apply_gradients(zip(grads, self.model_value.trainable_variables))
    def save_model(self, name):
        self.model_policy.save(f'saved_models/{name}')


''' This one is good
The NN receives as an input the state (whatever you want) 
and outputs the Q values for each action (expected reward from the expected state you would get with that action)
'''
class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        state_dim = state_shape[0]
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        # self.network = nn.Sequential()
        # self.network.add_module('layer1', nn.Linear(state_dim, 192))
        # self.network.add_module('relu1', nn.ReLU())
        # self.network.add_module('layer2', nn.Linear(192, 256))
        # self.network.add_module('relu2', nn.ReLU())
        # self.network.add_module('layer3', nn.Linear(256, 64))
        # self.network.add_module('relu3', nn.ReLU())
        # self.network.add_module('layer4', nn.Linear(64, n_actions))
        self.network = nn.Sequential()
        self.network.add_module('layer1', nn.Linear(state_dim, 80))
        self.network.add_module('relu3', nn.ReLU())
        # self.network.add_module('layer2', nn.Linear(50, 50))
        # self.network.add_module('relu3', nn.ReLU())
        self.network.add_module('layer4', nn.Linear(80, n_actions))
        # 
        self.parameters = self.network.parameters
        
    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.network(state_t)
        return qvalues

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = np.array(states, dtype=np.float32)
        states = torch.tensor(states, device='cpu', dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    

'''ReplayBuffer in order to store samples to learn without autocorrelation'''
class ReplayBuffer:
    def __init__(self, size):
        self.size = size #max number of items in buffer
        self.buffer =[] #array to holde buffer
        self.next_id = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
           self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size
        
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)