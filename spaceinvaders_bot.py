import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def dqn_agent(env_name, steps, episodes):
	env = gym.make(env_name)
	np.random.seed(123)
	env.seed(123)
	actions = env.action_space.n
	obs_space = (1,) + env.observation_space.shape

	model = Sequential()
	model.add(Flatten(input_shape= obs_space))
	#model.add(Dense(512))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(actions))
	model.add(Activation('linear'))
	#print(model.summary())

	policy = EpsGreedyQPolicy()
	memory = SequentialMemory(limit=50000, window_length=1)
	dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
	dqn.compile(Adam(lr=1e-3), metrics=['mae'])
	#dqn.fit(env, nb_steps=steps, visualize=True, verbose=2500)
	dqn.fit(env, nb_steps=steps)
	#dqn.fit(env, nb_steps=5000)
	dqn.test(env, nb_episodes=episodes, visualize=True)


if __name__ == "__main__": 
	ENV_NAME = 'SpaceInvaders-v0'
	dqn_agent(ENV_NAME, 5000, 5)
