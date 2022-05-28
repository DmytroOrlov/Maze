from stable_baselines3.common.policies import ActorCriticPolicy  # MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO  # PPO2
from rl.agents import DQNAgent
from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from env import MazeEnv

env = DummyVecEnv([lambda: MazeEnv()])
model = PPO(ActorCriticPolicy, env, learning_rate=0.001)
model.learn(100000)

# env = MazeEnv()
# states = env.observation_space.shape
# actions = env.action_space.n

# def build_model(states, actions):
#     model = Sequential()
#     model.add(Dense(24, activation='relu', input_shape=states))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model

# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy,
#                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn

# model = build_model(states, actions)
# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
