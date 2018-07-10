import gym
from train import reinforce
from lib.envs.cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()

reinforce(env)
