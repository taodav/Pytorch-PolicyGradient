import gym
from train import reinforce, actor_critic
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

env = CliffWalkingEnv()

stats = reinforce(env)

# stats = actor_critic(env)

plotting.plot_episode_stats(stats, smoothing_window=25)