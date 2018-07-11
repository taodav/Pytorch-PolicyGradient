import itertools
import numpy as np
import torch
import torch.nn as nn
from utils.helper import one_hot, device, Transition

from policy import PolicyEstimator
from value import ValueEstimator
from torch.distributions import Categorical
from lib import plotting


def reinforce(env, num_episodes=5000, gamma=1, print_every=100):
    n_state_space = env.observation_space.n
    n_action_space = env.action_space.n
    policy = PolicyEstimator(n_state_space, n_action_space).to(device)
    value_estimator = ValueEstimator(n_state_space).to(device)
    value_loss = nn.MSELoss()
    optimizer_policy = torch.optim.Adam(policy.parameters())
    optimizer_value = torch.optim.Adam(value_estimator.parameters())

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    for ep in range(num_episodes):
        state = env.reset()

        episode = []

        # Go through all steps in the episode
        for t in itertools.count():
            state_encoded = one_hot(state, n_state_space)
            action_dist, action_logits = policy(state_encoded)
            action = np.random.choice(np.arange(n_action_space), p=action_dist.squeeze(0).cpu().detach().numpy())
            next_state, reward, done, _ = env.step(action)
            next_state_encoded = one_hot(next_state, n_state_space)

            episode.append(Transition(state, action, reward, next_state, done))

            stats.episode_rewards[ep] += reward
            stats.episode_lengths[ep] = t

            if done:
                break

            state = next_state

        # for average baseline
        # baseline_value = sum(tr.reward for tr in episode) / len(episode)

        for t, transition in enumerate(episode):
            # total return at timestep t
            total_return = sum(gamma ** i * tr.reward for i, tr in enumerate(episode[t:]))

            # calculate a baseline value with estimator
            baseline_value = value_estimator(one_hot(transition.state, n_state_space))
            advantage = total_return - baseline_value

            # update value estimator if we're using it
            val_loss = value_loss(baseline_value, torch.tensor([[total_return]], dtype=torch.float, device=device, requires_grad=False))
            optimizer_value.zero_grad()
            val_loss.backward(retain_graph=True)
            optimizer_value.step()

            # we now update our policy value
            action_dist, action_logits = policy(one_hot(transition.state, n_state_space))
            policy_loss = -torch.log(action_dist.squeeze(0)[transition.action]) * advantage
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()


        print("\rStep {} @ Episode {}/{}, reward:({})".format(
            t, ep + 1, num_episodes, stats.episode_rewards[ep - 1]), end="")


    return stats



def actor_critic(env, num_episodes=5000, gamma=1, print_every=100):
    n_state_space = env.observation_space.n
    n_action_space = env.action_space.n
    policy = PolicyEstimator(n_state_space, n_action_space).to(device)
    value_estimator = ValueEstimator(n_state_space).to(device)
    value_loss = nn.MSELoss()
    optimizer_policy = torch.optim.Adam(policy.parameters())
    optimizer_value = torch.optim.Adam(value_estimator.parameters())

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    episodes = []
    for ep in range(num_episodes):
        state = env.reset()

        for t in itertools.count():
            state_encoded = one_hot(state, n_state_space)
            action_dist, action_logits = policy(state_encoded)
            action = np.random.choice(np.arange(n_action_space), p=action_dist.squeeze(0).cpu().detach().numpy())
            next_state, reward, done, _ = env.step(action)
            next_state_encoded = one_hot(next_state, n_state_space)
            episodes.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

            stats.episode_rewards[ep] += reward
            stats.episode_lengths[ep] = t

            # TD target for TD(0)
            curr_val_estimate = value_estimator(state_encoded)
            next_value = value_estimator(next_state_encoded)
            td_target = reward + gamma * next_value
            td_error = td_target - curr_val_estimate

            val_loss = value_loss(curr_val_estimate, td_target.data)
            optimizer_value.zero_grad()
            val_loss.backward(retain_graph=True)
            optimizer_value.step()

            policy_loss = -torch.log(action_dist.squeeze(0)[action]) * td_error
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, ep + 1, num_episodes, stats.episode_rewards[ep - 1]), end="")

            if done:
                break
            state = next_state
    return stats