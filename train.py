import itertools
import numpy as np
import torch
import torch.nn as nn
from utils.helper import one_hot, device

from policy import PolicyEstimator
from value import ValueEstimator
from torch.distributions import Categorical


def reinforce(env, num_episodes=5000, gamma=1, print_every=100):
    n_state_space = env.observation_space.n
    n_action_space = env.action_space.n
    policy = PolicyEstimator(n_state_space, n_action_space).to(device)
    value_estimator = ValueEstimator(n_state_space)
    value_loss = nn.MSELoss()
    optimizer_policy = torch.optim.Adam(policy.parameters())
    optimizer_value = torch.optim.Adam(value_estimator.parameters())
    eps = np.finfo(np.float32).eps.item()

    total_t = 0
    for ep in range(num_episodes):
        state = env.reset()
        state = one_hot(state, n_state_space).unsqueeze(0)
        states = [state]

        # run an entire episode
        for t in itertools.count():
            action_dist, action_logits = policy(state)
            m = Categorical(action_dist.squeeze(0))
            action = m.sample()
            policy.saved_log_probs.append(m.log_prob(action))  # is this not the same as my logits?
            action = action.item()

            state, reward, done, _ = env.step(action)
            state = one_hot(state, n_state_space).unsqueeze(0)
            states.append(state)

            policy.rewards.append(reward)
            if done or t >= 10000:
                break


        R = 0
        values=[]
        # policy_losses = []
        # rewards = []
        # advantages = []
        for state, r, log_prob in zip(states[::-1], policy.rewards[::-1], policy.saved_log_probs[::-1]):
            optimizer_value.zero_grad()
            optimizer_policy.zero_grad()

            R = r + gamma * R
            values.insert(0, R)
            val_calc = value_estimator(state)
            val_loss = value_loss(val_calc, torch.tensor([[R]], dtype=torch.float, device=device))
            val_loss.backward(retain_graph=True)
            optimizer_value.step()

            # rewards.insert(0, R)
            advantage = (R - val_calc)
            pol_loss = -log_prob.item() * advantage
            pol_loss.backward()
            optimizer_policy.step()

        # rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # rewards = (rewards - baseline)/(rewards.std() + eps)
        # for log_prob, reward in zip(policy.saved_log_probs, rewards):
        #     policy_losses.append(-log_prob * reward)
        # policy_loss = torch.stack(policy_losses, dim=0).sum()
        # policy_loss.backward()

        # if ep % print_every == 0:
        #     argmax_states = [arr.argmax(dim=1).item() for arr in states]
        #     print(policy.rewards, state.argmax(dim=1))
        #     print(list(zip(argmax_states, values)))

        del policy.rewards[:]
        del policy.saved_log_probs[:]

        if ep + 1 % print_every == 0:

            print('Episode {}\tLast length: {:5d}\tEpisode Rewards: {:.2f}'.format(ep, t + 1, R))

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break
