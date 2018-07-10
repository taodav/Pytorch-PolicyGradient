import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyEstimator(nn.Module):
    def __init__(self, num_state_space, num_action_space):
        """
        Policy estimator that takes in a discrete space observation (one hot encoding)
        :param num_state_space: number of possible states
        :param num_action_space: number of possible actions
        """
        super(PolicyEstimator, self).__init__()

        self.num_state_space = num_state_space
        self.num_action_space = num_action_space
        self.rewards = []
        self.saved_log_probs = []

        self.l1 = nn.Linear(self.num_state_space, 128)
        self.l2 = nn.Linear(128, self.num_action_space)

    def forward(self, one_hot_input):
        """
        Takes in a one-hot encoding of the state, returns probability distribution over actions
        :param one_hot_input: of size b x num_state_space
        :return: distribution over actions
        """
        output1 = F.relu(self.l1(one_hot_input))  # b x num_action_space
        output_logits = self.l2(output1)
        output = F.softmax(output_logits, dim=1)  # b x num_action_space

        return output, output_logits
