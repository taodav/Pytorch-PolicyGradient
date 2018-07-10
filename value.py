import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueEstimator(nn.Module):
    def __init__(self, num_state_space):
        """
        Value estimator that takes in a discrete space observation (one hot encoding)
        :param num_state_space: number of possible states
        """
        super(ValueEstimator, self).__init__()

        self.num_state_space = num_state_space

        self.l1 = nn.Linear(self.num_state_space, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, state):

        output1 = F.relu(self.l1(state))
        output = self.l2(output1)

        return output