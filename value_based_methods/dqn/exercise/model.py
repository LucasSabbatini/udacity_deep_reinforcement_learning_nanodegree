import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        # # define the model
        # self.model = nn.Sequential(
        #     nn.Linear(state_size, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, action_size)
        # )

        self.fc1 = nn.Linear(state_size, 64)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(64, action_size)


    def preprocess_input(self, state):
        return state

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # perform the forward pass
        # x = self.preprocess_input(state)
        x = self.ac1(self.fc1(state))
        x = self.ac2(self.fc2(x))
        x = self.fc3(x)
        # x = self.model(state)
        return x

