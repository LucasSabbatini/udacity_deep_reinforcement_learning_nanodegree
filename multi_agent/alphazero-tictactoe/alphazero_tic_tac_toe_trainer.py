
LR = 1.e-4
WEIGHT_DECAY = 1.e-4

# Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
import random

from ConnectN import ConnectN
import torch.optim as optim

from copy import copy
import random





class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size,32)

        # layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)
        
        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

        
    def forward(self, x):

        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))
        
        
        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        avail = avail.view(-1, 9)
        
        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)
        
        
        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))
        return prob.view(3,3), value


def Policy_Player_MCTS(game):
    mytree = MCTS.node(copy(game))
    for _ in range(50):
        mytree.explore(policy)
    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)
    return mytreenext.game.last_move


def Random_Player(game):
    return random.choice(game.available_moves())




game_setting = {"size": (3,3), "N": 3}
game = ConnectN(**game_setting)

from Play import Play
gameplay = Play(ConnectN(**game_setting), 
              player1=None, 
              player2=Policy_Player_MCTS)




game=ConnectN(**game_setting)
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=.01, weight_decay=1.e-4)


# TRAINING

from collections import deque
import MCTS

episodes = 400
outcomes = []
losses = []

# progress bar
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()


for e in range(episodes):
    mytree = MCTS.Node(ConnectN(**game_setting))
    vterm = []
    logterm = []
    
    while mytree.outcome is None:
        for _ in range(50):
            mytree.explore(policy)
            
        current_player = mytree.game.player
        mytree, (v, nn_v, p, nn_p) = mytree.next()
        mytree.detach_mother()
        
        
        loglist = torch.log(nn_p)*p
        
        constant = torch.where(p > 0, p*torch.log(p), torch.tensor(0.))