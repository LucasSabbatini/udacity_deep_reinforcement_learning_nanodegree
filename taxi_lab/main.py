import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.005
        self.alpha = 1
        self.gamma = 1.0

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            return None
            
        policy_s = np.ones(self.nA) * (self.epsilon/self.nA)
        policy_s[np.argmax(self.Q[next_state])] += (1-self.epsilon)
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        self.Q[state][action] += self.alpha*((reward + self.gamma*Qsa_next)-self.Q[state][action])




import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
#         self.epsilon = 0.005
        self.epsilon = 1.0
        self.alpha = 1
        self.gamma = 1.0
        
        self._episodes_count = 0
        self._epsilon_decay_rate = -3e-5

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            self._episodes_count += 1
#             self._epsilon = 1/self._episodes_count
#             print(f"Episodes count: {self._episodes_count}")
            if self.epsilon != 0.1:
                self.epsilon = 1 + self._episodes_count*self._epsilon_decay_rate 
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
            return None
            
        policy_s = np.ones(self.nA) * (self.epsilon/self.nA)
        policy_s[np.argmax(self.Q[next_state])] += (1-self.epsilon)
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        self.Q[state][action] += self.alpha*((reward + self.gamma*Qsa_next)-self.Q[state][action])