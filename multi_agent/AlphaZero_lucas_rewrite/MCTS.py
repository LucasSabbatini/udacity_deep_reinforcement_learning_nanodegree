import torch
import random
import copy

# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# device ='cpu'

# Define transformations for game states (rotations and reflections)
t0= lambda x: x
t1= lambda x: x[:,::-1].copy()
t2= lambda x: x[::-1,:].copy()
t3= lambda x: x[::-1,::-1].copy()
t4= lambda x: x.T
t5= lambda x: x[:,::-1].T.copy()
t6= lambda x: x[::-1,:].T.copy()
t7= lambda x: x[::-1,::-1].T.copy()

tlist=[t0, t1,t2,t3,t4,t5,t6,t7]
tlist_half=[t0,t1,t2,t3]

def flip(x, dim):
    """
    Flips a tensor along a specified dimension.

    Parameters:
    - x (Tensor): The tensor to be flipped.
    - dim (int): The dimension along which to flip the tensor.

    Returns:
    - Tensor: The flipped tensor.
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

# Inverse transformations for undoing the original transformations
t0inv= lambda x: x
t1inv= lambda x: flip(x,1)
t2inv= lambda x: flip(x,0)
t3inv= lambda x: flip(flip(x,0),1)
t4inv= lambda x: x.t()
t5inv= lambda x: flip(x,0).t()
t6inv= lambda x: flip(x,1).t()
t7inv= lambda x: flip(flip(x,0),1).t()

tinvlist = [t0inv, t1inv, t2inv, t3inv, t4inv, t5inv, t6inv, t7inv]
tinvlist_half=[t0inv, t1inv, t2inv, t3inv]

transformation_list = list(zip(tlist, tinvlist))
transformation_list_half = list(zip(tlist_half, tinvlist_half))


def process_policy(policy, game):
    if game.size[0]==game.size[1]:
        t, tinv = random.choice(transformation_list)
    else:
        t, tinv = random.choice(transformation_list_half)

    frame = torch.tensor(t(game.state*game.player), dtype=torch.float) # convert to tensor
    input = frame.unsqueeze(0).unsqueeze(0) # add batch and channel dimension
    prob, v = policy(input)
    mask = torch.tensor(game.available_mask())
    return game.available_moves(), tinv(prob)[mask].view(-1), v.squeeze().squeeze()
        
        
        
class Node:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) for game AI.

    Attributes:
    - game (ConnectN): The current state of the game at this node.
    - mother (Node, optional): The parent node in the MCTS.
    - child (dict): Dictionary mapping actions to child nodes.
    - U (float): Upper Confidence Bound used for action selection.
    - prob (Tensor): Probability of selecting this node, as determined by the policy network.
    - nn_v (Tensor): Predicted value of this node, as estimated by the neural network.
    - N (int): Number of visits to this node.
    - V (float): Expected value of this node, updated as the MCTS progresses.
    - outcome (int or None): The outcome of the game at this node, if the game has ended.

    Methods:
    - __init__: Constructor for the Node class.
    - create_children: Expands the node by creating child nodes.
    - explore: Explores the MCTS from this node.
    - next: Selects the next action based on the exploration results.
    - detach_mother: Detaches this node from its parent, aiding in garbage collection.
    """
    def __init__(self, game, mother=None, prob=torch.tensor(0., dtype=torch.float)):
        """
        Initializes a Node in the MCTS.

        Parameters:
        - game (ConnectN): The current game state associated with this node.
        - mother (Node, optional): The parent node in the tree. Defaults to None.
        - prob (Tensor): The probability of reaching this node as determined by the policy network.
        """
        self.game = game
        self.mother = mother
        self.child = {}
        
        self.U = 0 # numbers for determining which actions to take next
        self.prob = prob # V from neural net output
        self.nn_v = torch.tensor(0., dtype=torch.float) # predicted expectation (state value) from the nn
        
        self.N = 0 # visit count
        self.V = 0 # expected V from MCTS
        
        self.outcome = self.game.score
        
        if self.game.score is not None:
            self.V = self.game.score*self.game.player
            self.U = 0 if self.game.score is 0 else self.V*float("inf") # WHY?

    def create_children(self, actions, probs):
        """
        Creates child nodes for each possible action from the current game state.

        Parameters:
        - actions (list): A list of possible actions from the current game state.
        - probs (Tensor): Probabilities corresponding to each action, as predicted by the neural network.
        """
        games = [copy(self.game) for a in actions]
        for action, game in zip(actions, games):
            game.move(action)
        children = {tuple(a):Node(g, self, p) for a, g, p in zip(actions, games, probs)}
        self.children = children

    def explore(self, policy):
        """
        Explores the MCTS tree starting from this node.

        Parameters:
        - policy (callable): The policy function used to evaluate game states.

        Raises:
        - ValueError: If the game has already ended.
        """
        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))
        
        current = self
        
        ############################################################
        # Actual search starts here: choose based on max U
        ############################################################
        while current.children and current.outcome is None:
            children = current.children
            max_U = max(c.U for c in children.values())
            actions = [a for a,c in children.items() if c.U == max_U] # states with max_U
            if len(actions) == 0:
                print("error zero length ", max_U)
                print(current.game.state)
            
            action = random.choice(actions) # If there are more than one max_U, then choose one randomly
            
            # check if terminal state
            if max_U == -float("inf"): # child loss
                current.U = float("inf") # if child max_U is -inf, then current max_U is inf, because current.U = -child.U
                current.V = 1.0
                break
            elif max_U == float("inf"): # child win
                current.U = -float("inf")
                current.V = -1.0
                break
                
            current = children[action] # at the end, reach leaf node
        
        #############################################
        # If node hasn't been expanded yet, expand it
        #############################################
        if not current.child:
            next_actions, probs, v = process_policy(policy, current.game) # Only use the policy once for each node, when it is first expanded
            current.nn_v = -v # unlike probs, V is added when its children are expanded, instead of when it is created
            current.create_children(next_actions, probs)
            current.V = -float(v)
        
        self.N += 1
        
        #############################################
        # BACKPROPAGATION -> UPDATE U VALUES
        #############################################
        while current.mother:
            mother = current.mother
            mother.N += 1
            mother.V += (-current.V - mother.V)/mother.N # update mother's V - V IS ONLU USED FOR UPDATING U, AND NOT FOR THE LOSS, BECAUSE WE USE THE GAME OUTCOME
            
            for sibling in mother.children.values():
                if sibling.U is not float("inf") and sibling.U is not -float("inf"): # Only update if sibling is not terminal state
                    # This is the UBC1 formula, but with a factor of 2. 
                    # The NN output for V is the exploitation term, and the NN output prob is in the exploration term.
                    sibling.U = sibling.V + 2*sibling.prob*torch.sqrt(mother.N)/(1+sibling.N)
                    
            current = mother # goes back from leaf node to root node

    def next(self, temperature=0.1):
        """
        Determines the next action from this node based on the exploration data.

        Parameters:
        - temperature (float): Parameter controlling the level of exploration. A lower temperature favors exploitation.

        Returns:
        - tuple: Next state node, along with various statistics like node value, neural network value, and action probabilities.

        Raises:
        - ValueError: If the game has ended or no children are found.
        """
        #############################################
        # CHECK FOR ERRORS
        #############################################
        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))
        if not self.child:
            print(self.game.state)
            raise ValueError("No children found and game hasn't ended")
        
        
        #############################################
        # CHECK FOR TERMINAL STATE
        #############################################
        children = self.children
        max_U = max(c.U for c in children.values())
        if max_U == float("inf"):
            prob = torch.tensor([1.0 if c.U == float("inf") else 0 for c in children.values()], device=device)
        
        #############################################
        # SET PROBABILITY FOR NEXT STATE BASED ON THE NUMBER OF VISITS N
        #############################################
        else:
            maxN = max(node.N for node in children.values()) + 1
            prob = torch.tensor([(node.N/maxN)**(1/temperature) for node in children.values()], device=device)
            
        # normalize prob
        if torch.sum(prob) > 0:
            prob /= torch.sum(prob)
        else:
            prob = torch.tensor(1.0/len(children), device=device).repeat(len(children))
        # nn output probs
        nn_prob = torch.stack([node.prob for node in children.values()]).to(device)
        
        #############################################
        # SELECT NEXT STATE
        #############################################
        next_state = random.choises(list(children.values()), weights=prob)[0] # So next state is chosen using the number of visits N, and NOT the nn output probs
        return next_state, (-self.V, -self.nn_v, prob, nn_prob)

    def detach_mother(self):
        """
        Detaches this node from its parent node.

        This method is useful for garbage collection, as it helps in removing references to parent nodes, which may no longer be needed.
        """
        del self.mother
        self.mother = None