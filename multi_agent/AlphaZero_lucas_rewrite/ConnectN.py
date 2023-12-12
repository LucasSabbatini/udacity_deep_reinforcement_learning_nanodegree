import numpy as np


def get_runs(v, i):
    """
    Identifies continuous runs of a specified value in a vector.

    Parameters:
    - v (np.array): The vector in which runs are to be identified.
    - i (int): The value of interest to identify runs for.

    Returns:
    - tuple: Contains three arrays, representing the start indices, end indices,
             and the lengths of the runs of the value 'i' in the vector 'v'.
    """
    bounded = np.hstack(([0], v == i, [0]))
    difs = np.diff(bounded)
    starts, = np.where(difs > 0)
    ends, = np.where(difs < 0)
    return starts, ends, ends - starts

def in_a_row(v, N, i):
    """
    Checks if there's a continuous run of a specific value of at least length N in a vector.

    Parameters:
    - v (np.array): The vector to be checked.
    - N (int): The minimum length of the run to be identified.
    - i (int): The value of interest to identify runs for.

    Returns:
    - bool: True if there is at least one run of value 'i' of length N or more, False otherwise.
    """
    if len(v) < N:
        return False
    else:
        _, _, total = get_runs(v, i)
        return np.any(total >= N)

def get_lines(matrix, loc):
    """
    Extracts horizontal, vertical, and diagonal lines from a 2D matrix based on a specific location.

    Parameters:
    - matrix (np.array): The 2D matrix from which lines are extracted.
    - loc (tuple): A tuple (i, j) representing the location in the matrix around which lines are extracted.

    Returns:
    - tuple: Contains four arrays representing the horizontal, vertical, and two diagonal lines passing through 'loc'.
    """
    i, j = loc
    flat = matrix.reshape(-1, *matrix.shape[2:])

    w = matrix.shape[0]
    h = matrix.shape[1]
    
    def flat_pos(pos):
        return pos[0] * h + pos[1]

    pos = flat_pos((i, j))

    # Index for flipping matrix across different axes
    ic = w - 1 - i
    jc = h - 1 - j

    # Top left and bottom right for diagonal
    tl = (i - j, 0) if i > j else (0, j - i)
    tl = flat_pos(tl)

    br = (w - 1 - (ic - jc), h - 1) if ic > jc else (w - 1, h - 1 - (jc - ic))
    br = flat_pos(br)

    # Horizontal and vertical lines
    hor = matrix[:, j]
    ver = matrix[i, :]

    # Diagonal lines
    diag_right = np.concatenate([flat[tl:pos:h + 1], flat[pos:br + 1:h + 1]])
    diag_left = np.concatenate([flat[br:pos:h - 1], flat[pos:tl + 1:h - 1]])

    return hor, ver, diag_right, diag_left


class ConnectN:
     def __init__(self, size, N, pie_rule=False):
          """
          Initializes the ConnectN game board.

          Parameters:
          - size (tuple): A tuple (width, height) representing the dimensions of the game board.
          - N (int): The number of consecutive symbols needed to win.
          - pie_rule (bool): Determines whether the 'pie rule' is in effect. If True, the second player can switch sides after the first move.

          Raises:
          - ValueError: If the game board dimensions or winning condition N are invalid.
          """
          self.size = size
          self.w, self.h = size
          self.N = N
          
          # make sure game is well defined
          if self.w<0 or self.h<0 or self.N<2 or \
             (self.N > self.w and self.N > self.h):
               raise ValueError('Game cannot initialize with a {0:d}x{1:d} grid, and winning condition {2:d} in a row'.format(self.w, self.h, self.N))

          self.score = None
          self.state = np.zeros(size, dtype=np.float)
          self.player = 1
          self.last_move = None
          self.n_moves = 0
          self.pie_rule = pie_rule
          self.switched_side = False

     def __copy__(self):
          """
          Creates a copy of the current game state.

          Returns:
          - ConnectN: A new instance of ConnectN with the same state as the current game.
          """
          cls = self.__class__
          new_game = cls.__new__(cls)
          new_game.__dict__.update(self.__dict__)
          new_game.N = self.N
          new_game.pie_rule = self.pie_rule
          new_game.state = self.state.copy()
          new_game.switched_side = self.switched_side
          new_game.n_moves = self.n_moves
          new_game.last_move = self.last_move
          new_game.player = self.player
          new_game.score = self.score
          return new_game

     def get_score(self):
          """
          Determines the current score of the game.

          Returns:
          - int or None: The player number who has won (1 or -1), or None if the game is still ongoing.
          """
          
          if self.n_moves < 2*self.N-1:
               return None

          i, j = self.last_move
          hor, ver, diag_right, diag_left = get_lines(self.state, (i, j))
     
          for line in [ver, hor, diag_right, diag_left]:
               if in_a_row(line, self.N, self.player):
                    return self.player

     def get_winning_loc(self):
          """
          Identifies the locations on the board that contribute to the winning condition.

          Returns:
          - np.array or list: An array of indices that form the winning line, or an empty list if no player has won yet.
          """
          if self.n_moves < 2*self.N-1:
               return None

          loc = self.last_move
          hor, ver, diag_right, diag_left = get_lines(self.state, loc)
          ind = np.indices(self.state.shape)
          ind = np.moveaxis(ind, 0, -1)
          hor_ind, ver_ind, diag_right_ind, diag_left_ind = get_lines(ind, loc)
          # loop over each possibility
        
          pieces = [hor, ver, diag_right, diag_left]
          indices = [hor_ind, ver_ind, diag_right_ind, diag_left_ind]
        
          #winning_loc = np.full(self.state.shape, False, dtype=bool)
        
          for line, index in zip(pieces, indices):
               starts, ends, runs = get_runs(line, self.player)

               # get the start and end location
               winning = (runs >= self.N)
               print(winning)
               if not np.any(winning):
                    continue
            
               starts_ind = starts[winning][0]
               ends_ind = ends[winning][0]
               indices = index[starts_ind:ends_ind]
               #winning_loc[indices[:,0], indices[:,1]] = True
               return indices
            
          return []

     def move(self, loc):
          """
          Executes a move on the board.

          Parameters:
          - loc (tuple): The location (i, j) where the move is to be made.

          Returns:
          - bool: True if the move was successful, False otherwise.
          """
          i, j = loc
          success = False
          
          if self.w > i >= 0 and self.h>j>=0:
               if self.state[i, j] == 0:
                    self.state[i,j] = self.player
                    
                    if self.pie_rule:
                         if self.n_moves==1:
                              self.state[tuple(self.last_move)] = -self.player
                              self.switched_side = False
                    
                         elif self.n_moves==0:
                              self.state[i, j] = self.player/2.0
                              self.switched_side = True
          
                    success = True
                    
               elif self.pie_rule and self.state[i, j] == -self.player/2.0:
                    self.state[i, j] = self.player
                    self.switched_side = True
                    success = True
                    
          if success:
               self.n_moves += 1
               self.last_move = tuple((i, j))
               self.score = self.get_score()
               
               if self.score is None:
                    self.player *= -1
                    
               return True

          return False

     def available_moves(self):
          """
          Identifies all the available moves on the board.

          Returns:
          - np.array: An array of indices where moves can be made.
          """
          indices = np.moveaxis(np.indices(self.state.shape), 0, -1)
          return indices[np.abs(self.state) != 1]

     def available_mask(self):
          """
          Returns a mask of the board indicating available moves.

          Returns:
          - np.array: A binary array where 1 indicates an available move and 0 indicates an occupied space.
          """
          return (np.abs(self.state) != 1).astype(np.uint8)
