import numpy as np

# Constants
BOARD_SIZE = 9
NUM_PLAYERS = 4
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}  # 0=A(Team0), 1=B(Team1), 2=C(Team0), 3=D(Team1)
# 0: Empty, 1: P0, 2: P1, 3: P2, 4: P3
EMPTY = 0

class GoEnv:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 0
        self.turn_count = 0
        self.prisoners = {0: 0, 1: 0, 2: 0, 3: 0}
        self.consecutive_passes = 0
        self.history = set() # Store board states to prevent loops (Superko)
        self.done = False
        self.history.add(self.board.tobytes())
        return self.get_state()

    def copy(self):
        """Fast copy of the environment"""
        new_env = GoEnv(self.board_size)
        new_env.board = np.copy(self.board)
        new_env.current_player = self.current_player
        new_env.turn_count = self.turn_count
        new_env.prisoners = self.prisoners.copy()
        new_env.consecutive_passes = self.consecutive_passes
        new_env.history = self.history.copy()
        new_env.done = self.done
        return new_env
        # Return state representation for Neural Network
        # Channels: 
        # 0: Current Player stones
        # 1: Teammate stones
        # 2: Next Enemy stones
        # 3: Prev Enemy stones
        # 4: All stones (binary)
        state = np.zeros((5, self.board_size, self.board_size), dtype=np.float32)
        
        p = self.current_player
        teammate = (p + 2) % 4
        enemy1 = (p + 1) % 4
        enemy2 = (p + 3) % 4
        
        state[0] = (self.board == (p + 1))
        state[1] = (self.board == (teammate + 1))
        state[2] = (self.board == (enemy1 + 1))
        state[3] = (self.board == (enemy2 + 1))
        state[4] = (self.board > 0)
        
        return state

    def get_valid_moves(self):
        # Simplification: Allow all empty spots not strictly suicide
        # For full rules, need true suicide check and ko check
        moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y, x] == 0:
                    # TODO: Add suicide and Ko check here for strict compliance
                    moves.append(y * self.board_size + x)
        moves.append(self.board_size * self.board_size) # Pass
        return moves

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        # Pass
        if action == self.board_size * self.board_size:
            self.consecutive_passes += 1
            if self.consecutive_passes >= 4:
                self.done = True
            self.current_player = (self.current_player + 1) % 4
            self.turn_count += 1
            return self.get_state(), 0, self.done, {}

        self.consecutive_passes = 0
        y, x = divmod(action, self.board_size)
        
        if self.board[y, x] != 0:
            # Invalid move (should be masked out by get_valid_moves in real training)
            return self.get_state(), -1, True, {'error': 'Invalid move'}

        # Place stone
        self.board[y, x] = self.current_player + 1
        
        # Check captures (neighbors of enemy/teammate colors)
        captured_count = self._handle_captures(x, y)
        
        # Check self-suicide (if no liberties after capture resolution)
        if not self._has_liberties(x, y):
            # Undo move (Suicide is invalid)
            self.board[y, x] = 0
            # For RL, we might treat this as a loss or just force pass
            # Here we just treat it as a pass but punish?
            # Let's assume invalid move logic handles this before step
            pass 

        # Superko Check: If state repeats, treat as invalid move (loss)
        board_bytes = self.board.tobytes()
        if board_bytes in self.history:
             # Undo move to keep state clean (optional, but good for rendering)
             # self.board[y, x] = 0 # Actually, let's just end game as loss
             return self.get_state(), -1, True, {'error': 'Superko violation'}
        self.history.add(board_bytes)

        self.prisoners[self.current_player] += captured_count
        self.current_player = (self.current_player + 1) % 4
        self.turn_count += 1
        
        if self.turn_count > self.board_size * self.board_size * 4: # Max turn limit
            self.done = True

        return self.get_state(), 0, self.done, {}

    def _handle_captures(self, x, y):
        # Check neighbors
        neighbors = self._get_neighbors(x, y)
        total_captured = 0
        
        for nx, ny in neighbors:
            stone = self.board[ny, nx]
            if stone != 0 and stone != (self.current_player + 1):
                # Check if this group has 0 liberties
                if not self._check_group_liberties(nx, ny):
                    total_captured += self._remove_group(nx, ny)
        return total_captured

    def _check_group_liberties(self, x, y):
        group_color = self.board[y, x]
        if group_color == 0: return True
        
        stack = [(x, y)]
        visited = set()
        visited.add((x, y))
        
        while stack:
            cx, cy = stack.pop()
            
            # Check neighbors for liberties
            for nx, ny in self._get_neighbors(cx, cy):
                if self.board[ny, nx] == 0:
                    return True # Found a liberty
                if self.board[ny, nx] == group_color and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
        return False

    def _remove_group(self, x, y):
        group_color = self.board[y, x]
        stack = [(x, y)]
        removed = 0
        self.board[y, x] = 0
        removed += 1
        
        while stack:
            cx, cy = stack.pop()
            for nx, ny in self._get_neighbors(cx, cy):
                if self.board[ny, nx] == group_color:
                    self.board[ny, nx] = 0
                    removed += 1
                    stack.append((nx, ny))
        return removed

    def _has_liberties(self, x, y):
        return self._check_group_liberties(x, y)

    def _get_neighbors(self, x, y):
        n = []
        if x > 0: n.append((x-1, y))
        if x < self.board_size-1: n.append((x+1, y))
        if y > 0: n.append((x, y-1))
        if y < self.board_size-1: n.append((x, y+1))
        return n

    def get_scores(self):
        # Calculate scores using Area Scoring (Stones + Territory)
        # With special 4-player weak alliance rules:
        # - Mixed AC territory is invalid (0 points)
        # - Mixed BD territory is invalid (0 points)
        # - Mixed Enemy territory is Neutral (0 points)
        
        # 1. Count stones on board
        scores = {0: 0, 1: 0, 2: 0, 3: 0} # P0, P1, P2, P3
        for y in range(self.board_size):
            for x in range(self.board_size):
                stone = self.board[y, x]
                if stone != 0:
                    scores[stone - 1] += 1

        # 2. Calculate territory
        visited = set()
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y, x] == 0 and (x, y) not in visited:
                    # Found an empty region, start Flood Fill
                    region_points = []
                    neighbors = set()
                    
                    stack = [(x, y)]
                    visited.add((x, y))
                    region_points.append((x, y))
                    
                    while stack:
                        cx, cy = stack.pop()
                        
                        for nx, ny in self._get_neighbors(cx, cy):
                            n_stone = self.board[ny, nx]
                            if n_stone == 0:
                                if (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    region_points.append((nx, ny))
                                    stack.append((nx, ny))
                            else:
                                neighbors.add(n_stone - 1) # Store player index (0-3)
                    
                    # Analyze neighbors to determine ownership
                    # Neighbors set contains player indices surrounding this region
                    
                    if len(neighbors) == 0:
                        # Whole board empty? No points.
                        continue
                        
                    # Check if all neighbors belong to same Team?
                    # Team 0: {0, 2}, Team 1: {1, 3}
                    
                    team0_neighbors = {p for p in neighbors if p in [0, 2]}
                    team1_neighbors = {p for p in neighbors if p in [1, 3]}
                    
                    if len(team0_neighbors) > 0 and len(team1_neighbors) > 0:
                        # Mixed enemies -> Neutral (0 points)
                        continue
                    
                    if len(team0_neighbors) > 0:
                        # Only Team 0 surrounds
                        if len(team0_neighbors) == 1:
                            # Pure single player territory (A or C)
                            owner = list(team0_neighbors)[0]
                            scores[owner] += len(region_points)
                        else:
                            # Mixed A and C -> Invalid (0 points)
                            pass
                            
                    elif len(team1_neighbors) > 0:
                        # Only Team 1 surrounds
                        if len(team1_neighbors) == 1:
                            # Pure single player territory (B or D)
                            owner = list(team1_neighbors)[0]
                            scores[owner] += len(region_points)
                        else:
                            # Mixed B and D -> Invalid (0 points)
                            pass

        # Return Team Scores
        team0_score = scores[0] + scores[2]
        team1_score = scores[1] + scores[3]
        
        return {0: team0_score, 1: team1_score}
