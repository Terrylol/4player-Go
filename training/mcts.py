import math
import numpy as np
import torch

class MCTSNode:
    def __init__(self, parent=None, prior_prob=0):
        self.parent = parent
        self.children = {} # action -> node
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, num_simulations=50, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, env):
        root = MCTSNode()
        
        # Expand root
        state = env.get_state()
        device = next(self.model.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
        
        valid_moves = env.get_valid_moves()
        # Mask invalid moves
        mask = np.zeros_like(policy_probs)
        for move in valid_moves:
            mask[move] = 1
        policy_probs *= mask
        if np.sum(policy_probs) > 0:
            policy_probs /= np.sum(policy_probs) # Renormalize
        else:
            # If all valid moves have 0 prob (unlikely with softmax), uniform
            policy_probs[valid_moves] = 1.0 / len(valid_moves)

        for move in valid_moves:
            root.children[move] = MCTSNode(root, policy_probs[move])

        # Simulations
        for _ in range(self.num_simulations):
            node = root
            scratch_env = self._clone_env(env)
            
            # Select
            while node.children:
                action, node = self._select_child(node)
                scratch_env.step(action)
            
            # Expand & Evaluate
            # (In simplified version, we just use the value from network for leaf)
            state_tensor = torch.FloatTensor(scratch_env.get_state()).unsqueeze(0).to(device)
            with torch.no_grad():
                p, v = self.model(state_tensor)
                value = v.item()
            
            # Backpropagate
            # Note: Value is for current player. 
            # If we went down N steps, we need to flip sign based on turns?
            # 4-player team: P0(+), P1(-), P2(+), P3(-)
            # This is complex in 4p. Simplified:
            while node:
                node.visit_count += 1
                node.value_sum += value 
                node = node.parent
                value = -value # Simplified team toggle
        
        # Return policy
        board_size = env.board_size
        counts = [0] * (board_size * board_size + 1)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        
        probs = np.array(counts)
        if sum(counts) > 0:
            probs = probs / sum(counts)
        return probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            q = child.value()
            score = q + u
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _clone_env(self, env):
        return env.copy()
