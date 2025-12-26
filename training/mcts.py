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
    def __init__(self, model_or_func, num_simulations=50, c_puct=1.0, device=None):
        """
        model_or_func: Either a torch.nn.Module or a callable function.
                       If function: expects input (state_numpy) -> returns (policy_probs_numpy, value_float)
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
        if isinstance(model_or_func, torch.nn.Module):
            self.model = model_or_func
            self.mode = 'local'
            if device:
                self.device = device
            else:
                self.device = next(model_or_func.parameters()).device
        else:
            self.predict_func = model_or_func
            self.mode = 'remote'

    def _predict(self, state):
        if self.mode == 'local':
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, v = self.model(state_tensor)
                value = v.item()
                policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
            return policy_probs, value
        else:
            # Remote/Callback mode: expect numpy arrays back
            return self.predict_func(state)

    def search(self, env, add_dirichlet_noise=False):
        root = MCTSNode()
        
        # Expand root
        state = env.get_state()
        policy_probs, value = self._predict(state)
        
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

        # Add Dirichlet noise to the root node for exploration
        if add_dirichlet_noise:
            # Alpha=0.3 is standard for Go (usually 0.03 for 19x19, 0.3 for 9x9 is okay-ish, maybe 0.15)
            # We only add noise to valid moves
            valid_indices = np.array(valid_moves)
            noise = np.random.dirichlet([0.3] * len(valid_indices))
            
            # Mix noise: 75% policy + 25% noise
            # We need to map noise back to full policy array
            noise_full = np.zeros_like(policy_probs)
            noise_full[valid_indices] = noise
            
            policy_probs = 0.75 * policy_probs + 0.25 * noise_full
            # Renormalize just in case
            policy_probs /= np.sum(policy_probs)

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
            policy_probs, value = self._predict(scratch_env.get_state())

            # Expand the node if game is not done
            if not scratch_env.done:
                valid_moves = scratch_env.get_valid_moves()
                # Mask invalid moves
                mask = np.zeros_like(policy_probs)
                for move in valid_moves:
                    mask[move] = 1
                policy_probs *= mask
                
                sum_probs = np.sum(policy_probs)
                if sum_probs > 0:
                    policy_probs /= sum_probs
                else:
                    policy_probs[valid_moves] = 1.0 / len(valid_moves)

                for move in valid_moves:
                    if move not in node.children:
                        node.children[move] = MCTSNode(node, policy_probs[move])
            
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
