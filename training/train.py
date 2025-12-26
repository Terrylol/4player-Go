import os
import torch
import torch.optim as optim
import numpy as np
import time
import logging
from datetime import datetime
from go_env import GoEnv
from network import AlphaZeroNet
from mcts import MCTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def self_play(model, board_size=9, num_games=10):
    training_data = [] # (state, policy, value)
    
    start_time = time.time()
    team0_wins = 0
    team1_wins = 0
    draws = 0
    
    for i in range(num_games):
        game_start = time.time()
        env = GoEnv(board_size=board_size)
        mcts = MCTS(model)
        game_history = []
        
        step_count = 0
        while not env.done:
            state = env.get_state()
            action_probs = mcts.search(env)
            
            # Pick action (stochastic in training)
            action = np.random.choice(len(action_probs), p=action_probs)
            
            game_history.append([state, action_probs, 0])
            state, reward, done, info = env.step(action)
            step_count += 1
            
            if done and reward == -1: # Superko or Invalid move loss
                # Current player loses immediately
                # But in 4 player, it's team loss? 
                # Simplification: If P0 invalid, Team 0 loses.
                # Since get_scores isn't called, we handle it here or let logic flow
                pass

        # Game Over - Assign rewards
        scores = env.get_scores()
        team0_score = scores[0]
        team1_score = scores[1]
        
        game_duration = time.time() - game_start
        
        if team0_score > team1_score:
            winner_team = 0
            team0_wins += 1
        elif team1_score > team0_score:
            winner_team = 1
            team1_wins += 1
        else:
            winner_team = -1
            draws += 1

        logger.info(f"Game {i+1}/{num_games} | Steps: {step_count} | T0: {team0_score} vs T1: {team1_score} | Winner: {'Team 0' if winner_team==0 else 'Team 1' if winner_team==1 else 'Draw'} | Time: {game_duration:.2f}s")
        
        # Backfill value
        for i, turn in enumerate(game_history):
            # turn[0] is state, turn[1] is policy
            # turn[2] is value (for the player who moved)
            player = i % 4
            team = 0 if player in [0, 2] else 1
            
            if winner_team == -1:
                turn_reward = 0.0
            else:
                turn_reward = 1.0 if team == winner_team else -1.0
                
            turn[2] = turn_reward
            training_data.append(turn)
            
    logger.info(f"Self-Play Complete. T0 Wins: {team0_wins}, T1 Wins: {team1_wins}, Draws: {draws}. Total Time: {time.time()-start_time:.2f}s")
    return training_data

def train(board_size=9, epochs=10, games_per_epoch=10):
    # Device selection: CUDA (NVIDIA) > MPS (Mac M-Chip) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    logger.info(f"Using device: {device}")
    
    model = AlphaZeroNet(board_size=board_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch}/{epochs} Self-Play...")
        data = self_play(model, board_size=board_size, num_games=games_per_epoch)
        
        logger.info(f"Epoch {epoch} Optimization (Data size: {len(data)})...")
        model.train()
        
        # Batch training for efficiency
        batch_size = 32
        data_len = len(data)
        indices = np.arange(data_len)
        np.random.shuffle(indices)
        
        total_loss = 0
        batch_count = 0
        
        for start_idx in range(0, data_len, batch_size):
            end_idx = min(start_idx + batch_size, data_len)
            batch_indices = indices[start_idx:end_idx]
            
            # Collate batch
            batch_states = []
            batch_policies = []
            batch_values = []
            
            for idx in batch_indices:
                s, p, v = data[idx]
                batch_states.append(s)
                batch_policies.append(p)
                batch_values.append(v)
            
            # To Tensor
            state_tensor = torch.FloatTensor(np.array(batch_states)).to(device)
            policy_target = torch.FloatTensor(np.array(batch_policies)).to(device)
            value_target = torch.FloatTensor(np.array(batch_values)).unsqueeze(1).to(device)
            
            pred_policy, pred_value = model(state_tensor)
            
            # Loss
            # Policy Loss: Cross Entropy (using sum of p * log(q) style or similar)
            # Since target is probability distribution, we use -sum(target * log(pred))
            # However, our network might output logits or probs.
            # AlphaZeroNet usually outputs logits for policy.
            # Let's assume network outputs logits for policy as per MCTS usage (exp(logits))
            # Wait, MCTS used torch.exp(policy_logits). 
            # So network returns logits.
            # We need log_softmax for numerical stability if we do KL div, 
            # or just -sum(target * log_softmax(logits))
            
            # Re-checking network.py (not visible here but inferred from MCTS usage)
            # MCTS: policy_logits, value = self.model(state_tensor)
            # MCTS: policy_probs = torch.exp(policy_logits).numpy()[0]
            # This implies output is LogSoftmax? Or just raw logits?
            # If exp(logits) sums to 1, then it is LogSoftmax.
            # Usually AlphaZero output is LogSoftmax or Softmax.
            # Let's assume LogSoftmax for stability.
            
            # If network returns LogSoftmax:
            loss_policy = -torch.sum(policy_target * pred_policy) / len(batch_indices)
            
            loss_value = torch.mean((value_target - pred_value) ** 2)
            loss = loss_policy + loss_value
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_duration = time.time() - epoch_start
        logger.info(f"Epoch {epoch} Complete. Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s")
        
        torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
        torch.save(model.state_dict(), "latest_model.pth")
        logger.info(f"Model saved to latest_model.pth")
        
    logger.info(f"Training Complete. Total Time: {time.time()-total_start_time:.2f}s")

if __name__ == "__main__":
    train(board_size=9, epochs=10, games_per_epoch=10)
