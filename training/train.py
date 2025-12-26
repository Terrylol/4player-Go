import os
import torch
import torch.optim as optim
import numpy as np
from go_env import GoEnv
from network import AlphaZeroNet
from mcts import MCTS

def self_play(model, board_size=9, num_games=10):
    training_data = [] # (state, policy, value)
    
    for i in range(num_games):
        env = GoEnv(board_size=board_size)
        mcts = MCTS(model)
        game_history = []
        
        while not env.done:
            state = env.get_state()
            action_probs = mcts.search(env)
            
            # Pick action (stochastic in training)
            action = np.random.choice(len(action_probs), p=action_probs)
            
            game_history.append([state, action_probs, 0])
            env.step(action)
        
        # Game Over - Assign rewards
        scores = env.get_scores()
        team0_score = scores[0]
        team1_score = scores[1]
        
        print(f"Game {i+1}/{num_games} Over. Scores: Team0(AC)={team0_score}, Team1(BD)={team1_score}")
        
        if team0_score > team1_score:
            winner_team = 0
        elif team1_score > team0_score:
            winner_team = 1
        else:
            # Draw
            winner_team = -1

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
            
    return training_data

def train(board_size=9, epochs=10, games_per_epoch=10):
    # Device selection: CUDA (NVIDIA) > MPS (Mac M-Chip) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    model = AlphaZeroNet(board_size=board_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} Self-Play...")
        data = self_play(model, board_size=board_size, num_games=games_per_epoch)
        
        print(f"Epoch {epoch} Optimization (Data size: {len(data)})...")
        model.train()
        
        # Batch training could be added here for efficiency
        # For simplicity, we just iterate all data (shuffle in real impl)
        import random
        random.shuffle(data)
        
        total_loss = 0
        for state, policy_target, value_target in data:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy_target = torch.FloatTensor(policy_target).unsqueeze(0).to(device)
            value_target = torch.FloatTensor([value_target]).unsqueeze(0).to(device)
            
            pred_policy, pred_value = model(state_tensor)
            
            loss_policy = -torch.sum(policy_target * pred_policy)
            loss_value = torch.mean((value_target - pred_value) ** 2)
            loss = loss_policy + loss_value
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if len(data) > 0:
            print(f"Epoch {epoch} Loss: {total_loss / len(data):.4f}")
        else:
            print(f"Epoch {epoch} Loss: 0.0000 (No data)")
            
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoint_{epoch}.pth")
        torch.save(model.state_dict(), "latest_model.pth")
        print(f"Epoch {epoch} Complete. Saved to latest_model.pth")

if __name__ == "__main__":
    train(board_size=9, epochs=10, games_per_epoch=10)
