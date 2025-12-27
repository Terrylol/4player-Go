
import os
import time
import datetime
import torch
import torch.multiprocessing as mp
import numpy as np
from go_env import GoEnv
from network import AlphaZeroNet
from mcts import MCTS

# Configuration
BOARD_SIZE = 5
NUM_WORKERS = 10      # Optimized for RTX 4060 + i5/i7 (8-12 threads)
BATCH_SIZE = 256      # Small board allows larger batch size on 8GB VRAM
MCTS_SIMS = 20        # Reduced from 30 to 20 for even faster 5x5 training
MAX_TURNS = 30        # 5x5 board max moves around 25-30
SAVE_DIR = "models_5x5"

def self_play_worker(worker_id, input_queue, output_queue, result_queue, num_games, board_size, mcts_sims):
    """
    Worker process that runs Self-Play games.
    Sends state evaluation requests to input_queue.
    Receives (policy, value) from output_queue.
    """
    np.random.seed(os.getpid() + int(time.time()))
    
    def predict_callback(state):
        # Send request
        input_queue.put((worker_id, state))
        # Wait for response (blocking)
        policy, value = output_queue.get()
        return policy, value

    all_training_data = []
    
    for i in range(num_games):
        env = GoEnv(board_size=board_size, max_turns=MAX_TURNS)
        mcts = MCTS(predict_callback, num_simulations=mcts_sims)
        game_history = []
        
        while not env.done:
            state = env.get_state()
            # Add noise only at root for the first 10 moves (shorter for 5x5)
            add_noise = (env.turn_count < 10)
            probs = mcts.search(env, add_dirichlet_noise=add_noise)
            
            action = np.random.choice(len(probs), p=probs)
            
            # Data Augmentation: Rotate state and policy 4 times
            for k in range(4):
                rot_state = np.rot90(state, k=k, axes=(1, 2)).copy()
                
                board_probs = probs[:-1].reshape(board_size, board_size)
                rot_board_probs = np.rot90(board_probs, k=k).copy()
                rot_probs = np.append(rot_board_probs.flatten(), probs[-1])
                
                game_history.append([rot_state, rot_probs, 0])

            env.step(action)
            
        # Game Over
        scores = env.get_scores()
        team0_score = scores[0]
        team1_score = scores[1]
        
        if team0_score > team1_score:
            winner_team = 0
        elif team1_score > team0_score:
            winner_team = 1
        else:
            winner_team = -1
            
        # Backfill rewards
        rewards_by_player = {}
        if winner_team == -1:
            for p in range(4): rewards_by_player[p] = 0.0
        else:
            for p in range(4):
                p_team = 0 if p in [0, 2] else 1
                rewards_by_player[p] = 1.0 if p_team == winner_team else -1.0

        for s_idx, sample in enumerate(game_history):
            turn_idx = s_idx // 4 
            player = turn_idx % 4
            sample[2] = rewards_by_player[player]
            all_training_data.append(sample)
            
        if worker_id == 0:
            print(f"[Worker {worker_id}] Game {i+1}/{num_games} Finished. Turns: {env.turn_count}")

    result_queue.put(all_training_data)
    input_queue.put((worker_id, None)) 

def inference_server(model, input_queue, output_queues, device, num_workers):
    """
    Main process function that handles GPU batch inference.
    """
    model.eval()
    workers_active = num_workers
    
    while workers_active > 0:
        batch_states = []
        batch_worker_ids = []
        
        try:
            item = input_queue.get()
        except:
            break
            
        worker_id, state = item
        if state is None:
            workers_active -= 1
            continue
            
        batch_states.append(state)
        batch_worker_ids.append(worker_id)
        
        while len(batch_states) < BATCH_SIZE:
            try:
                item = input_queue.get_nowait()
                w_id, s = item
                if s is None:
                    workers_active -= 1
                    continue
                batch_states.append(s)
                batch_worker_ids.append(w_id)
            except:
                break
        
        if not batch_states:
            continue
            
        states_tensor = torch.FloatTensor(np.array(batch_states)).to(device)
        with torch.no_grad():
            logits, values = model(states_tensor)
            policy_probs = torch.exp(logits).cpu().numpy()
            value_preds = values.cpu().numpy()
            
        for i, w_id in enumerate(batch_worker_ids):
            output_queues[w_id].put((policy_probs[i], value_preds[i].item()))

def train_parallel(board_size=5, epochs=1000, games_per_epoch=500):
    mp.set_start_method('spawn', force=True)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "latest_model.pth")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA) for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    model = AlphaZeroNet(board_size=board_size).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    games_per_worker = max(1, games_per_epoch // NUM_WORKERS)
    print(f"Starting 5x5 4060 Training: {NUM_WORKERS} workers, {games_per_worker} games each.")

    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        start_time = time.time()
        
        input_queue = mp.Queue()
        output_queues = [mp.Queue() for _ in range(NUM_WORKERS)]
        result_queue = mp.Queue()
        
        workers = []
        for i in range(NUM_WORKERS):
            p = mp.Process(target=self_play_worker, args=(
                i, input_queue, output_queues[i], result_queue, 
                games_per_worker, board_size, MCTS_SIMS
            ))
            p.start()
            workers.append(p)
            
        inference_server(model, input_queue, output_queues, device, NUM_WORKERS)
        
        training_data = []
        for _ in range(NUM_WORKERS):
            data = result_queue.get()
            training_data.extend(data)
            
        for p in workers:
            p.join()
            
        duration = time.time() - start_time
        print(f"Self-Play Finished in {duration:.2f}s. Collected {len(training_data)} samples.")
        print(f"Speed: {len(training_data)/duration:.1f} samples/s")

        print("Training Network...")
        model.train()
        np.random.shuffle(training_data)
        
        batch_idx = 0
        total_loss = 0
        steps = 0
        train_batch_size = 128 # Smaller training batch for faster updates
        
        while batch_idx < len(training_data):
            batch = training_data[batch_idx : batch_idx + train_batch_size]
            batch_idx += train_batch_size
            
            if not batch: break
            
            states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
            policies = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
            values = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred_logits, pred_values = model(states)
            
            loss_policy = -torch.sum(policies * pred_logits) / len(batch)
            loss_value = torch.nn.functional.mse_loss(pred_values, values)
            
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        print(f"Epoch {epoch} Loss: {total_loss/steps:.4f}")
        
        torch.save(model.state_dict(), model_path)
        
        if epoch % 50 == 0:
             torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pth"))
        
        elapsed = time.time() - train_start_time
        avg_time_per_epoch = elapsed / epoch
        remaining_epochs = epochs - epoch
        eta_seconds = remaining_epochs * avg_time_per_epoch
        
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        print(f"Time Elapsed: {elapsed_str} | ETA: {eta_str} | Avg/Epoch: {avg_time_per_epoch:.2f}s")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 5x5 Extreme Speed Training
    train_parallel(board_size=5, epochs=1000, games_per_epoch=100)
