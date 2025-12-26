
import os
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from go_env import GoEnv
from network import AlphaZeroNet
from mcts import MCTS

# Configuration
BOARD_SIZE = 9
NUM_WORKERS = 4 # Adjust based on CPU cores (e.g., 4-8 for M1/M2/M3)
GAMES_PER_WORKER = 2
BATCH_SIZE = 64
MCTS_SIMS = 50

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
        # Since MCTS is serial within this worker, we expect the next item 
        # in the queue to be the answer for this request.
        policy, value = output_queue.get()
        return policy, value

    all_training_data = []
    
    for i in range(num_games):
        env = GoEnv(board_size=board_size)
        mcts = MCTS(predict_callback, num_simulations=mcts_sims)
        game_history = []
        
        while not env.done:
            state = env.get_state()
            probs = mcts.search(env)
            
            # Add dirichlet noise for exploration (optional but recommended for training)
            # For simplicity, we stick to basic probabilistic sampling
            action = np.random.choice(len(probs), p=probs)
            
            game_history.append([state, probs, 0])
            env.step(action)
            
            # if worker_id == 0 and env.turn_count % 10 == 0:
            #    print(f"[Worker {worker_id}] Game {i+1} Step {env.turn_count}...", end='\r')

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
        for idx, turn in enumerate(game_history):
            player = idx % 4
            team = 0 if player in [0, 2] else 1
            
            if winner_team == -1:
                reward = 0.0
            else:
                reward = 1.0 if team == winner_team else -1.0
            
            turn[2] = reward
            all_training_data.append(turn)
            
        print(f"[Worker {worker_id}] Game {i+1}/{num_games} Finished. Turns: {env.turn_count}")

    # Send all collected data back
    result_queue.put(all_training_data)
    # Signal this worker is done (for inference server to know, though strictly not needed if we manage externally)
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
        
        # 1. Blocking wait for the first item
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
        
        # 2. Try to fill the batch with available items (non-blocking)
        # Dynamic timeout can be used, but usually 'get_nowait' is enough to grab what's pending
        while len(batch_states) < BATCH_SIZE:
            try:
                # Small timeout to allow other workers to submit
                # But for max speed, usually get_nowait is better if load is high.
                # If load is low, we might process small batches (which is fine).
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
            
        # 3. Batch Inference
        states_tensor = torch.FloatTensor(np.array(batch_states)).to(device)
        with torch.no_grad():
            logits, values = model(states_tensor)
            policy_probs = torch.exp(logits).cpu().numpy()
            value_preds = values.cpu().numpy()
            
        # 4. Distribute results
        for i, w_id in enumerate(batch_worker_ids):
            output_queues[w_id].put((policy_probs[i], value_preds[i].item()))

def train_parallel(board_size=9, epochs=10, games_per_epoch=10):
    mp.set_start_method('spawn', force=True)
    
    # Device setup
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
    # Optional: Load checkpoint
    # model.load_state_dict(torch.load("latest_model.pth"))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Calculate games per worker
    games_per_worker = max(1, games_per_epoch // NUM_WORKERS)
    print(f"Starting Parallel Training: {NUM_WORKERS} workers, {games_per_worker} games each.")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        start_time = time.time()
        
        # Setup Queues
        input_queue = mp.Queue()
        output_queues = [mp.Queue() for _ in range(NUM_WORKERS)]
        result_queue = mp.Queue()
        
        # Start Workers
        workers = []
        for i in range(NUM_WORKERS):
            p = mp.Process(target=self_play_worker, args=(
                i, input_queue, output_queues[i], result_queue, 
                games_per_worker, board_size, MCTS_SIMS
            ))
            p.start()
            workers.append(p)
            
        # Run Inference Loop (in main process)
        # This blocks until all workers send 'None'
        inference_server(model, input_queue, output_queues, device, NUM_WORKERS)
        
        # Collect Results
        training_data = []
        for _ in range(NUM_WORKERS):
            data = result_queue.get()
            training_data.extend(data)
            
        # Join workers
        for p in workers:
            p.join()
            
        duration = time.time() - start_time
        print(f"Self-Play Finished in {duration:.2f}s. Collected {len(training_data)} samples.")
        print(f"Speed: {len(training_data)/duration:.1f} samples/s")

        # Training Step
        print("Training Network...")
        model.train()
        
        # Shuffle data
        np.random.shuffle(training_data)
        
        # Mini-batch training
        batch_idx = 0
        total_loss = 0
        steps = 0
        train_batch_size = 64 # Training batch size
        
        while batch_idx < len(training_data):
            batch = training_data[batch_idx : batch_idx + train_batch_size]
            batch_idx += train_batch_size
            
            if not batch: break
            
            states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
            policies = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
            values = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred_logits, pred_values = model(states)
            pred_probs = torch.exp(pred_logits)
            
            # Loss: Policy (Cross Entropy) + Value (MSE)
            loss_policy = -torch.sum(policies * pred_logits) / len(batch) # Average over batch
            loss_value = torch.nn.functional.mse_loss(pred_values, values)
            
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        print(f"Epoch {epoch} Loss: {total_loss/steps:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), "latest_model.pth")
        
if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows (and macOS)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Context might already be set

    # Example: Run 5 epochs, 20 games per epoch
    train_parallel(epochs=5, games_per_epoch=20)
