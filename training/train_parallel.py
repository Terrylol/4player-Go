
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
NUM_WORKERS = 10 # Optimized for i5-12400 (6P cores + HT = 12 threads), leave 2 for system
GAMES_PER_WORKER = 2 # Will be dynamically calculated based on total games
BATCH_SIZE = 256 # Optimized for RTX 4060 (8GB VRAM)
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
            # Add noise only at root for the first 30 moves to encourage exploration
            add_noise = (env.turn_count < 30)
            probs = mcts.search(env, add_dirichlet_noise=add_noise)
            
            # Select action based on visit counts (probs from search)
            # In competitive play, we pick argmax. In training, we sample.
            # Temperature annealing: high temp early, low temp later?
            # For simplicity, we just sample directly from visit probabilities.
            action = np.random.choice(len(probs), p=probs)
            
            # Data Augmentation: Rotate state and policy 4 times
            # 9x9 board flattened policy: 0..80 are board positions, 81 is pass
            for k in range(4):
                # 1. Rotate State (5, 9, 9)
                # k=0: 0 deg, k=1: 90 deg, etc.
                rot_state = np.rot90(state, k=k, axes=(1, 2)).copy()
                
                # 2. Rotate Policy (82,)
                # Separate board moves and pass move
                board_probs = probs[:-1].reshape(board_size, board_size)
                rot_board_probs = np.rot90(board_probs, k=k).copy()
                rot_probs = np.append(rot_board_probs.flatten(), probs[-1])
                
                # Store
                game_history.append([rot_state, rot_probs, 0])

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
        # game_history structure: [state, policy, value]
        # We collected 4 augmented samples per turn.
        # So we need to process them in chunks of 4.
        
        # Calculate rewards for the whole game first
        # Players: 0,1,2,3. Teams: 0(AC), 1(BD)
        rewards_by_player = {}
        if winner_team == -1:
            for p in range(4): rewards_by_player[p] = 0.0
        else:
            for p in range(4):
                p_team = 0 if p in [0, 2] else 1
                rewards_by_player[p] = 1.0 if p_team == winner_team else -1.0

        # game_history contains 4 entries per actual turn
        # The turns are sequential: Turn 0 (Aug 0,1,2,3), Turn 1 (Aug 0,1,2,3)...
        for i, sample in enumerate(game_history):
            turn_idx = i // 4 # Integer division to get the actual turn number
            player = turn_idx % 4
            
            sample[2] = rewards_by_player[player]
            all_training_data.append(sample)
            
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

def arena_match(old_model, new_model, board_size, games=10, mcts_sims=50):
    """
    Compare old_model vs new_model.
    Returns: new_model_win_rate (0.0 - 1.0)
    """
    wins_new = 0
    draws = 0
    wins_old = 0
    
    # We can run this in serial for simplicity or parallelize if needed.
    # Given games=10, serial is fast enough.
    print(f"Arena: Playing {games} games (New vs Old)...")
    
    # Helper to get prediction from a model (local)
    def make_predict_func(model):
        device = next(model.parameters()).device
        def predict(state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, v = model(state_tensor)
                policy = torch.exp(logits).cpu().numpy()[0]
                value = v.item()
            return policy, value
        return predict

    pred_new = make_predict_func(new_model)
    pred_old = make_predict_func(old_model)

    for i in range(games):
        env = GoEnv(board_size=board_size)
        # Randomize who goes first: 0/2 is Team A, 1/3 is Team B
        # Let's say New Model plays Team A (P0, P2), Old Model plays Team B (P1, P3)
        # We should swap roles halfway
        
        # Simple setup:
        # Games 0-4: New=Team0, Old=Team1
        # Games 5-9: Old=Team0, New=Team1
        new_is_team0 = (i < games // 2)
        
        while not env.done:
            state = env.get_state()
            current_team = 0 if env.current_player in [0, 2] else 1
            
            # Decide who plays
            if (current_team == 0 and new_is_team0) or (current_team == 1 and not new_is_team0):
                # New Model to move
                mcts = MCTS(pred_new, num_simulations=mcts_sims)
            else:
                # Old Model to move
                mcts = MCTS(pred_old, num_simulations=mcts_sims)
            
            # Greedy action for evaluation (no noise)
            probs = mcts.search(env, add_dirichlet_noise=False)
            action = np.argmax(probs)
            env.step(action)
            
        scores = env.get_scores()
        # Team 0 score vs Team 1 score
        if scores[0] > scores[1]:
            winner_team = 0
        elif scores[1] > scores[0]:
            winner_team = 1
        else:
            winner_team = -1
            
        if winner_team == -1:
            draws += 1
        elif (winner_team == 0 and new_is_team0) or (winner_team == 1 and not new_is_team0):
            wins_new += 1
        else:
            wins_old += 1
            
    win_rate = (wins_new + 0.5 * draws) / games
    print(f"Arena Result: New {wins_new} - {wins_old} Old ({draws} Draws). Win Rate: {win_rate:.2f}")
    return win_rate

def train_parallel(board_size=9, epochs=300, games_per_epoch=100):
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
    if os.path.exists("latest_model.pth"):
        model.load_state_dict(torch.load("latest_model.pth", map_location=device))
        print("Loaded existing model.")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # LR Decay: Reduce LR by factor of 0.9 every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

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
            
        print(f"Epoch {epoch} Loss: {total_loss/steps:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Arena Evaluation: New Model vs Old Model
        # Save a temp copy of the new model
        torch.save(model.state_dict(), "temp_new_model.pth")
        
        # Load old model for comparison (if it exists, otherwise we just improved from scratch)
        if os.path.exists("latest_model.pth"):
            old_model = AlphaZeroNet(board_size=board_size).to(device)
            old_model.load_state_dict(torch.load("latest_model.pth", map_location=device))
            old_model.eval()
            
            # Run Arena
            win_rate = arena_match(old_model, model, board_size, games=10, mcts_sims=50)
            
            if win_rate >= 0.55:
                print(f"New model accepted! (Win Rate: {win_rate:.2f})")
                torch.save(model.state_dict(), "latest_model.pth")
            else:
                print(f"New model rejected. (Win Rate: {win_rate:.2f})")
                # Revert model to old state for next epoch (optional, or just keep training from here?
                # Usually we revert to keep the 'good' base.)
                model.load_state_dict(old_model.state_dict())
        else:
            print("First model saved.")
            torch.save(model.state_dict(), "latest_model.pth")
        
        # Cleanup temp file
        if os.path.exists("temp_new_model.pth"):
            os.remove("temp_new_model.pth")
            
    # Cleanup workers
    for q in input_queues:
        q.put("DONE")
    for w in workers:
        w.join()

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Windows (and macOS)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Context might already be set

    # Example: Run 300 epochs, 100 games per epoch (optimized for 20 hours)
    train_parallel(epochs=300, games_per_epoch=100)
