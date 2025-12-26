import type { GameState, Player } from './types';
import { PLAYER_CONFIG } from './types';
import * as ort from 'onnxruntime-web';

// Cache sessions by board size
const sessionCache: Record<number, ort.InferenceSession> = {};

// This class handles the inference using the trained ONNX model
export class NeuralAI {
    player: Player;
    team: 'AC' | 'BD';

    constructor(player: Player) {
        this.player = player;
        this.team = PLAYER_CONFIG[player].team;
    }

    static async loadModel(size: number) {
        if (sessionCache[size]) return sessionCache[size];
        
        try {
            // Configure for WebAssembly execution
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
            // Try to load model for specific size
            // Expects models at /models/model_9x9.onnx, etc.
            const modelPath = `/models/model_${size}x${size}.onnx`;
            const session = await ort.InferenceSession.create(modelPath);
            sessionCache[size] = session;
            console.log(`Neural AI Model ${size}x${size} loaded successfully`);
            return session;
        } catch (e) {
            console.error(`Failed to load Neural AI model for size ${size}:`, e);
            return null;
        }
    }

    // Convert GameState to Tensor (1, 5, size, size)
    // Channels: [MyStones, TeammateStones, NextEnemy, PrevEnemy, AllStones]
    createInputTensor(gameState: GameState): ort.Tensor {
        const size = gameState.boardSize;
        const data = new Float32Array(5 * size * size);
        const pIndex = ['A', 'B', 'C', 'D'].indexOf(this.player);
        
        const myIdx = pIndex;
        const teammateIdx = (pIndex + 2) % 4;
        const nextEnemyIdx = (pIndex + 1) % 4;
        const prevEnemyIdx = (pIndex + 3) % 4;
        
        const players = ['A', 'B', 'C', 'D'];

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const stone = gameState.board[y][x];
                const idx = y * size + x;
                
                // Channel offsets
                const c0 = 0 * (size * size);
                const c1 = 1 * (size * size);
                const c2 = 2 * (size * size);
                const c3 = 3 * (size * size);
                const c4 = 4 * (size * size);

                if (stone) {
                    data[c4 + idx] = 1.0; // All stones channel
                    
                    if (stone === players[myIdx]) data[c0 + idx] = 1.0;
                    else if (stone === players[teammateIdx]) data[c1 + idx] = 1.0;
                    else if (stone === players[nextEnemyIdx]) data[c2 + idx] = 1.0;
                    else if (stone === players[prevEnemyIdx]) data[c3 + idx] = 1.0;
                }
            }
        }
        
        return new ort.Tensor('float32', data, [1, 5, size, size]);
    }

    async getBestMove(gameState: GameState): Promise<{ x: number, y: number } | null> {
        const size = gameState.boardSize;
        const session = await NeuralAI.loadModel(size);
        
        if (!session) {
            // Fallback or return null to let SimpleAI handle it?
            // For now return null, caller should handle fallback
            return null;
        }

        const inputTensor = this.createInputTensor(gameState);
        
        try {
            const results = await session.run({ input: inputTensor });
            const policy = results.policy.data as Float32Array; // Flattened array of probs
            
            // Find best legal move
            // Policy size: size*size + 1 (last is pass)
            
            let bestMove = -1;
            let bestScore = -Infinity;
            
            // Iterate all board points
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    // Check legality (simple empty check here, real game needs ko/suicide check)
                    if (gameState.board[y][x] === null) {
                        const idx = y * size + x;
                        const score = policy[idx];
                        if (score > bestScore) {
                            bestScore = score;
                            bestMove = idx;
                        }
                    }
                }
            }
            
            // Check Pass score (last element)
            const passScore = policy[size * size];
            if (passScore > bestScore) {
                // AI decides to pass
                return null; // Logic needs to handle pass vs no move found
            }
            
            if (bestMove !== -1) {
                const y = Math.floor(bestMove / size);
                const x = bestMove % size;
                return { x, y };
            }
            
            return null;

        } catch (e) {
            console.error("Inference failed:", e);
            return null;
        }
    }
}
