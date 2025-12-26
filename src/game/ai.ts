import { PLAYER_CONFIG } from './types';
import type { GameState, Player } from './types';
import { getNeighbors, getGroup, placeStone } from './logic';
import _ from 'lodash';

// Helper: Check if a point is a true eye for the player
// A simple eye: Surrounded by own stones on N/S/E/W.
// A true eye also needs diagonals (should not be false eye).
function isTrueEye(board: (Player | null)[][], x: number, y: number, player: Player): boolean {
    const size = board.length;
    const neighbors = getNeighbors(x, y, size);
    // 1. Check orthogonal neighbors (must be all own stones or edge)
    // Note: getNeighbors only returns valid board points. We need to check if 'missing' neighbors are edges.
    // Actually, getNeighbors returns valid coords. If a neighbor is NOT in the list, it's off-board (edge).
    // So we just iterate what we have.
    for (const [nx, ny] of neighbors) {
        if (board[ny][nx] !== player) return false;
    }

    // 2. Check diagonals for false eye detection
    // Rule: To be a true eye, at least 3 of the 4 diagonals must be own stones or off-board.
    // (If on edge/corner, need all valid diagonals).
    const diagonals = [
        [x - 1, y - 1], [x + 1, y - 1],
        [x - 1, y + 1], [x + 1, y + 1]
    ];
    
    let enemyDiagonals = 0;
    let validDiagonals = 0;

    for (const [dx, dy] of diagonals) {
        if (dx < 0 || dx >= size || dy < 0 || dy >= size) {
            // Off-board counts as friendly for eye purposes usually (or neutral)
            // But standard rule: Corner eye needs 1 occupied diagonal? 
            // Simplified: Off-board diagonals are "controlled".
            continue; 
        }
        validDiagonals++;
        const stone = board[dy][dx];
        // If empty or enemy, it might break the eye
        if (stone !== player) {
             enemyDiagonals++;
        }
    }

    // False eye logic:
    // Center: Needs >= 3 friendly diagonals (so <= 1 enemy/empty)
    // Edge/Corner: Strict rules apply, but generally if too many enemies at diagonals, it's false.
    // Let's use a simplified heuristic: If > 1 enemy diagonal, it's a false eye.
    if (enemyDiagonals > 1) return false;

    return true;
}

// Helper: Evaluate Shape (3x3 local pattern)
function evaluateShape(board: (Player | null)[][], x: number, y: number, player: Player): number {
    let score = 0;
    const team = PLAYER_CONFIG[player].team;
    const size = board.length;

    // Relative coordinates
    const N = y > 0 ? board[y-1][x] : 'EDGE';
    const S = y < size-1 ? board[y+1][x] : 'EDGE';
    const W = x > 0 ? board[y][x-1] : 'EDGE';
    const E = x < size-1 ? board[y][x+1] : 'EDGE';

    const NW = (y > 0 && x > 0) ? board[y-1][x-1] : 'EDGE';
    const NE = (y > 0 && x < size-1) ? board[y-1][x+1] : 'EDGE';
    const SW = (y < size-1 && x > 0) ? board[y+1][x-1] : 'EDGE';
    const SE = (y < size-1 && x < size-1) ? board[y+1][x+1] : 'EDGE';

    const isAlly = (p: string | null) => p === player; // Strict ally (same color)
    const isEnemy = (p: string | null) => p && p !== 'EDGE' && PLAYER_CONFIG[p as Player].team !== team;

    // 1. Empty Triangle (Bad Shape) - Penalize
    // Shape: X + 2 adjacent allies form a triangle
    if (isAlly(N) && isAlly(W) && isAlly(NW)) score -= 30;
    if (isAlly(N) && isAlly(E) && isAlly(NE)) score -= 30;
    if (isAlly(S) && isAlly(W) && isAlly(SW)) score -= 30;
    if (isAlly(S) && isAlly(E) && isAlly(SE)) score -= 30;

    // 2. Tiger's Mouth (Good Shape) - Encourage forming or completing
    // Playing X creates a Tiger's mouth if we have (N, W) or (N, E) etc?
    // Actually, playing AT the 'mouth' connects.
    // Let's look for "Hane" (Bending around enemy)
    // Hane: Ally at W, Enemy at NW. Playing X (N) puts pressure.
    if (isAlly(W) && isEnemy(NW)) score += 15; 
    if (isAlly(E) && isEnemy(NE)) score += 15;
    if (isAlly(N) && isEnemy(NW)) score += 15;
    if (isAlly(S) && isEnemy(SW)) score += 15;

    // 3. Cutting Point (Cut!)
    // If we have Enemy at W and N, and NW is Empty/Enemy...
    // Cutting is usually good if supported.
    // Simple Cut: Enemy at W, Enemy at N. Playing X separates them? No, they are diagonal.
    // Cut: Enemy at NW, Enemy at SE. We play X. (Cross-cut?)
    
    // 4. Ponuki / Capture Shape (Already handled by capture logic)

    // 5. Bamboo Joint (Strong connection)
    // Ally at N, Ally at W, Ally at NW... wait that's triangle.
    // Bamboo: N=Ally, S=Ally, W=Ally, E=Ally? No.
    // Bamboo is || shape.

    return score;
}

// Improved Heuristic AI
export class SimpleAI {
    player: Player;
    team: 'AC' | 'BD';

    constructor(player: Player) {
        this.player = player;
        this.team = PLAYER_CONFIG[player].team;
    }

    // Evaluate a potential move
    // Returns a score (higher is better)
    evaluateMove(gameState: GameState, x: number, y: number): number {
        // 0. Eye Protection (Critical)
        if (isTrueEye(gameState.board, x, y, this.player)) {
            return -Infinity; // Do not fill your own eye!
        }

        const size = gameState.board.length;
        let score = Math.random() * 5; // Reduced randomness for more consistent play

        // 1. Position Heuristics (Static)
        // Distance from center
        const center = (size - 1) / 2;
        const distToCenter = Math.abs(x - center) + Math.abs(y - center);
        // Distance from edge
        const distToEdgeX = Math.min(x, size - 1 - x);
        const distToEdgeY = Math.min(y, size - 1 - y);
        const minEdgeDist = Math.min(distToEdgeX, distToEdgeY);

        // Opening Strategy (Turn < 40)
        // Scale turn count threshold by board size area relative to 19x19
        const scaleFactor = (size * size) / (19 * 19);
        if (gameState.turnCount < 40 * scaleFactor) {
            // Prefer 3rd and 4th lines (Corner/Side)
            if (minEdgeDist === 2 || minEdgeDist === 3) {
                score += 25; // High priority for corners/sides
            }
            // Avoid 1st and 2nd lines (too low) and Center (too slow) early on
            if (minEdgeDist < 2) score -= 40;
            if (distToCenter < size / 4) score -= 15; 
        } else {
            // Midgame: Center becomes important
            score += (10 - distToCenter); 
        }

        // Avoid edge of death (Line 1) unless urgent
        if (minEdgeDist === 0) score -= 60;

        // 2. Local Context (Neighbors)
        const neighbors = getNeighbors(x, y, size);
        let enemyContact = 0;
        let allyContact = 0;
        let teammateContact = 0; // Same team, different player

        for (const [nx, ny] of neighbors) {
            const stone = gameState.board[ny][nx];
            if (stone) {
                if (PLAYER_CONFIG[stone].team !== this.team) {
                    enemyContact++;
                } else {
                    if (stone === this.player) {
                        allyContact++;
                    } else {
                        teammateContact++;
                    }
                }
            }
        }

        // Fighting (Contact)
        if (enemyContact > 0) score += 20; // Fight enemies!
        
        // Shape: Connect with allies (Tiger's mouth etc.)
        if (allyContact > 0) score += 10; 

        // Weak Alliance Nuance:
        // Touching a teammate (e.g. A touching C) creates mixed walls.
        // Mixed walls risk creating invalid territory (0 points).
        // So, penalize touching teammate unless necessary for life.
        if (teammateContact > 0) score -= 15; 


        // 3. Simulation (Lightweight)
        const nextState = placeStone(gameState, x, y);
        if (!nextState) return -Infinity; // Invalid move (Suicide or Ko)

        // Self-Atari Check
        const group = getGroup(nextState.board, x, y);
        if (group) {
            // Liberties check
            if (group.liberties === 1) {
                // Self-atari is VERY bad unless it captures something
                const prisonersDiff = nextState.prisoners[this.player] - gameState.prisoners[this.player];
                if (prisonersDiff > 0) {
                    score += prisonersDiff * 60; // Capturing is good
                } else {
                    score -= 200; // Avoid self-atari at all costs!
                }
            } else if (group.liberties === 2) {
                score -= 20; // Be careful with 2 liberties
            } else if (group.liberties >= 4) {
                score += 10; // Strong shape
            }
        }

        // 4. Shape Analysis (New!)
        score += evaluateShape(gameState.board, x, y, this.player);

        // 5. Save Teammate Check (Altruism)
        // Check if this move increases liberties of a neighbor teammate group that is in Atari
        for (const [nx, ny] of neighbors) {
            const stone = gameState.board[ny][nx];
            if (stone && PLAYER_CONFIG[stone].team === this.team && stone !== this.player) {
                const teammateGroup = getGroup(gameState.board, nx, ny);
                if (teammateGroup && teammateGroup.liberties === 1) {
                    // Teammate is in danger!
                    // Does this move save them?
                    // We need to check if the new group (merged or adjacent) has > 1 liberty.
                    // Note: In this variant, different players DON'T merge into one group.
                    // So we just check if we killed an enemy adjacent to them, OR if we occupied a liberty they needed?
                    // Wait, if A places stone at C's liberty, A BLOCKS that liberty!
                    // So placing a stone NEXT to a teammate in Atari actually KILLS them if it was their last liberty!
                    
                    // BUT, if we capture an enemy stone that was surrounding the teammate, that saves them.
                    const prisonersDiff = nextState.prisoners[this.player] - gameState.prisoners[this.player];
                    if (prisonersDiff > 0) {
                         score += 100; // Saving teammate by capturing is HUGE
                    }
                }
            }
        }
        
        // 6. Save Self Check
        // If we are in Atari somewhere else, does this move help? 
        // (Simplified: We only check if we are extending our own group locally, which is covered by 'liberties' check above)

        return score;
    }

    getBestMove(gameState: GameState): { x: number, y: number } | null {
        const size = gameState.board.length;
        let bestScore = -Infinity;
        let bestMoves: { x: number, y: number }[] = [];

        // Optimization: Candidate pruning
        const candidates = new Set<string>();
        
        // Always check Star Points
        const starPoints: [number, number][] = [];
        if (size === 19) {
            starPoints.push(
                [3, 3], [3, 9], [3, 15],
                [9, 3], [9, 9], [9, 15],
                [15, 3], [15, 9], [15, 15]
            );
        } else if (size === 9) {
            starPoints.push(
                [2, 2], [2, 6],
                [4, 4],
                [6, 2], [6, 6]
            );
        } else if (size === 13) {
            starPoints.push(
                [3, 3], [3, 6], [3, 9],
                [6, 3], [6, 6], [6, 9],
                [9, 3], [9, 6], [9, 9]
            );
        }
        
        starPoints.forEach(([sx, sy]) => candidates.add(`${sx},${sy}`));

        // Check near existing stones (Range 2)
        let hasStones = false;
        for(let y=0; y<size; y++) {
            for(let x=0; x<size; x++) {
                if(gameState.board[y][x]) {
                    hasStones = true;
                    // Add neighbors (range 2)
                    for(let dy=-2; dy<=2; dy++) {
                        for(let dx=-2; dx<=2; dx++) {
                            const nx = x + dx;
                            const ny = y + dy;
                            if(nx >= 0 && nx < size && ny >= 0 && ny < size) {
                                candidates.add(`${nx},${ny}`);
                            }
                        }
                    }
                }
            }
        }

        // If board is empty (or near empty), add random points
        const scaleFactor = (size * size) / (19 * 19);
        if (!hasStones || gameState.turnCount < 8 * scaleFactor) {
            for(let i=0; i<15; i++) {
                const rx = Math.floor(Math.random() * size);
                const ry = Math.floor(Math.random() * size);
                candidates.add(`${rx},${ry}`);
            }
        }

        const candidateMoves = Array.from(candidates).map(s => {
            const [x, y] = s.split(',').map(Number);
            return { x, y };
        });

        // Evaluate candidates
        for (const move of candidateMoves) {
            if (gameState.board[move.y][move.x] !== null) continue; // Skip occupied

            const score = this.evaluateMove(gameState, move.x, move.y);
            
            if (score > bestScore) {
                bestScore = score;
                bestMoves = [move];
            } else if (Math.abs(score - bestScore) < 0.001) {
                bestMoves.push(move);
            }
        }

        // If no good moves found (or all negative), maybe pass? 
        // For this simple AI, we always try to play if score > -100 (not suicide)
        if (bestMoves.length > 0 && bestScore > -100) {
             return bestMoves[Math.floor(Math.random() * bestMoves.length)];
        }

        return null; // Pass
    }
}
