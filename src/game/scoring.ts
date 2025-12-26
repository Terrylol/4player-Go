import type { GameState, Player, Team } from './types';
import { getNeighbors } from './logic';

// Helper to get owner of a territory region
// Returns:
// - Player: if the region is exclusively surrounded by one player
// - 'NEUTRAL': if surrounded by multiple players (mixed territory) or empty
export const calculateTerritoryOwner = (
    board: (Player | null)[][],
    region: [number, number][]
): Player | 'NEUTRAL' => {
    const size = board.length;
    const boundaryStones = new Set<Player>();
    let touchedEdge = false; // In some rules, touching edge is fine.

    for (const [x, y] of region) {
        const neighbors = getNeighbors(x, y, size);
        // If neighbors count < 4, it means it touches the board edge.
        // In Go, board edge acts as a wall, so it doesn't invalidate territory.
        
        for (const [nx, ny] of neighbors) {
            const stone = board[ny][nx];
            if (stone !== null) {
                boundaryStones.add(stone);
            }
        }
    }

    if (boundaryStones.size === 1) {
        // Only one color surrounds this region
        return Array.from(boundaryStones)[0];
    }

    return 'NEUTRAL';
};

export const calculateScores = (state: GameState): Record<Team, number> => {
    const size = state.board.length;
    const visited = new Set<string>();
    const territoryCounts = { A: 0, B: 0, C: 0, D: 0 };
    
    // 1. Identify territories
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            if (state.board[y][x] === null && !visited.has(`${x},${y}`)) {
                // Start flood fill for this empty region
                const region: [number, number][] = [];
                const queue: [number, number][] = [[x, y]];
                visited.add(`${x},${y}`);

                while (queue.length > 0) {
                    const [cx, cy] = queue.shift()!;
                    region.push([cx, cy]);

                    const neighbors = getNeighbors(cx, cy, size);
                    for (const [nx, ny] of neighbors) {
                        if (state.board[ny][nx] === null && !visited.has(`${nx},${ny}`)) {
                            visited.add(`${nx},${ny}`);
                            queue.push([nx, ny]);
                        }
                    }
                }

                // Determine owner
                const owner = calculateTerritoryOwner(state.board, region);
                if (owner !== 'NEUTRAL') {
                    territoryCounts[owner] += region.length;
                }
            }
        }
    }

    // 2. Count Stones on Board
    const stoneCounts = { A: 0, B: 0, C: 0, D: 0 };
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const stone = state.board[y][x];
            if (stone) {
                stoneCounts[stone]++;
            }
        }
    }

    // 3. Sum up scores (Area Scoring: Stones + Territory)
    // Note: Prisoners are NOT counted in Area Scoring.
    // Team AC
    const scoreA = territoryCounts.A + stoneCounts.A;
    const scoreC = territoryCounts.C + stoneCounts.C;
    
    // Team BD
    const scoreB = territoryCounts.B + stoneCounts.B;
    const scoreD = territoryCounts.D + stoneCounts.D;

    return {
        AC: scoreA + scoreC,
        BD: scoreB + scoreD
    };
};
