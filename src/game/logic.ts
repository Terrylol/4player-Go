import { BOARD_SIZE, TURN_ORDER } from './types';
import type { GameState, Player } from './types';
import { PLAYER_CONFIG } from './types';
import _ from 'lodash';

// Initialize empty board
export const createInitialState = (size: number = 19): GameState => {
  const board = Array(size).fill(null).map(() => Array(size).fill(null));
  return {
    board,
    currentPlayer: 'A',
    turnCount: 0,
    prisoners: { A: 0, B: 0, C: 0, D: 0 },
    history: [],
    consecutivePasses: 0,
    scores: { AC: 0, BD: 0 },
    gameOver: false,
    boardSize: size,
  };
};

// Check if a move is within bounds
export const isValidCoordinate = (x: number, y: number, size: number): boolean => {
  return x >= 0 && x < size && y >= 0 && y < size;
};

// Get neighbors (up, down, left, right)
export const getNeighbors = (x: number, y: number, size: number): [number, number][] => {
  const diffs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
  return diffs
    .map(([dx, dy]) => [x + dx, y + dy] as [number, number])
    .filter(([nx, ny]) => isValidCoordinate(nx, ny, size));
};

// Calculate liberties for a group of stones
// IMPORTANT: In this variant, different players (even teammates) DO NOT share liberties.
// A group is defined strictly by SAME PLAYER stones connected orthogonally.
export const getGroup = (board: (Player | null)[][], x: number, y: number): { stones: [number, number][], liberties: number } | null => {
  const player = board[y][x];
  if (!player) return null;

  const size = board.length;
  const visited = new Set<string>();
  const stones: [number, number][] = [];
  const liberties = new Set<string>();
  const queue: [number, number][] = [[x, y]];
  visited.add(`${x},${y}`);

  while (queue.length > 0) {
    const [cx, cy] = queue.shift()!;
    stones.push([cx, cy]);

    const neighbors = getNeighbors(cx, cy, size);
    for (const [nx, ny] of neighbors) {
      const neighborStone = board[ny][nx];
      if (neighborStone === null) {
        liberties.add(`${nx},${ny}`);
      } else if (neighborStone === player) {
        if (!visited.has(`${nx},${ny}`)) {
          visited.add(`${nx},${ny}`);
          queue.push([nx, ny]);
        }
      }
      // If neighbor is enemy or teammate (different player), it blocks the liberty.
    }
  }

  return { stones, liberties: liberties.size };
};

// Attempt to place a stone
// Returns new state if valid, or null if invalid
export const placeStone = (state: GameState, x: number, y: number): GameState | null => {
  if (state.board[y][x] !== null) return null; // Already occupied

  const size = state.board.length;
  const newBoard = _.cloneDeep(state.board);
  const currentPlayer = state.currentPlayer;
  newBoard[y][x] = currentPlayer;

  // 1. Check for captures
  // We check all neighbors. If a neighbor belongs to another player (enemy OR teammate) and has 0 liberties, it is captured.
  let capturedStones: [number, number][] = [];
  const neighbors = getNeighbors(x, y, size);
  
  // Use a set to avoid checking same group multiple times
  const checkedGroups = new Set<string>();

  for (const [nx, ny] of neighbors) {
    const neighborPlayer = newBoard[ny][nx];
    if (neighborPlayer && neighborPlayer !== currentPlayer) {
       // Check if this group is captured
       // We use a representative point to identify groups to avoid double counting
       if (!checkedGroups.has(`${nx},${ny}`)) {
         const group = getGroup(newBoard, nx, ny);
         if (group && group.liberties === 0) {
           capturedStones = [...capturedStones, ...group.stones];
           group.stones.forEach(([gx, gy]) => checkedGroups.add(`${gx},${gy}`));
         }
       }
    }
  }

  // Remove captured stones
  capturedStones.forEach(([cx, cy]) => {
    newBoard[cy][cx] = null;
  });

  // 2. Check for suicide
  // After captures, does the placed stone have liberties?
  const selfGroup = getGroup(newBoard, x, y);
  if (selfGroup && selfGroup.liberties === 0) {
    // Suicide move is generally forbidden in Go, unless it captures something.
    // If we captured something, we definitely have liberties (the empty spots left by captured stones).
    // But wait, if we captured stones, the 'newBoard' already has them removed, so getGroup should find liberties.
    // So if liberties are STILL 0, it's a suicide.
    return null; 
  }

  // 3. Ko rule (Simple Ko: cannot return to exact previous state)
  // For 4 player, strict Ko might be too complex, but let's prevent immediate board repetition.
  if (state.history.length > 0) {
      const lastBoard = state.history[state.history.length - 1];
      if (_.isEqual(newBoard, lastBoard)) {
          return null; // Ko violation
      }
  }

  // Update State
  const newPrisoners = { ...state.prisoners };
  
  // Scoring Logic for Captures:
  // - Capturing Enemies (Different Team): +1 point per stone
  // - Capturing Teammates (Same Team): 0 points (prevent farming teammates)
  const currentTeam = PLAYER_CONFIG[currentPlayer].team;

  capturedStones.forEach(([cx, cy]) => {
      // The stone at cx, cy was already removed from newBoard, 
      // so we need to look at the OLD board (state.board) to see who owned it.
      const capturedOwner = state.board[cy][cx];
      if (capturedOwner) {
          const capturedTeam = PLAYER_CONFIG[capturedOwner].team;
          if (capturedTeam !== currentTeam) {
              newPrisoners[currentPlayer] += 1;
          } else {
              // Friendly fire! No points awarded.
              // Optional: We could track 'friendly_kills' for stats.
          }
      }
  });

  const nextPlayerIndex = (TURN_ORDER.indexOf(currentPlayer) + 1) % 4;
  
  return {
    ...state,
    board: newBoard,
    currentPlayer: TURN_ORDER[nextPlayerIndex],
    turnCount: state.turnCount + 1,
    prisoners: newPrisoners,
    consecutivePasses: 0,
    history: [...state.history, state.board], // Store old board
  };
};

export const passTurn = (state: GameState): GameState => {
    const nextPlayerIndex = (TURN_ORDER.indexOf(state.currentPlayer) + 1) % 4;
    const newConsecutivePasses = state.consecutivePasses + 1;
    const isGameOver = newConsecutivePasses >= 4;

    return {
        ...state,
        currentPlayer: TURN_ORDER[nextPlayerIndex],
        turnCount: state.turnCount + 1,
        consecutivePasses: newConsecutivePasses,
        gameOver: isGameOver,
        history: [...state.history, state.board]
    };
}
