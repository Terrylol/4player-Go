export type Player = 'A' | 'B' | 'C' | 'D';
export type Team = 'AC' | 'BD';

export type StoneColor = 'black' | 'white' | 'blue' | 'red';

export interface Cell {
  x: number;
  y: number;
  stone: Player | null;
}

export interface GameState {
  board: (Player | null)[][];
  currentPlayer: Player;
  turnCount: number;
  prisoners: Record<Player, number>; // How many stones each player has captured
  history: (Player | null)[][][]; // For undo/ko check (simplified)
  consecutivePasses: number;
  scores: Record<Team, number>;
  gameOver: boolean;
  boardSize: number;
}

export const DEFAULT_BOARD_SIZE = 19;
export const BOARD_SIZE = 19; // Deprecated, but kept for compatibility during refactor if needed, though we should try to remove usage.

export const PLAYER_CONFIG: Record<Player, { color: StoneColor; team: Team; name: string }> = {
  'A': { color: 'black', team: 'AC', name: 'Player A' },
  'B': { color: 'white', team: 'BD', name: 'Player B' },
  'C': { color: 'blue', team: 'AC', name: 'Player C' },
  'D': { color: 'red', team: 'BD', name: 'Player D' },
};

export const TURN_ORDER: Player[] = ['A', 'B', 'C', 'D'];
