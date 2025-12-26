import React, { useState } from 'react';
import { PLAYER_CONFIG } from '../game/types';
import type { GameState } from '../game/types';
import { placeStone } from '../game/logic';
import clsx from 'clsx';

interface BoardProps {
  gameState: GameState;
  onMove: (newState: GameState) => void;
  isHumanTurn: boolean;
}

const Board: React.FC<BoardProps> = ({ gameState, onMove, isHumanTurn }) => {
  const [hoverPos, setHoverPos] = useState<[number, number] | null>(null);
  const size = gameState.board.length;

  const handleCellClick = (x: number, y: number) => {
    if (!isHumanTurn) return;
    const newState = placeStone(gameState, x, y);
    if (newState) {
      onMove(newState);
    }
  };

  const currentPlayerConfig = PLAYER_CONFIG[gameState.currentPlayer];
  
  const isStarPoint = (x: number, y: number) => {
    if (size === 19) {
      const points = [
        [3, 3], [3, 9], [3, 15],
        [9, 3], [9, 9], [9, 15],
        [15, 3], [15, 9], [15, 15]
      ];
      return points.some(([sx, sy]) => sx === x && sy === y);
    }
    if (size === 9) {
      const points = [
        [2, 2], [2, 6],
        [4, 4],
        [6, 2], [6, 6]
      ];
      return points.some(([sx, sy]) => sx === x && sy === y);
    }
    if (size === 13) {
      const points = [
        [3, 3], [3, 6], [3, 9],
        [6, 3], [6, 6], [6, 9],
        [9, 3], [9, 6], [9, 9]
      ];
      return points.some(([sx, sy]) => sx === x && sy === y);
    }
    return false;
  };

  return (
    <div 
      className="relative bg-[#E3C07F] p-4 shadow-2xl rounded-sm mx-auto select-none"
      style={{
        width: 'min(90vw, 600px)',
        height: 'min(90vw, 600px)',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2), inset 0 0 20px rgba(0,0,0,0.1)'
      }}
    >
      {/* Board Grid Container */}
      <div 
        className="w-full h-full grid"
        style={{
          gridTemplateColumns: `repeat(${size}, 1fr)`,
          gridTemplateRows: `repeat(${size}, 1fr)`,
        }}
      >
        {gameState.board.map((row, y) => (
          row.map((cell, x) => (
            <div
              key={`${x}-${y}`}
              className="relative flex items-center justify-center"
              onClick={() => handleCellClick(x, y)}
              onMouseEnter={() => setHoverPos([x, y])}
              onMouseLeave={() => setHoverPos(null)}
              style={{ cursor: isHumanTurn ? 'pointer' : 'default' }}
            >
              {/* --- Grid Lines --- */}
              {/* Horizontal Line */}
              <div 
                className="absolute bg-black pointer-events-none"
                style={{
                  height: '1px',
                  top: '50%',
                  left: x === 0 ? '50%' : '0',
                  right: x === size - 1 ? '50%' : '0',
                }}
              />
              {/* Vertical Line */}
              <div 
                className="absolute bg-black pointer-events-none"
                style={{
                  width: '1px',
                  left: '50%',
                  top: y === 0 ? '50%' : '0',
                  bottom: y === size - 1 ? '50%' : '0',
                }}
              />

              {/* --- Star Point --- */}
              {isStarPoint(x, y) && (
                <div className="absolute w-2 h-2 bg-black rounded-full pointer-events-none" />
              )}

              {/* --- Stone --- */}
              {cell && (
                <div 
                  className={clsx(
                    "rounded-full w-[90%] h-[90%] z-10 transition-transform duration-150",
                    "shadow-[2px_2px_4px_rgba(0,0,0,0.4)]"
                  )}
                  style={{
                    background: (() => {
                      const color = PLAYER_CONFIG[cell].color;
                      if (color === 'black') return 'radial-gradient(circle at 30% 30%, #666, #000)';
                      if (color === 'white') return 'radial-gradient(circle at 30% 30%, #fff, #ddd)';
                      if (color === 'blue') return 'radial-gradient(circle at 30% 30%, #60a5fa, #1d4ed8)';
                      if (color === 'red') return 'radial-gradient(circle at 30% 30%, #f87171, #b91c1c)';
                      return 'gray';
                    })()
                  }}
                />
              )}

              {/* --- Ghost Stone (Hover) --- */}
              {!cell && isHumanTurn && hoverPos && hoverPos[0] === x && hoverPos[1] === y && (
                <div 
                  className="rounded-full w-[80%] h-[80%] z-10 opacity-50"
                  style={{
                    background: currentPlayerConfig.color
                  }}
                />
              )}
            </div>
          ))
        ))}
      </div>
    </div>
  );
};

export default Board;
