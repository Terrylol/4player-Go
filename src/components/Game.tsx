import React, { useState, useEffect } from 'react';
import Board from './Board';
import { createInitialState, passTurn, placeStone } from '../game/logic';
import { calculateScores } from '../game/scoring';
import { PLAYER_CONFIG } from '../game/types';
import type { GameState, Team, Player } from '../game/types';
import { SimpleAI } from '../game/ai';
import { NeuralAI } from '../game/NeuralAI';

type GameMode = 'PvP' | 'PvE' | 'EvE'; // Player vs Player, Player vs AI, AI vs AI

const Game = () => {
  const [boardSize, setBoardSize] = useState<number>(19);
  const [gameState, setGameState] = useState<GameState>(createInitialState(19));
  const [scores, setScores] = useState<Record<Team, number>>({ AC: 0, BD: 0 });
  const [gameMode, setGameMode] = useState<GameMode>('PvP');
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);

  // Update scores whenever board changes
  useEffect(() => {
      const newScores = calculateScores(gameState);
      setScores(newScores);
  }, [gameState.board, gameState.prisoners]);

  // AI Logic
  useEffect(() => {
    if (gameState.gameOver) {
        setIsAutoPlaying(false);
        return;
    }

    const currentTeam = PLAYER_CONFIG[gameState.currentPlayer].team;
    let shouldAIPlay = false;

    if (gameMode === 'EvE' && isAutoPlaying) {
        shouldAIPlay = true;
    } else if (gameMode === 'PvE') {
        // In PvE, let's say Human is Team AC (A, C) and AI is Team BD (B, D)
        // Or strictly Human is A, everyone else is AI?
        // Let's implement: Human = Team AC, AI = Team BD.
        if (currentTeam === 'BD') {
            shouldAIPlay = true;
        }
    }

    if (shouldAIPlay) {
        const timer = setTimeout(async () => {
            let move = null;
            
            // Try Neural AI first
            try {
                const neuralAI = new NeuralAI(gameState.currentPlayer);
                move = await neuralAI.getBestMove(gameState);
            } catch (e) {
                console.warn("Neural AI failed, falling back to SimpleAI", e);
            }

            // Fallback to SimpleAI
            if (!move) {
                const simpleAI = new SimpleAI(gameState.currentPlayer);
                move = simpleAI.getBestMove(gameState);
            }
            
            if (move) {
                const newState = placeStone(gameState, move.x, move.y);
                if (newState) {
                    setGameState(newState);
                } else {
                    setGameState(passTurn(gameState));
                }
            } else {
                setGameState(passTurn(gameState));
            }
        }, 500); // 500ms delay for better UX
        return () => clearTimeout(timer);
    }
  }, [gameState, gameMode, isAutoPlaying]);

  const handlePass = () => {
    setGameState(passTurn(gameState));
  };

  const handleReset = () => {
    if (confirm("Restart game?")) {
        setGameState(createInitialState(boardSize));
        setIsAutoPlaying(false);
    }
  };

  const handleSizeChange = (size: number) => {
      if (confirm(`Switch to ${size}x${size} board? Current game progress will be lost.`)) {
          setBoardSize(size);
          setGameState(createInitialState(size));
          setIsAutoPlaying(false);
      }
  };

  const currentPlayer = PLAYER_CONFIG[gameState.currentPlayer];
  
  // Determine if human can play
  const isHumanTurn = (() => {
      if (gameState.gameOver) return false;
      if (gameMode === 'PvP') return true;
      if (gameMode === 'EvE') return false; // AI vs AI, human watches
      if (gameMode === 'PvE') {
          // Human is AC
          return PLAYER_CONFIG[gameState.currentPlayer].team === 'AC';
      }
      return false;
  })();

  const winningTeam = scores.AC > scores.BD ? 'AC' : scores.BD > scores.AC ? 'BD' : 'Draw';

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 gap-6 bg-[#f0e6d2]">
      {/* Game Over Overlay */}
      {gameState.gameOver && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="bg-white p-8 rounded-xl shadow-2xl max-w-md w-full text-center animate-bounce-in">
                  <h2 className="text-4xl font-bold mb-4 text-gray-800">Game Over!</h2>
                  <div className="text-2xl mb-6">
                      Winner: <span className={`font-bold ${winningTeam === 'AC' ? 'text-blue-600' : winningTeam === 'BD' ? 'text-red-600' : 'text-gray-600'}`}>
                          {winningTeam === 'Draw' ? 'It\'s a Draw!' : `Team ${winningTeam}`}
                      </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mb-8">
                      <div className="bg-blue-50 p-4 rounded-lg">
                          <div className="text-blue-800 font-bold">Team AC</div>
                          <div className="text-3xl font-mono">{scores.AC}</div>
                      </div>
                      <div className="bg-red-50 p-4 rounded-lg">
                          <div className="text-red-800 font-bold">Team BD</div>
                          <div className="text-3xl font-mono">{scores.BD}</div>
                      </div>
                  </div>
                  <button 
                    onClick={handleReset}
                    className="px-8 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-bold shadow-lg transition-transform hover:scale-105"
                  >
                      Play Again
                  </button>
              </div>
          </div>
      )}

      <h1 className="text-4xl font-bold text-gray-800 font-serif">4-Player Weak-Alliance Go</h1>
      
      {/* Settings Panel */}
      <div className="flex flex-wrap gap-4 justify-center">
          {/* Board Size Selection */}
          <div className="flex gap-2 p-2 bg-white/50 rounded-lg shadow-sm items-center">
              <span className="font-bold text-gray-700 mr-2">Board Size:</span>
              {[9, 13, 19].map(size => (
                  <button
                    key={size}
                    onClick={() => {
                        if (boardSize !== size) handleSizeChange(size);
                    }}
                    className={`px-3 py-1 rounded font-bold transition-all ${boardSize === size ? 'bg-indigo-600 text-white shadow-md' : 'bg-white text-gray-600 hover:bg-gray-100'}`}
                  >
                      {size}x{size}
                  </button>
              ))}
          </div>

          {/* Game Mode Selection */}
          <div className="flex gap-4 p-2 bg-white/50 rounded-lg shadow-sm items-center">
              <span className="font-bold text-gray-700 mr-2">Mode:</span>
              <label className="flex items-center gap-2 cursor-pointer">
                  <input 
                    type="radio" 
                    name="mode" 
                    checked={gameMode === 'PvP'} 
                    onChange={() => setGameMode('PvP')}
                  />
                  <span className="font-bold">Human vs Human</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                  <input 
                    type="radio" 
                    name="mode" 
                    checked={gameMode === 'PvE'} 
                    onChange={() => setGameMode('PvE')}
                  />
                  <span className="font-bold">PvE (AC vs BD)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                  <input 
                    type="radio" 
                    name="mode" 
                    checked={gameMode === 'EvE'} 
                    onChange={() => {
                        setGameMode('EvE');
                        setIsAutoPlaying(false); // Stop pending auto play if any
                    }}
                  />
                  <span className="font-bold">AI vs AI</span>
              </label>
          </div>
      </div>

      <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl justify-center items-start">
        {/* Left Panel: Team AC */}
        <div className="flex flex-col gap-4 p-6 bg-white rounded-xl shadow-lg w-64 border-l-8 border-blue-500">
            <h2 className="text-2xl font-bold text-blue-800 border-b pb-2">Team AC</h2>
            <div className="text-5xl font-mono font-bold text-center text-gray-800">{scores.AC}</div>
            <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between"><span>Prisoners (A):</span> <span className="font-bold">{gameState.prisoners.A}</span></div>
                <div className="flex justify-between"><span>Prisoners (C):</span> <span className="font-bold">{gameState.prisoners.C}</span></div>
            </div>
            
            <div className="mt-4 space-y-2">
                <div className={`flex items-center gap-2 p-2 rounded transition-all ${gameState.currentPlayer === 'A' ? 'bg-black text-white shadow-md scale-105' : 'text-gray-400'}`}>
                    <div className="w-4 h-4 rounded-full bg-black border border-gray-600"></div>
                    <span className="font-bold">Player A</span>
                </div>
                <div className={`flex items-center gap-2 p-2 rounded transition-all ${gameState.currentPlayer === 'C' ? 'bg-blue-600 text-white shadow-md scale-105' : 'text-gray-400'}`}>
                    <div className="w-4 h-4 rounded-full bg-blue-600 border border-blue-800"></div>
                    <span className="font-bold">Player C</span>
                </div>
            </div>
        </div>

        {/* Center: Board */}
        <div className="flex flex-col items-center gap-4">
            <Board gameState={gameState} onMove={setGameState} isHumanTurn={isHumanTurn} />
            
            {/* Status Bar */}
            <div className="flex items-center gap-4 px-6 py-3 bg-white rounded-full shadow-lg">
                <span className="text-gray-500 font-bold uppercase tracking-wider text-xs">Current Turn</span>
                <div className="flex items-center gap-2">
                    <div 
                        className="w-6 h-6 rounded-full shadow-inner"
                        style={{ 
                            background: currentPlayer.color === 'black' ? 'black' : 
                                       currentPlayer.color === 'white' ? 'white' :
                                       currentPlayer.color === 'blue' ? '#2563eb' : '#dc2626',
                            border: currentPlayer.color === 'white' ? '1px solid #ccc' : 'none'
                        }}
                    />
                    <span className="text-xl font-bold text-gray-800">{currentPlayer.name}</span>
                </div>
            </div>
        </div>

        {/* Right Panel: Team BD */}
        <div className="flex flex-col gap-4 p-6 bg-white rounded-xl shadow-lg w-64 border-r-8 border-red-500">
            <h2 className="text-2xl font-bold text-red-800 border-b pb-2">Team BD</h2>
            <div className="text-5xl font-mono font-bold text-center text-gray-800">{scores.BD}</div>
            <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between"><span>Prisoners (B):</span> <span className="font-bold">{gameState.prisoners.B}</span></div>
                <div className="flex justify-between"><span>Prisoners (D):</span> <span className="font-bold">{gameState.prisoners.D}</span></div>
            </div>

            <div className="mt-4 space-y-2">
                <div className={`flex items-center gap-2 p-2 rounded transition-all ${gameState.currentPlayer === 'B' ? 'bg-white border border-gray-300 shadow-md scale-105' : 'text-gray-400'}`}>
                    <div className="w-4 h-4 rounded-full bg-white border border-gray-300"></div>
                    <span className="font-bold">Player B</span>
                </div>
                <div className={`flex items-center gap-2 p-2 rounded transition-all ${gameState.currentPlayer === 'D' ? 'bg-red-600 text-white shadow-md scale-105' : 'text-gray-400'}`}>
                    <div className="w-4 h-4 rounded-full bg-red-600 border border-red-800"></div>
                    <span className="font-bold">Player D</span>
                </div>
            </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-4 mt-4">
          <button 
            onClick={handlePass}
            disabled={!isHumanTurn}
            className="px-8 py-3 bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg shadow-md font-bold transition-all transform hover:-translate-y-1"
          >
              Pass Turn
          </button>
          
          {gameMode === 'EvE' && (
              <button 
                onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                className={`px-8 py-3 rounded-lg shadow-md font-bold text-white transition-all transform hover:-translate-y-1 ${isAutoPlaying ? 'bg-red-500 hover:bg-red-600' : 'bg-green-600 hover:bg-green-700'}`}
              >
                  {isAutoPlaying ? 'Pause Auto-Play' : 'Start Auto-Play'}
              </button>
          )}

          <button 
            onClick={handleReset}
            className="px-8 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg shadow-md font-bold transition-all transform hover:-translate-y-1"
          >
              Restart Game
          </button>
      </div>
      
      <div className="text-gray-500 text-sm max-w-2xl text-center mt-8">
          <p>Rules: 4 Players, Weak Alliance. AC vs BD. Friendly fire allowed.</p>
          <p>Mixed territory (surrounded by both A and C) scores 0. Only pure single-color territory counts.</p>
      </div>
    </div>
  );
};

export default Game;
