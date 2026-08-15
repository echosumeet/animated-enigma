import React from 'react';
import { Card, GamePhase, Player } from '../types/game';
import { CardView } from './CardView';
import { Bot, User, Award, ShieldAlert, Zap, TrendingUp, DollarSign } from 'lucide-react';

interface PokerTableProps {
  players: Player[];
  communityCards: Card[];
  mainPot: number;
  currentTurnIndex: number;
  dealerIndex: number;
  phase: GamePhase;
  tableName?: string;
  tableTheme?: 'emerald' | 'sapphire' | 'amber' | 'ruby';
  handNumber?: number;
  tournamentLevel?: number;
  synergyMultiplierPct?: number;
  compact?: boolean;
  turnTimeRemaining?: number;
  totalTurnTime?: number;
}

export const PokerTable: React.FC<PokerTableProps> = ({
  players,
  communityCards,
  mainPot,
  currentTurnIndex,
  dealerIndex,
  phase,
  tableName = 'Tournament Arena',
  tableTheme = 'emerald',
  handNumber,
  tournamentLevel,
  synergyMultiplierPct,
  compact = false,
  turnTimeRemaining = 15,
  totalTurnTime = 15,
}) => {
  // 5 Seat positions around the oval table
  const seatLayouts = [
    { top: '78%', left: '50%', transform: 'translate(-50%, -50%)' }, // Seat 1 (User)
    { top: '54%', left: '12%', transform: 'translate(-50%, -50%)' }, // Seat 2 (Quant)
    { top: '18%', left: '28%', transform: 'translate(-50%, -50%)' }, // Seat 3 (Trader)
    { top: '18%', left: '72%', transform: 'translate(-50%, -50%)' }, // Seat 4 (Shark)
    { top: '54%', left: '88%', transform: 'translate(-50%, -50%)' }, // Seat 5 (Degen)
  ];

  const isUserTurnOnThisTable =
    currentTurnIndex === 0 &&
    phase.endsWith('_BETTING') &&
    players[0]?.status === 'active';

  const isUrgent = isUserTurnOnThisTable && turnTimeRemaining <= 5;

  const getFeltGradient = () => {
    switch (tableTheme) {
      case 'sapphire':
        return 'from-blue-950 via-indigo-950 to-slate-950 border-blue-500/30';
      case 'amber':
        return 'from-amber-950 via-stone-950 to-slate-950 border-amber-500/30';
      case 'ruby':
        return 'from-rose-950 via-red-950 to-slate-950 border-rose-500/30';
      case 'emerald':
      default:
        return 'from-emerald-950 via-teal-950 to-slate-950 border-emerald-500/20';
    }
  };

  const getWatermarkColor = () => {
    switch (tableTheme) {
      case 'sapphire':
        return 'text-blue-500/5';
      case 'amber':
        return 'text-amber-500/5';
      case 'ruby':
        return 'text-rose-500/5';
      case 'emerald':
      default:
        return 'text-emerald-500/5';
    }
  };

  const getPersonalityIcon = (p?: string) => {
    switch (p) {
      case 'quant':
        return <Award className="w-3 h-3 text-cyan-400" />;
      case 'trader':
        return <TrendingUp className="w-3 h-3 text-emerald-400" />;
      case 'poker_shark':
        return <ShieldAlert className="w-3 h-3 text-amber-400" />;
      case 'degen':
        return <Zap className="w-3 h-3 text-rose-400" />;
      default:
        return <Bot className="w-3 h-3 text-indigo-400" />;
    }
  };

  return (
    <div className={`relative w-full aspect-[16/10] sm:aspect-[16/9] ${compact ? 'max-w-3xl' : 'max-w-4xl'} mx-auto rounded-3xl bg-slate-950 p-2 sm:p-5 flex items-center justify-center select-none shadow-2xl border border-slate-800/80 overflow-hidden`}>
      {/* Table Outer Rail */}
      <div className="relative w-full h-full rounded-[34px] sm:rounded-[60px] bg-gradient-to-b from-stone-800 via-stone-900 to-black p-2 sm:p-4 shadow-[inset_0_4px_12px_rgba(0,0,0,0.8)] border border-stone-700/50 flex items-center justify-center">
        {/* Table Inner Felt */}
        <div className={`relative w-full h-full rounded-[26px] sm:rounded-[45px] bg-gradient-to-b ${getFeltGradient()} shadow-[inset_0_0_80px_rgba(0,0,0,0.9)] border-2 flex flex-col items-center justify-center overflow-hidden`}>
          {/* Subtle Felt Texture & Grid */}
          <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:12px_12px]" />

          {/* Table Header Overlay Pill */}
          <div className="absolute top-2.5 left-4 z-10 flex items-center gap-2 pointer-events-none">
            <span className="text-[10px] font-mono uppercase font-bold tracking-wider px-2.5 py-0.5 rounded-full bg-slate-950/80 border border-slate-800 text-slate-300 backdrop-blur">
              {tableName}
            </span>
            {handNumber !== undefined && (
              <span className="text-[10px] font-mono text-amber-400/90 font-bold">
                Hand #{handNumber}
              </span>
            )}
            {tournamentLevel !== undefined && (
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                Lvl {tournamentLevel} (+{synergyMultiplierPct}%)
              </span>
            )}
          </div>

          {/* User Turn High-Alert Live Indicator Banner */}
          {isUserTurnOnThisTable && (
            <div className={`absolute top-2.5 right-4 z-20 flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold backdrop-blur shadow-lg animate-pulse ${
              isUrgent
                ? 'bg-rose-950/90 border-rose-500 text-rose-300 ring-2 ring-rose-500/50'
                : 'bg-amber-950/90 border-amber-400 text-amber-300 ring-2 ring-amber-400/40'
            }`}>
              <span className={`w-2 h-2 rounded-full ${isUrgent ? 'bg-rose-500 animate-ping' : 'bg-amber-400 animate-ping'}`} />
              <span>YOUR TURN: <strong className="font-mono text-white font-extrabold">{turnTimeRemaining}s</strong></span>
            </div>
          )}

          {/* Table Center Watermark */}
          <div className={`absolute font-extrabold text-2xl sm:text-4xl tracking-widest ${getWatermarkColor()} pointer-events-none uppercase`}>
            {tableName}
          </div>

          {/* Center Community Cards Board & Pot */}
          <div className="relative z-10 flex flex-col items-center gap-2 -mt-4 sm:-mt-6">
            {/* Main Pot */}
            <div className="flex items-center gap-2 bg-slate-950/85 border border-emerald-500/30 px-3 py-1 rounded-full shadow-lg backdrop-blur">
              <div className="w-4 h-4 rounded-full bg-amber-500 flex items-center justify-center text-[10px] font-bold text-black shadow-sm">
                $
              </div>
              <div className="text-xs sm:text-sm font-mono font-extrabold text-emerald-400">
                POT: <span className="text-white">{mainPot.toLocaleString()}</span>
              </div>
            </div>

            {/* Community Cards Display (5 slots) */}
            <div className="flex items-center gap-1 sm:gap-2 bg-slate-950/70 p-1.5 sm:p-2.5 rounded-xl border border-slate-800/80 shadow-inner">
              {[0, 1, 2, 3, 4].map((idx) => {
                const card = communityCards[idx];
                return (
                  <div key={idx} className="relative">
                    {card ? (
                      <CardView card={card} size={compact ? 'sm' : 'md'} />
                    ) : (
                      <div className={`${compact ? 'w-9 h-14' : 'w-11 sm:w-12 h-16 sm:h-18'} rounded-md border border-emerald-500/15 bg-emerald-950/20 flex items-center justify-center text-[9px] sm:text-[10px] font-mono text-emerald-600/40`}>
                        {idx < 3 ? 'FLOP' : idx === 3 ? 'TURN' : 'RIVER'}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* 5 Player Seats */}
          {players.map((player, idx) => {
            const isCurrentTurn = idx === currentTurnIndex && phase.endsWith('_BETTING');
            const isDealer = idx === dealerIndex;
            const isFolded = player.status === 'folded';
            const layout = seatLayouts[idx];

            return (
              <div
                key={player.id}
                style={{
                  position: 'absolute',
                  top: layout.top,
                  left: layout.left,
                  transform: layout.transform,
                }}
                className={`z-20 flex flex-col items-center transition-all ${
                  isFolded ? 'opacity-40 grayscale' : 'opacity-100'
                }`}
              >
                {/* Current Bet Floating Chips & Turn Timer */}
                <div className="mb-1 flex items-center gap-1.5">
                  {player.currentBet > 0 && (
                    <div className="bg-amber-500/90 text-slate-950 font-mono font-bold text-[10px] sm:text-xs px-2 py-0.5 rounded-full shadow-md flex items-center gap-1 border border-amber-300 animate-bounce">
                      <DollarSign className="w-2.5 h-2.5" />
                      {player.currentBet}
                    </div>
                  )}

                  {isCurrentTurn && (
                    <div className={`px-2 py-0.5 rounded-full font-mono font-extrabold text-[10px] flex items-center gap-1 border shadow-md ${
                      idx === 0
                        ? turnTimeRemaining <= 5
                          ? 'bg-rose-500 text-white border-rose-300 animate-pulse'
                          : 'bg-amber-500 text-slate-950 border-amber-300'
                        : 'bg-indigo-600 text-white border-indigo-400'
                    }`}>
                      <span className="w-1.5 h-1.5 rounded-full bg-white animate-ping" />
                      <span>{turnTimeRemaining}s</span>
                    </div>
                  )}
                </div>

                {/* Player Card Frame */}
                <div
                  className={`relative p-2 rounded-2xl flex flex-col items-center bg-slate-950/90 backdrop-blur border ${
                    isCurrentTurn
                      ? idx === 0
                        ? 'border-amber-400 ring-4 ring-amber-400/50 shadow-[0_0_25px_rgba(245,158,11,0.5)]'
                        : 'border-indigo-400 ring-2 ring-indigo-400/40 shadow-xl'
                      : 'border-slate-800 shadow-xl'
                  }`}
                >
                  {/* Dealer Button Badge */}
                  {isDealer && (
                    <div className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-amber-400 text-slate-950 font-black text-[10px] flex items-center justify-center border-2 border-white shadow-md">
                      D
                    </div>
                  )}

                  {/* Hole Cards */}
                  <div className="flex gap-1 mb-1.5">
                    {player.cards && player.cards.length > 0 ? (
                      <>
                        <CardView
                          card={player.cards[0]}
                          hidden={!player.showCards}
                          size="sm"
                        />
                        <CardView
                          card={player.cards[1]}
                          hidden={!player.showCards}
                          size="sm"
                        />
                      </>
                    ) : (
                      <>
                        <CardView hidden size="sm" />
                        <CardView hidden size="sm" />
                      </>
                    )}
                  </div>

                  {/* Player Name & Info */}
                  <div className="flex flex-col items-center text-center">
                    <div className="flex items-center gap-1">
                      {player.isBot ? (
                        getPersonalityIcon(player.botPersonality)
                      ) : (
                        <User className="w-3 h-3 text-indigo-400" />
                      )}
                      <span className="text-[11px] sm:text-xs font-bold text-white max-w-[90px] truncate">
                        {player.name}
                      </span>
                    </div>

                    {/* Cash Balance */}
                    <div className="font-mono font-semibold text-[10px] sm:text-[11px] text-emerald-400">
                      ${player.cashBalance.toLocaleString()}
                    </div>

                    {/* Last Action Pill */}
                    {player.lastAction && (
                      <span className="mt-0.5 text-[9px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                        {player.lastAction}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
