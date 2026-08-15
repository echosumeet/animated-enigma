import React from 'react';
import { Card, Suit } from '../types/game';

interface CardViewProps {
  card?: Card;
  hidden?: boolean;
  size?: 'sm' | 'md' | 'lg';
  highlight?: boolean;
}

const SUIT_SYMBOLS: Record<Suit, string> = {
  spades: '♠',
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
};

const SUIT_COLORS: Record<Suit, string> = {
  spades: 'text-slate-900',
  clubs: 'text-slate-900',
  hearts: 'text-red-600',
  diamonds: 'text-blue-600',
};

export const CardView: React.FC<CardViewProps> = ({
  card,
  hidden = false,
  size = 'md',
  highlight = false,
}) => {
  const sizeClasses = {
    sm: 'w-8 h-12 text-xs rounded',
    md: 'w-12 h-18 text-sm rounded-md',
    lg: 'w-16 h-24 text-base rounded-lg',
  }[size];

  if (hidden || !card) {
    return (
      <div
        className={`${sizeClasses} bg-gradient-to-br from-indigo-900 via-slate-900 to-slate-950 border border-indigo-500/30 flex items-center justify-center shadow-lg relative overflow-hidden select-none transition-transform hover:scale-105`}
      >
        <div className="absolute inset-0 opacity-20 bg-[radial-gradient(#6366f1_1px,transparent_1px)] [background-size:6px_6px]" />
        <div className="w-4 h-6 border border-indigo-400/40 rounded-sm flex items-center justify-center">
          <span className="text-[10px] text-indigo-400/60 font-mono">♠</span>
        </div>
      </div>
    );
  }

  const isRed = card.suit === 'hearts' || card.suit === 'diamonds';
  const suitSymbol = SUIT_SYMBOLS[card.suit];

  return (
    <div
      className={`${sizeClasses} bg-white border ${
        highlight ? 'border-amber-400 ring-2 ring-amber-400/60 scale-105' : 'border-slate-300 shadow-md'
      } flex flex-col justify-between p-1 select-none font-bold relative transition-all`}
    >
      {/* Top Left Rank & Suit */}
      <div className={`flex flex-col items-center leading-none ${isRed ? 'text-rose-600' : 'text-slate-900'}`}>
        <span className="font-mono">{card.rank}</span>
        <span className="text-[11px] leading-tight">{suitSymbol}</span>
      </div>

      {/* Center Watermark Suit */}
      <div className={`self-center text-lg ${isRed ? 'text-rose-500/20' : 'text-slate-900/15'}`}>
        {suitSymbol}
      </div>

      {/* Bottom Right Rank & Suit (Upside down) */}
      <div
        className={`flex flex-col items-center leading-none rotate-180 self-end ${
          isRed ? 'text-rose-600' : 'text-slate-900'
        }`}
      >
        <span className="font-mono">{card.rank}</span>
        <span className="text-[11px] leading-tight">{suitSymbol}</span>
      </div>
    </div>
  );
};
