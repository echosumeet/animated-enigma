import React from 'react';
import {
  TrendingUp,
  Volume2,
  VolumeX,
  BookOpen,
  Cpu,
  BarChart3,
  Coins,
  ShieldCheck,
  Trophy,
  Flame,
  Zap,
  Sparkles,
  Landmark,
} from 'lucide-react';
import { GamePhase, Player, Position, Market } from '../types/game';
import { LedgerEngine } from '../engine/ledger';

interface HeaderProps {
  handNumber: number;
  tournamentLevel?: number;
  handsInCurrentLevel?: number;
  handsPerLevel?: number;
  synergyMultiplierPct?: number;
  smallBlind?: number;
  bigBlind?: number;
  phase: GamePhase;
  userPlayer: Player;
  positions: Position[];
  markets: Market[];
  isSoundOn: boolean;
  onToggleSound: () => void;
  onOpenTutorial: () => void;
  onOpenSimulation: () => void;
  onOpenInspector: () => void;
  onOpenShowcase: () => void;
  onOpenLoanModal?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  handNumber,
  tournamentLevel = 1,
  handsInCurrentLevel = 1,
  handsPerLevel = 3,
  synergyMultiplierPct = 15,
  smallBlind = 50,
  bigBlind = 100,
  phase,
  userPlayer,
  positions,
  markets,
  isSoundOn,
  onToggleSound,
  onOpenTutorial,
  onOpenSimulation,
  onOpenInspector,
  onOpenShowcase,
  onOpenLoanModal,
}) => {
  const equityData = LedgerEngine.calculatePlayerEquity(userPlayer, positions, markets);
  const pnlColor = equityData.totalEquity >= 10000 ? 'text-emerald-400' : 'text-rose-400';
  const pnlSign = equityData.totalEquity >= 10000 ? '+' : '';
  const loanBalance = userPlayer.loanBalance || 0;
  const isCashLow = userPlayer.cashBalance < (bigBlind || 100) * 2;

  const formatPhaseName = (p: GamePhase) => {
    switch (p) {
      case 'WAITING':
        return 'Waiting for Hand';
      case 'DEALING':
        return 'Dealing Cards';
      case 'PREFLOP_TRADING':
        return 'Preflop Trading';
      case 'PREFLOP_BETTING':
        return 'Preflop Betting';
      case 'FLOP_REVEAL':
      case 'FLOP_TRADING':
        return 'Flop Trading';
      case 'FLOP_BETTING':
        return 'Flop Betting';
      case 'TURN_REVEAL':
      case 'TURN_TRADING':
        return 'Turn Trading';
      case 'TURN_BETTING':
        return 'Turn Betting';
      case 'RIVER_REVEAL':
      case 'RIVER_TRADING':
        return 'River Trading';
      case 'RIVER_BETTING':
        return 'River Betting';
      case 'SHOWDOWN':
        return 'Showdown';
      case 'MARKET_SETTLEMENT':
        return 'Settlement';
      case 'HAND_RESULTS':
        return 'Hand Analytics';
      default:
        return p;
    }
  };

  const isTradingPhase = phase.includes('TRADING');

  return (
    <header className="bg-slate-950/90 backdrop-blur border-b border-slate-800/80 px-3 sm:px-4 py-2 flex items-center justify-between z-30 sticky top-0">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-indigo-600 via-amber-500 to-emerald-400 p-0.5 shadow-md flex items-center justify-center">
          <div className="w-full h-full bg-slate-950 rounded-[7px] flex items-center justify-center font-black text-amber-400 text-xs tracking-tighter">
            M♠P
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-extrabold text-sm sm:text-base tracking-wider bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
              MARKET POKER
            </h1>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              Live Tournament
            </span>
          </div>
          <p className="text-[11px] text-slate-400 flex items-center gap-1.5 flex-wrap">
            <span className="font-mono text-slate-300">Hand #{handNumber || 1}</span>
            <span>•</span>
            <span
              className={`font-semibold inline-flex items-center gap-1 ${
                isTradingPhase ? 'text-amber-400 animate-pulse' : 'text-slate-300'
              }`}
            >
              {isTradingPhase && <TrendingUp className="w-3 h-3 inline" />}
              {formatPhaseName(phase)}
            </span>
          </p>
        </div>
      </div>

      {/* Tournament Level & Escalating Blinds Meter */}
      <div className="hidden lg:flex items-center gap-3 bg-slate-900/90 border border-slate-800 rounded-xl px-3 py-1 shadow-inner">
        <div className="flex items-center gap-2">
          <div className="p-1 rounded-lg bg-amber-500/15 text-amber-400 border border-amber-500/30">
            <Trophy className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-bold text-white">Level {tournamentLevel}</span>
              <span className="text-[10px] font-mono text-amber-400 font-bold bg-amber-500/10 px-1.5 py-0.2 rounded border border-amber-500/20">
                {smallBlind}/{bigBlind} Blinds
              </span>
            </div>
            <div className="text-[10px] text-slate-400">
              Hand {handsInCurrentLevel}/{handsPerLevel} in level
            </div>
          </div>
        </div>

        <div className="h-6 w-px bg-slate-800" />

        {/* Dual Win Bounty Multiplier */}
        <div className="flex items-center gap-1.5 bg-gradient-to-r from-amber-950/60 to-purple-950/60 px-2.5 py-1 rounded-lg border border-amber-500/30">
          <Zap className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
          <div>
            <div className="text-[9px] uppercase font-bold tracking-wider text-amber-300">
              Dual Synergy Bounty
            </div>
            <div className="text-xs font-mono font-extrabold text-amber-400">
              +{synergyMultiplierPct}% From Losers
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Economic Ticker */}
      <div className="hidden md:flex items-center gap-5 bg-slate-900/90 border border-slate-800 rounded-xl px-3.5 py-1.5 shadow-inner">
        {/* Total Equity */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1">
            <ShieldCheck className="w-3 h-3 text-indigo-400" />
            Total Equity
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono font-bold text-sm text-white">
              {equityData.totalEquity.toLocaleString()}
            </span>
            <span className={`font-mono text-[11px] font-semibold ${pnlColor}`}>
              ({pnlSign}{(equityData.totalEquity - 10000).toLocaleString()})
            </span>
          </div>
        </div>

        <div className="h-6 w-px bg-slate-800" />

        {/* Liquid Cash */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1">
            <Coins className="w-3 h-3 text-amber-400" />
            Liquid Cash
          </div>
          <div className="font-mono font-bold text-sm text-amber-400">
            ${equityData.liquidCash.toLocaleString()}
          </div>
        </div>

        {/* Loan Debt Indicator (if active) */}
        {loanBalance > 0 && (
          <>
            <div className="h-6 w-px bg-slate-800" />
            <div
              onClick={onOpenLoanModal}
              className="cursor-pointer group flex items-center gap-1.5 bg-amber-950/60 hover:bg-amber-900/70 border border-amber-500/40 px-2 py-1 rounded-lg transition-all"
            >
              <Landmark className="w-3.5 h-3.5 text-amber-400" />
              <div>
                <div className="text-[9px] uppercase font-bold text-amber-400 flex items-center gap-1">
                  <span>Loan (4% Int)</span>
                </div>
                <div className="font-mono font-extrabold text-xs text-amber-300">
                  ${loanBalance.toLocaleString()}
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Control Buttons */}
      <div className="flex items-center gap-2">
        {/* Borrow Facility Trigger Button */}
        {onOpenLoanModal && (
          <button
            onClick={onOpenLoanModal}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold text-xs transition-all cursor-pointer shadow-sm ${
              loanBalance > 0
                ? 'bg-amber-500/20 border border-amber-500/50 text-amber-300 hover:bg-amber-500/30'
                : isCashLow
                ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 hover:from-amber-400 hover:to-amber-500 animate-pulse ring-2 ring-amber-400/50'
                : 'bg-slate-900 border border-slate-800 text-amber-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Landmark className="w-3.5 h-3.5 text-amber-400" />
            <span>{loanBalance > 0 ? 'Manage Loan' : 'Borrow (4%)'}</span>
          </button>
        )}

        {/* Sound Toggle */}
        <button
          onClick={onToggleSound}
          title={isSoundOn ? 'Mute Audio' : 'Unmute Audio'}
          className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          {isSoundOn ? <Volume2 className="w-4 h-4 text-emerald-400" /> : <VolumeX className="w-4 h-4 text-slate-500" />}
        </button>

        {/* Tutorial */}
        <button
          onClick={onOpenTutorial}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
          <span className="hidden sm:inline">Guide</span>
        </button>

        {/* 100-Hand Simulator */}
        <button
          onClick={onOpenSimulation}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
          <span className="hidden sm:inline">Simulate</span>
        </button>

        {/* AI Engineering Showcase */}
        <button
          onClick={onOpenShowcase}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-gradient-to-r from-purple-950/80 to-indigo-950/80 border border-purple-500/40 text-xs text-purple-200 hover:text-white hover:border-purple-400 transition-all shadow-sm"
        >
          <Sparkles className="w-3.5 h-3.5 text-purple-400 animate-pulse" />
          <span className="font-bold">Showcase</span>
        </button>

        {/* Dev Inspector */}
        <button
          onClick={onOpenInspector}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-indigo-950/60 border border-indigo-800/60 text-xs text-indigo-300 hover:text-white hover:bg-indigo-900/60 transition-colors"
        >
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span className="hidden sm:inline">Inspector</span>
        </button>
      </div>
    </header>
  );
};
