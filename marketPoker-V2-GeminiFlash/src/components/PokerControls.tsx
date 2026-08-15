import React, { useState } from 'react';
import { GamePhase, Player } from '../types/game';
import { Check, XCircle, ArrowUpRight, FastForward, DollarSign, Landmark } from 'lucide-react';

interface PokerControlsProps {
  userPlayer: Player;
  currentTurnIndex: number;
  currentHighestBet: number;
  minRaise: number;
  mainPot: number;
  phase: GamePhase;
  turnTimeRemaining?: number;
  onAction: (action: 'CHECK' | 'CALL' | 'BET' | 'RAISE' | 'FOLD', amount?: number) => void;
  onAdvanceToBetting: () => void;
  onFastForward?: () => void;
  onOpenLoanModal?: () => void;
  onQuickBorrow?: (amount: number) => void;
}

export const PokerControls: React.FC<PokerControlsProps> = ({
  userPlayer,
  currentTurnIndex,
  currentHighestBet,
  minRaise,
  mainPot,
  phase,
  turnTimeRemaining = 15,
  onAction,
  onAdvanceToBetting,
  onFastForward,
  onOpenLoanModal,
  onQuickBorrow,
}) => {
  const isUserTurn = currentTurnIndex === 0 && phase.endsWith('_BETTING');
  const isTradingWindow = phase.endsWith('_TRADING');
  const isFolded = userPlayer.status === 'folded';
  const callAmount = Math.max(0, currentHighestBet - userPlayer.currentBet);
  const canCheck = callAmount === 0;
  const isUrgent = isUserTurn && turnTimeRemaining <= 5;
  const isCashLow = userPlayer.cashBalance < minRaise;

  const [raiseValue, setRaiseValue] = useState<number>(Math.max(minRaise, currentHighestBet + 100));

  const minValidRaise = Math.max(minRaise, currentHighestBet + 50);
  const maxValidRaise = userPlayer.cashBalance;

  // Preset buttons
  const setPreset = (fraction: number) => {
    const target = Math.round(mainPot * fraction);
    setRaiseValue(Math.min(maxValidRaise, Math.max(minValidRaise, target)));
  };

  if (isFolded && !isTradingWindow) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-3 sm:p-4 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-xl">
        <div className="flex items-center gap-2 text-xs text-slate-300 font-mono">
          <span className="w-2 h-2 rounded-full bg-rose-500" />
          <span>You folded this hand. Observing opponents and live prediction markets.</span>
        </div>
        {onFastForward && (
          <button
            onClick={onFastForward}
            className="w-full sm:w-auto px-4 py-2 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 shadow-md shadow-amber-500/20 transition-all cursor-pointer"
          >
            <FastForward className="w-3.5 h-3.5" />
            Fast-Forward to Showdown
          </button>
        )}
      </div>
    );
  }

  if (isTradingWindow) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-3 sm:p-4 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-xl">
        <div>
          <div className="text-xs uppercase tracking-wider text-amber-400 font-bold flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
            Trading Window Active {isFolded && <span className="text-slate-400 font-normal font-mono">(You Folded Poker)</span>}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Trade prediction contracts in the right panel or proceed directly to poker betting.
          </p>
        </div>
        <div className="flex items-center gap-2 w-full sm:w-auto">
          {isFolded && onFastForward && (
            <button
              onClick={onFastForward}
              className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs flex items-center justify-center gap-1.5 transition-all cursor-pointer"
            >
              <FastForward className="w-3.5 h-3.5 text-amber-400" />
              Fast-Forward
            </button>
          )}
          <button
            onClick={onAdvanceToBetting}
            className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-amber-500/20 transition-all cursor-pointer"
          >
            <FastForward className="w-4 h-4" />
            Start Betting Round
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Emergency Borrow Banner if user is out of cash */}
      {isCashLow && onOpenLoanModal && (
        <div className="bg-gradient-to-r from-amber-950/70 via-slate-900 to-slate-950 border border-amber-500/50 rounded-2xl p-2.5 sm:p-3 flex flex-wrap items-center justify-between gap-2 shadow-lg animate-pulse">
          <div className="flex items-center gap-2 text-xs">
            <span className="p-1 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/30">
              <Landmark className="w-3.5 h-3.5" />
            </span>
            <span className="text-amber-300 font-medium">
              Low on chips (${userPlayer.cashBalance.toLocaleString()} left)? <strong>Borrow from Market @ 4% Interest</strong>
            </span>
          </div>
          <div className="flex items-center gap-2">
            {onQuickBorrow && (
              <button
                type="button"
                onClick={() => onQuickBorrow(2500)}
                className="px-3 py-1 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-black text-xs transition-all cursor-pointer shadow-sm"
              >
                + Quick Borrow $2,500
              </button>
            )}
            <button
              type="button"
              onClick={onOpenLoanModal}
              className="px-3 py-1 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs border border-slate-700 transition-all cursor-pointer"
            >
              Choose Amount
            </button>
          </div>
        </div>
      )}

      <div
        className={`bg-slate-950/95 border rounded-2xl p-3 sm:p-4 shadow-2xl transition-all ${
          isUserTurn
            ? isUrgent
              ? 'border-rose-500 ring-2 ring-rose-500/50 shadow-[0_0_20px_rgba(244,63,94,0.3)]'
              : 'border-amber-400 ring-2 ring-amber-400/40 shadow-[0_0_20px_rgba(245,158,11,0.25)]'
            : 'border-slate-800 opacity-60 pointer-events-none'
        }`}
      >
        {/* Action Header with Timer when Active */}
        {isUserTurn && (
          <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-slate-800/80">
            <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${isUrgent ? 'bg-rose-500 animate-ping' : 'bg-amber-400 animate-ping'}`} />
              <span className="text-xs uppercase font-extrabold tracking-wider text-white">
                Action On You
              </span>
            </div>
            <div className={`px-2.5 py-0.5 rounded-full font-mono font-extrabold text-xs flex items-center gap-1.5 border shadow-sm ${
              isUrgent
                ? 'bg-rose-950/90 text-rose-300 border-rose-500 animate-pulse'
                : 'bg-amber-950/90 text-amber-300 border-amber-400/60'
            }`}>
              <span>⏳ Clock:</span>
              <span className="text-white font-black">{turnTimeRemaining}s</span>
            </div>
          </div>
        )}

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Quick Poker Action Buttons */}
          <div className="flex items-center gap-2 w-full sm:w-auto">
            {/* Fold */}
            <button
              onClick={() => onAction('FOLD')}
              className="flex-1 sm:flex-initial px-4 py-2.5 rounded-xl bg-rose-950/80 hover:bg-rose-900 border border-rose-800 text-rose-300 font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md cursor-pointer"
            >
              <XCircle className="w-4 h-4" />
              Fold
            </button>

            {/* Check / Call */}
            {canCheck ? (
              <button
                onClick={() => onAction('CHECK')}
                className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl bg-emerald-950/90 hover:bg-emerald-900 border border-emerald-700 text-emerald-300 font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md cursor-pointer"
              >
                <Check className="w-4 h-4" />
                Check
              </button>
            ) : (
              <button
                onClick={() => onAction('CALL', callAmount)}
                className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md shadow-emerald-500/20 cursor-pointer"
              >
                <DollarSign className="w-4 h-4" />
                Call ${callAmount}
              </button>
            )}

            {/* Bet / Raise Trigger */}
            <button
              onClick={() => onAction(canCheck ? 'BET' : 'RAISE', raiseValue)}
              disabled={userPlayer.cashBalance < minValidRaise}
              className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md shadow-amber-500/20 disabled:opacity-50 cursor-pointer"
            >
              <ArrowUpRight className="w-4 h-4" />
              {canCheck ? 'Bet' : 'Raise to'} ${raiseValue}
            </button>
          </div>

          {/* Raise Slider & Quick Preset Buttons */}
          <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
            {/* Presets */}
            <div className="flex items-center gap-1.5 w-full sm:w-auto justify-center">
              <button
                onClick={() => setPreset(0.5)}
                className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
              >
                1/2 Pot
              </button>
              <button
                onClick={() => setPreset(1.0)}
                className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
              >
                Pot
              </button>
              <button
                onClick={() => setRaiseValue(userPlayer.cashBalance)}
                className="px-2.5 py-1 rounded-lg bg-indigo-950 border border-indigo-800 text-[11px] text-indigo-300 hover:text-white hover:bg-indigo-900 transition-colors"
              >
                All In
              </button>
            </div>

            {/* Slider */}
            <div className="flex items-center gap-2 w-full sm:w-48">
              <input
                type="range"
                min={minValidRaise}
                max={maxValidRaise}
                step={25}
                value={raiseValue}
                onChange={(e) => setRaiseValue(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <span className="font-mono text-xs font-bold text-amber-400 w-12 text-right">
                ${raiseValue}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
