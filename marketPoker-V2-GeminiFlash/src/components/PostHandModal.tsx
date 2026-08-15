import React, { useEffect } from 'react';
import { PostHandAnalytics } from '../types/game';
import confetti from 'canvas-confetti';
import {
  Trophy,
  TrendingUp,
  Award,
  ArrowRight,
  ShieldCheck,
  Zap,
  Sparkles,
  Flame,
  UserX,
  Coins,
  Landmark,
  Percent,
} from 'lucide-react';

interface PostHandModalProps {
  analytics: PostHandAnalytics;
  onNextHand: () => void;
}

export const PostHandModal: React.FC<PostHandModalProps> = ({ analytics, onNextHand }) => {
  const isUserWinner = analytics.winners.some((w) => w.playerId === 'player-1');
  const synergy = analytics.dualSynergyBonus;
  const isSynergyTriggered = !!synergy?.triggered;
  const loanInterestPaid = analytics.loanInterestDeducted || 0;

  useEffect(() => {
    if (isUserWinner || analytics.userTotalPnl > 0 || isSynergyTriggered) {
      confetti({
        particleCount: isSynergyTriggered ? 120 : 80,
        spread: 80,
        origin: { y: 0.55 },
      });
    }
  }, [isUserWinner, analytics.userTotalPnl, isSynergyTriggered]);

  const pnlColor = analytics.userTotalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
  const pnlSign = analytics.userTotalPnl >= 0 ? '+' : '';

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
      <div className="bg-slate-950 border border-slate-800 rounded-3xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col my-auto max-h-[95vh]">
        {/* Header */}
        <div className="p-5 sm:p-6 border-b border-slate-800 text-center bg-gradient-to-b from-indigo-950/40 via-slate-950 to-slate-950 relative">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 mb-2 shadow-lg">
            <Trophy className="w-8 h-8" />
          </div>

          <div className="flex items-center justify-center gap-2 mb-1">
            <h2 className="text-xl sm:text-2xl font-black text-white tracking-wide">
              Hand #{analytics.handNumber} Completed
            </h2>
            {analytics.tournamentLevel && (
              <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-mono font-bold">
                Level {analytics.tournamentLevel}
              </span>
            )}
          </div>

          <div className="text-sm text-slate-300">
            {analytics.winners.map((w) => (
              <span key={w.playerId} className="font-bold text-amber-400">
                {w.name} won ${w.amount.toLocaleString()} ({w.handDescription})
              </span>
            ))}
          </div>
        </div>

        {/* Analytics Body */}
        <div className="p-4 sm:p-6 space-y-4 overflow-y-auto custom-scrollbar flex-1">
          {/* 4% AUTO-PAID LOAN INTEREST NOTICE */}
          {loanInterestPaid > 0 && (
            <div className="bg-amber-950/40 border border-amber-500/40 rounded-2xl p-3 flex items-center justify-between gap-3 text-xs shadow-md">
              <div className="flex items-center gap-2">
                <span className="p-1.5 rounded-xl bg-amber-500 text-slate-950 font-bold">
                  <Landmark className="w-4 h-4" />
                </span>
                <div>
                  <span className="font-extrabold text-amber-300 block">
                    💸 4% Market Loan Interest Auto-Paid From Winnings
                  </span>
                  <span className="text-slate-400 text-[11px]">
                    Deducted -${loanInterestPaid.toLocaleString()} from gross hand winnings (Active Debt: ${analytics.loanBalance?.toLocaleString() || 0})
                  </span>
                </div>
              </div>
              <span className="font-mono font-black text-rose-400 text-sm bg-slate-950 px-2.5 py-1 rounded-xl border border-slate-800">
                -${loanInterestPaid.toLocaleString()}
              </span>
            </div>
          )}
          {/* DUAL SYNERGY BOUNTY BANNER (If Triggered) */}
          {isSynergyTriggered && synergy && (
            <div className="bg-gradient-to-r from-amber-950/80 via-yellow-950/70 to-purple-950/80 border-2 border-amber-500/60 rounded-2xl p-4 shadow-xl text-left relative overflow-hidden">
              <div className="absolute -right-4 -bottom-4 opacity-10 text-amber-400 pointer-events-none">
                <Zap className="w-32 h-32" />
              </div>

              <div className="flex items-center gap-2 mb-2">
                <span className="p-1.5 rounded-lg bg-amber-500 text-slate-950 font-black">
                  <Flame className="w-4 h-4" />
                </span>
                <span className="text-sm sm:text-base font-extrabold text-amber-300">
                  🔥 DUAL WIN SYNERGY BOUNTY TRIGGERED!
                </span>
                <span className="text-xs font-mono font-black px-2 py-0.5 rounded-full bg-amber-400 text-slate-950 ml-auto">
                  +{synergy.multiplierPct}% BOUNTY
                </span>
              </div>

              <p className="text-xs text-amber-100/90 mb-3">
                <strong className="text-white">{synergy.winningPlayerName}</strong> achieved the ultimate synergy by winning both the <span className="underline decoration-amber-400">Poker Hand</span> and <span className="underline decoration-amber-400">Prediction Market Contracts</span>!
              </p>

              {/* Extraction Breakdown from Losing Players */}
              <div className="bg-slate-950/80 border border-amber-500/30 rounded-xl p-2.5">
                <div className="text-[10px] uppercase font-bold text-amber-400/90 mb-1.5 flex items-center justify-between">
                  <span>Pool Extracted from Losing Players:</span>
                  <span className="font-mono text-xs font-extrabold text-amber-400">
                    +${synergy.bonusAmount.toLocaleString()} Total Awarded
                  </span>
                </div>
                <div className="flex flex-wrap gap-2 text-xs font-mono">
                  {synergy.fundedBy.map((fb) => (
                    <div
                      key={fb.playerId}
                      className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-1.5"
                    >
                      <UserX className="w-3 h-3 text-rose-400" />
                      <span>{fb.name}:</span>
                      <span className="text-rose-400 font-bold">-${fb.amount}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* P&L Performance Cards */}
          <div className="grid grid-cols-3 gap-2.5 sm:gap-3">
            {/* Poker P&L */}
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-2xl text-center">
              <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
                Poker P&L
              </span>
              <span
                className={`font-mono text-sm sm:text-base font-extrabold ${
                  analytics.userPokerPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {analytics.userPokerPnl >= 0 ? '+' : ''}${analytics.userPokerPnl.toLocaleString()}
              </span>
            </div>

            {/* Prediction Market P&L */}
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-2xl text-center">
              <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
                Market P&L
              </span>
              <span
                className={`font-mono text-sm sm:text-base font-extrabold ${
                  analytics.userMarketPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {analytics.userMarketPnl >= 0 ? '+' : ''}${analytics.userMarketPnl.toLocaleString()}
              </span>
            </div>

            {/* Total Net P&L */}
            <div className="bg-slate-900/90 border border-indigo-500/30 p-3 rounded-2xl text-center shadow-inner">
              <span className="text-[10px] uppercase font-bold text-indigo-400 block mb-1">
                Net Hand Total
              </span>
              <span className={`font-mono text-sm sm:text-base font-black ${pnlColor}`}>
                {pnlSign}${analytics.userTotalPnl.toLocaleString()}
              </span>
            </div>
          </div>

          {/* Market Settlements Summary */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-3.5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 block flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
                Live Question & Market Settlements
              </span>
              <span className="text-[10px] font-mono text-slate-500">
                {analytics.marketOutcomes.length} Evaluated
              </span>
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono max-h-40 overflow-y-auto custom-scrollbar pr-1">
              {analytics.marketOutcomes.map((mo) => (
                <div
                  key={mo.marketId}
                  className="flex items-center justify-between p-2 rounded-xl bg-slate-950 border border-slate-800/80"
                >
                  <span className="text-slate-300 truncate max-w-[170px]">{mo.marketName}</span>
                  <span
                    className={`font-bold px-2 py-0.5 rounded text-[10px] ${
                      mo.outcome
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                        : 'bg-rose-950 text-rose-300 border border-rose-800'
                    }`}
                  >
                    {mo.outcome ? 'YES ($1.00)' : 'NO ($0.00)'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Skill Radar Breakdown */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-3.5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2.5 block flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5 text-amber-400" />
              Skill Radar & Elo Rating Evaluation
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs font-mono">
              <div className="p-2 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400">Card Skill</div>
                <div className="font-bold text-white text-sm mt-0.5">84</div>
              </div>
              <div className="p-2 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400">Market Skill</div>
                <div className="font-bold text-emerald-400 text-sm mt-0.5">88</div>
              </div>
              <div className="p-2 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400">Forecasting</div>
                <div className="font-bold text-indigo-400 text-sm mt-0.5">82</div>
              </div>
              <div className="p-2 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400">Risk Mgmt</div>
                <div className="font-bold text-cyan-400 text-sm mt-0.5">79</div>
              </div>
              <div className="col-span-2 sm:col-span-1 p-2 rounded-xl bg-indigo-950/80 border border-indigo-800">
                <div className="text-[10px] text-indigo-300">Rating</div>
                <div className="font-extrabold text-amber-400 text-sm mt-0.5">1,785</div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Button: Next Hand */}
        <div className="p-4 sm:p-5 border-t border-slate-800 bg-slate-900/40 flex justify-end">
          <button
            onClick={onNextHand}
            className="w-full sm:w-auto px-8 py-3 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-black text-sm flex items-center justify-center gap-2 shadow-xl shadow-emerald-500/20 transition-all cursor-pointer active:scale-95"
          >
            <span>Deal Next Hand</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
