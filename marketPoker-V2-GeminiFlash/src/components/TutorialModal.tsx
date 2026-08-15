import React from 'react';
import { BookOpen, X, TrendingUp, ShieldCheck, Zap, DollarSign, ArrowRight } from 'lucide-react';

interface TutorialModalProps {
  onClose: () => void;
}

export const TutorialModal: React.FC<TutorialModalProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
      <div className="bg-slate-950 border border-slate-800 rounded-3xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-5 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-white">
                How to Play Market Poker
              </h2>
              <p className="text-xs text-slate-400">
                Master the fusion of Texas Hold'em, order book trading, and game theory
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4 flex-1 overflow-y-auto text-xs text-slate-300 leading-relaxed">
          {/* Card 1: Unified Bankroll */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex gap-3.5">
            <div className="p-2 rounded-xl bg-amber-500/10 text-amber-400 shrink-0 h-fit">
              <DollarSign className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white mb-1">
                1. Unified Shared Bankroll
              </h3>
              <p>
                You start each session with <strong>10,000 play-money credits</strong>. Every poker bet and every prediction market order comes from this <strong>exact same cash pool</strong>. Choose whether to risk capital in the poker pot, buy undervalued contracts, or hold cash.
              </p>
            </div>
          </div>

          {/* Card 2: Private Alpha */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex gap-3.5">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 shrink-0 h-fit">
              <TrendingUp className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white mb-1">
                2. Private Cards as Trading Alpha
              </h3>
              <p>
                You hold 2 private hole cards that opponents cannot see. When you hold pocket Aces (<span className="text-white font-mono">A♠ A♦</span>), public markets might price your win probability at only 20¢ (20%). You can aggressively buy your YES contract at a massive quantitative discount!
              </p>
            </div>
          </div>

          {/* Card 3: Trading as Bluffing */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex gap-3.5">
            <div className="p-2 rounded-xl bg-rose-500/10 text-rose-400 shrink-0 h-fit">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white mb-1">
                3. Market Order Bluffing & Signaling
              </h3>
              <p>
                Prediction markets update with public order flow. Buying your own WIN contract aggressively moves the price upward, sending a visible signal of extreme strength to bait folds from cautious opponents!
              </p>
            </div>
          </div>

          {/* Card 4: Settlement */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex gap-3.5">
            <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-400 shrink-0 h-fit">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white mb-1">
                4. Showdown & Dual Settlement
              </h3>
              <p>
                At showdown, the poker pot is awarded to the best 5-card hand, and all 7 prediction contracts settle to either <strong>$1.00 (YES)</strong> or <strong>$0.00 (NO)</strong>. Post-hand analytics break down your edge, ROI, and 5-dimensional skill rating!
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/40 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs flex items-center gap-1.5 shadow-lg shadow-indigo-600/20 transition-all cursor-pointer"
          >
            <span>Got it, let's play</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
