import React, { useState } from 'react';
import { GameTableState } from '../types/game';
import { estimatePokerEquity } from '../engine/poker';
import { Cpu, X, CheckCircle, ShieldAlert, Terminal, FileText } from 'lucide-react';

interface DevInspectorProps {
  state: GameTableState;
  onClose: () => void;
}

export const DevInspector: React.FC<DevInspectorProps> = ({ state, onClose }) => {
  const [activeTab, setActiveTab] = useState<'EQUITIES' | 'EVENTS' | 'LEDGER'>('EQUITIES');

  const totalBankroll = state.players.reduce((sum, p) => sum + p.cashBalance + p.currentBet, 0);
  const invariantCheck = Math.abs(totalBankroll - 50000) < 5;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-3 sm:p-6">
      <div className="bg-slate-950 border border-indigo-500/30 rounded-3xl w-full max-w-4xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between bg-indigo-950/30">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
                Quantitative Developer Inspector
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300">
                  Authoritative Engine
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Inspect true card equities, internal state-machine variables, and ledger audit
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

        {/* Tab Switcher */}
        <div className="flex items-center gap-2 px-5 pt-3 border-b border-slate-800 text-xs font-mono">
          <button
            onClick={() => setActiveTab('EQUITIES')}
            className={`pb-2.5 px-3 border-b-2 font-bold transition-all ${
              activeTab === 'EQUITIES'
                ? 'border-indigo-400 text-indigo-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            True Equities vs Market Price
          </button>
          <button
            onClick={() => setActiveTab('EVENTS')}
            className={`pb-2.5 px-3 border-b-2 font-bold transition-all ${
              activeTab === 'EVENTS'
                ? 'border-indigo-400 text-indigo-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Event Stream ({state.events.length})
          </button>
          <button
            onClick={() => setActiveTab('LEDGER')}
            className={`pb-2.5 px-3 border-b-2 font-bold transition-all ${
              activeTab === 'LEDGER'
                ? 'border-indigo-400 text-indigo-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            Economic Invariants & Ledger
          </button>
        </div>

        {/* Tab Body */}
        <div className="p-5 flex-1 overflow-y-auto font-mono text-xs">
          {activeTab === 'EQUITIES' && (
            <div className="space-y-4">
              <div className="p-3 rounded-xl bg-indigo-950/40 border border-indigo-800/40 text-indigo-200 text-[11px]">
                💡 In Market Poker, true mathematical probabilities derived from private hole cards often diverge from public market prices. This divergence represents the strategic alpha exploited by humans and Quant bots.
              </div>

              <div className="space-y-2">
                {state.players.map((player) => {
                  const trueEquity = player.cards.length >= 2
                    ? estimatePokerEquity(player.cards, state.communityCards, 4, 150)
                    : 0.20;
                  const winMarket = state.markets.find((m) => m.targetPlayerId === player.id);
                  const mktPrice = winMarket ? winMarket.currentYesPrice : 0.20;
                  const edge = trueEquity - mktPrice;

                  return (
                    <div
                      key={player.id}
                      className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white">{player.name}</span>
                        {player.cards.length >= 2 && (
                          <span className="text-slate-400">
                            [{player.cards.map((c) => `${c.rank}${c.suit[0]}`).join(' ')}]
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-4">
                        <div>
                          <span className="text-slate-400">True Equity: </span>
                          <span className="font-bold text-indigo-300">
                            {(trueEquity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Market Price: </span>
                          <span className="font-bold text-amber-400">
                            ${mktPrice.toFixed(2)}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400">Edge: </span>
                          <span
                            className={`font-bold ${
                              edge >= 0 ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {edge >= 0 ? '+' : ''}{(edge * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'EVENTS' && (
            <div className="space-y-1.5 bg-slate-950 p-3 rounded-xl border border-slate-800">
              {state.events.slice(-20).map((evt) => (
                <div
                  key={evt.id}
                  className={`text-[11px] py-1 border-b border-slate-900 flex items-start gap-2 ${
                    evt.highlight ? 'text-amber-300 font-semibold' : 'text-slate-400'
                  }`}
                >
                  <span className="text-slate-600">[{new Date(evt.timestamp).toLocaleTimeString()}]</span>
                  <span>{evt.text}</span>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'LEDGER' && (
            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="font-bold text-white">System Economic Invariance Check</div>
                  <div className="text-[11px] text-slate-400">
                    Total circulating credits across 5 player wallets + pot: ${totalBankroll.toLocaleString()}
                  </div>
                </div>
                <div
                  className={`px-3 py-1 rounded-full font-bold text-xs flex items-center gap-1.5 ${
                    invariantCheck
                      ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                      : 'bg-rose-950 text-rose-300 border border-rose-800'
                  }`}
                >
                  <CheckCircle className="w-3.5 h-3.5" />
                  {invariantCheck ? 'Invariants Verified (Balanced)' : 'Discrepancy Detected'}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
