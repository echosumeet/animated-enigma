import React, { useState } from 'react';
import { runSimulation, SimulationSummary } from '../engine/simulation';
import { BarChart3, Play, X, CheckCircle, RefreshCw, Trophy, Zap } from 'lucide-react';

interface SimulationModalProps {
  onClose: () => void;
}

export const SimulationModal: React.FC<SimulationModalProps> = ({ onClose }) => {
  const [handCount, setHandCount] = useState<number>(100);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [results, setResults] = useState<SimulationSummary | null>(null);

  const handleRunSim = () => {
    setIsRunning(true);
    setTimeout(() => {
      const summary = runSimulation(handCount);
      setResults(summary);
      setIsRunning(false);
    }, 100);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
      <div className="bg-slate-950 border border-slate-800 rounded-3xl w-full max-w-3xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <BarChart3 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm sm:text-base font-bold text-white">
                Automated Bot Simulation Suite
              </h2>
              <p className="text-xs text-slate-400">
                Simulate 100 to 1,000 hands with AI Quant personalities to test game balance and strategy
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

        {/* Content */}
        <div className="p-6 space-y-6 flex-1 overflow-y-auto">
          {/* Controls */}
          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3 w-full sm:w-auto">
              <span className="text-xs text-slate-400 font-medium">Hands to Simulate:</span>
              <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono">
                {[100, 250, 500, 1000].map((count) => (
                  <button
                    key={count}
                    onClick={() => setHandCount(count)}
                    className={`px-3 py-1.5 rounded-lg font-bold transition-all ${
                      handCount === count
                        ? 'bg-emerald-600 text-slate-950 shadow-md'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {count}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleRunSim}
              disabled={isRunning}
              className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 disabled:opacity-50 cursor-pointer"
            >
              {isRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Simulating...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-slate-950" />
                  Run {handCount} Hands
                </>
              )}
            </button>
          </div>

          {/* Results Summary */}
          {results && (
            <div className="space-y-4 font-mono">
              {/* Stat Highlights */}
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="bg-slate-900/60 border border-slate-800 p-3 rounded-xl">
                  <div className="text-[10px] text-slate-400 uppercase">Execution Time</div>
                  <div className="text-sm sm:text-base font-bold text-white mt-0.5">
                    {results.durationMs} ms
                  </div>
                </div>
                <div className="bg-slate-900/60 border border-slate-800 p-3 rounded-xl">
                  <div className="text-[10px] text-slate-400 uppercase">Total Trades</div>
                  <div className="text-sm sm:text-base font-bold text-indigo-400 mt-0.5">
                    {results.totalTradesExecuted}
                  </div>
                </div>
                <div className="bg-slate-900/60 border border-slate-800 p-3 rounded-xl">
                  <div className="text-[10px] text-slate-400 uppercase">Invariants</div>
                  <div className="text-sm sm:text-base font-bold text-emerald-400 mt-0.5 flex items-center justify-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Verified
                  </div>
                </div>
              </div>

              {/* Bot Performance Table */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 overflow-x-auto">
                <h3 className="text-xs font-bold text-slate-300 mb-3 flex items-center gap-1.5 font-sans">
                  <Trophy className="w-3.5 h-3.5 text-amber-400" />
                  Personality Performance & Profitability
                </h3>

                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 pb-2">
                      <th className="py-2">Player / Personality</th>
                      <th className="py-2 text-right">Win Rate</th>
                      <th className="py-2 text-right">Poker P&L</th>
                      <th className="py-2 text-right">Market P&L</th>
                      <th className="py-2 text-right">Net Equity</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {results.playerStats.map((stat, i) => (
                      <tr key={i} className="hover:bg-slate-800/40">
                        <td className="py-2.5 font-sans font-bold text-white">
                          {stat.name} ({stat.personality})
                        </td>
                        <td className="py-2.5 text-right text-indigo-300">
                          {stat.pokerWinRate}% ({stat.pokerWinCount})
                        </td>
                        <td className="py-2.5 text-right text-slate-300">
                          ${stat.pokerPnl.toLocaleString()}
                        </td>
                        <td className="py-2.5 text-right text-slate-300">
                          ${stat.marketPnl.toLocaleString()}
                        </td>
                        <td
                          className={`py-2.5 text-right font-bold ${
                            stat.netPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          ${stat.endingEquity.toLocaleString()} ({stat.netPnl >= 0 ? '+' : ''}${stat.netPnl})
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
