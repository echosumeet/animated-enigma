import React from 'react';
import { Market, Position } from '../types/game';
import { Layers, TrendingUp, X, ArrowDownRight, ArrowUpRight } from 'lucide-react';

interface PortfolioDrawerProps {
  positions: Position[];
  markets: Market[];
  onClosePosition: (marketId: string, side: 'SELL_YES' | 'SELL_NO', qty: number) => void;
}

export const PortfolioDrawer: React.FC<PortfolioDrawerProps> = ({
  positions,
  markets,
  onClosePosition,
}) => {
  const activePositions = positions.filter((p) => p.yesContracts > 0 || p.noContracts > 0);
  const marketMap = new Map<string, Market>(markets.map((m) => [m.id, m]));

  if (activePositions.length === 0) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 text-center">
        <div className="text-xs text-slate-400 font-mono flex items-center justify-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-slate-500" />
          No open prediction positions held. Click on any contract to take a position.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-950/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-3">
        <span className="text-xs uppercase tracking-wider text-slate-400 font-bold flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-indigo-400" />
          Active Prediction Portfolio ({activePositions.length})
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {activePositions.map((pos) => {
          const m = marketMap.get(pos.marketId);
          if (!m) return null;

          const isYes = pos.yesContracts > 0;
          const qty = isYes ? pos.yesContracts : pos.noContracts;
          const avgPrice = isYes ? pos.avgYesPrice : pos.avgNoPrice;
          const spotPrice = isYes ? m.currentYesPrice : m.currentNoPrice;
          const unrealized = Number((qty * (spotPrice - avgPrice)).toFixed(2));
          const pnlColor = unrealized >= 0 ? 'text-emerald-400' : 'text-rose-400';

          return (
            <div
              key={pos.marketId}
              className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-xs font-bold text-white truncate max-w-[140px]">
                    {m.name}
                  </span>
                  <span
                    className={`font-mono text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      isYes ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-rose-950 text-rose-300 border border-rose-800'
                    }`}
                  >
                    {qty} {isYes ? 'YES' : 'NO'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 my-2">
                  <div>
                    <span>Avg Entry: </span>
                    <span className="text-white font-semibold">${avgPrice.toFixed(2)}</span>
                  </div>
                  <div>
                    <span>Current Spot: </span>
                    <span className="text-white font-semibold">${spotPrice.toFixed(2)}</span>
                  </div>
                  <div>
                    <span>Unrealized: </span>
                    <span className={`font-bold ${pnlColor}`}>
                      {unrealized >= 0 ? '+' : ''}${unrealized}
                    </span>
                  </div>
                  <div>
                    <span>Realized: </span>
                    <span className="text-slate-300 font-semibold">${pos.realizedPnl}</span>
                  </div>
                </div>
              </div>

              {/* Close Position Button */}
              <button
                onClick={() => onClosePosition(pos.marketId, isYes ? 'SELL_YES' : 'SELL_NO', qty)}
                className="w-full mt-1 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white font-mono text-[11px] font-semibold transition-colors flex items-center justify-center gap-1"
              >
                Close Position ({qty} @ ${spotPrice.toFixed(2)})
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
