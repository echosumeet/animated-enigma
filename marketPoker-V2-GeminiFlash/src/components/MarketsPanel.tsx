import React, { useState } from 'react';
import { Market, Position } from '../types/game';
import {
  TrendingUp,
  Zap,
  Flame,
  Award,
  Layers,
  CheckCircle2,
  XCircle,
  Timer,
  Sparkles,
} from 'lucide-react';

interface MarketsPanelProps {
  markets: Market[];
  positions: Position[];
  selectedMarketId: string;
  synergyMultiplierPct: number;
  onSelectMarket: (marketId: string) => void;
  onQuickTrade: (marketId: string, side: 'BUY_YES' | 'BUY_NO', dollars?: number) => void;
}

type FilterTab = 'ALL' | 'LIVE_STREET' | 'CORE_HAND' | 'RESOLVED';

export const MarketsPanel: React.FC<MarketsPanelProps> = ({
  markets,
  positions,
  selectedMarketId,
  synergyMultiplierPct,
  onSelectMarket,
  onQuickTrade,
}) => {
  const [activeTab, setActiveTab] = useState<FilterTab>('ALL');
  const [quickBetAmount, setQuickBetAmount] = useState<number>(50);

  const getPositionForMarket = (marketId: string) => {
    return positions.find((p) => p.marketId === marketId && (p.yesContracts > 0 || p.noContracts > 0));
  };

  const filteredMarkets = markets.filter((m) => {
    if (activeTab === 'RESOLVED') return m.resolved;
    if (m.resolved) return false;
    if (activeTab === 'LIVE_STREET') return m.lifespan === 'SHORT_LIVED';
    if (activeTab === 'CORE_HAND') return m.lifespan === 'CORE_HAND' || m.lifespan === 'LONG_LIVED';
    return true;
  });

  const activeLiveCount = markets.filter((m) => !m.resolved && m.lifespan === 'SHORT_LIVED').length;

  return (
    <div className="bg-slate-950/95 border border-slate-800/90 rounded-2xl p-3 sm:p-4 flex flex-col h-full shadow-2xl backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between pb-2.5 border-b border-slate-800/80 mb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10 border border-amber-500/25 text-amber-400">
            <Zap className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xs sm:text-sm font-bold text-white tracking-wide">
                Live Prediction Markets
              </h2>
              <span className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded-full bg-indigo-500/15 border border-indigo-500/30 text-indigo-300">
                Kalshi • Poly
              </span>
            </div>
            <p className="text-[10px] text-slate-400">
              Live questions with +{synergyMultiplierPct}% Dual Win Synergy Bounty
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 bg-slate-900/90 px-2 py-1 rounded-lg border border-slate-800 text-[10px] font-mono">
          <span className="text-slate-400">Bet Size:</span>
          {[25, 50, 100].map((amt) => (
            <button
              key={amt}
              onClick={() => setQuickBetAmount(amt)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold transition-all ${
                quickBetAmount === amt
                  ? 'bg-amber-500 text-slate-950 shadow-sm'
                  : 'text-slate-400 hover:text-white bg-slate-800/60'
              }`}
            >
              ${amt}
            </button>
          ))}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-1.5 pb-2 mb-2 border-b border-slate-800/60 overflow-x-auto no-scrollbar">
        <button
          onClick={() => setActiveTab('ALL')}
          className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all whitespace-nowrap flex items-center gap-1.5 ${
            activeTab === 'ALL'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
          }`}
        >
          <Flame className="w-3 h-3 text-amber-400" />
          All Active ({markets.filter((m) => !m.resolved).length})
        </button>

        <button
          onClick={() => setActiveTab('LIVE_STREET')}
          className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all whitespace-nowrap flex items-center gap-1.5 ${
            activeTab === 'LIVE_STREET'
              ? 'bg-amber-600 text-white shadow-md'
              : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
          }`}
        >
          <Zap className="w-3 h-3 text-amber-400" />
          Live Street Bets
          {activeLiveCount > 0 && (
            <span className="px-1 py-0.2 rounded-full bg-amber-400/20 text-amber-300 text-[9px] font-mono animate-pulse">
              {activeLiveCount}
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveTab('CORE_HAND')}
          className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all whitespace-nowrap flex items-center gap-1.5 ${
            activeTab === 'CORE_HAND'
              ? 'bg-purple-600 text-white shadow-md'
              : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
          }`}
        >
          <Award className="w-3 h-3 text-purple-400" />
          Hand & Props
        </button>

        <button
          onClick={() => setActiveTab('RESOLVED')}
          className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all whitespace-nowrap flex items-center gap-1.5 ${
            activeTab === 'RESOLVED'
              ? 'bg-slate-700 text-white'
              : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
          }`}
        >
          <CheckCircle2 className="w-3 h-3 text-emerald-400" />
          Resolved ({markets.filter((m) => m.resolved).length})
        </button>
      </div>

      {/* Markets List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
        {filteredMarkets.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs">
            No markets found for this filter tab.
          </div>
        ) : (
          filteredMarkets.map((market) => {
            const isSelected = market.id === selectedMarketId;
            const pos = getPositionForMarket(market.id);
            const yesProb = Math.round(market.currentYesPrice * 100);
            const isShortLived = market.lifespan === 'SHORT_LIVED';

            return (
              <div
                key={market.id}
                onClick={() => onSelectMarket(market.id)}
                className={`p-2.5 rounded-xl border transition-all cursor-pointer relative overflow-hidden ${
                  isSelected
                    ? 'bg-slate-900 border-indigo-500/60 ring-1 ring-indigo-500/40 shadow-lg'
                    : market.resolved
                    ? 'bg-slate-900/30 border-slate-800/40 opacity-75'
                    : isShortLived
                    ? 'bg-slate-900/70 border-amber-500/30 hover:border-amber-500/60'
                    : 'bg-slate-900/60 border-slate-800/70 hover:bg-slate-900/90 hover:border-slate-700'
                }`}
              >
                {/* Short-lived indicator accent bar */}
                {isShortLived && !market.resolved && (
                  <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500" />
                )}

                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="font-mono text-[9px] font-bold px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        {market.ticker}
                      </span>
                      {isShortLived && (
                        <span className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 flex items-center gap-1">
                          <Timer className="w-2.5 h-2.5 animate-spin" />
                          Resolves on {market.expiryStreet || 'Street'}
                        </span>
                      )}
                      {market.lifespan === 'LONG_LIVED' && (
                        <span className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded bg-purple-500/15 border border-purple-500/30 text-purple-300">
                          Tournament Prop
                        </span>
                      )}
                    </div>
                    <div className="text-xs font-bold text-white truncate mt-1">
                      {market.name}
                    </div>
                    <div className="text-[10px] text-slate-400 line-clamp-1">
                      {market.description}
                    </div>
                  </div>

                  {/* Implied Probability Gauge or Resolved Outcome */}
                  <div className="text-right flex-shrink-0">
                    {market.resolved ? (
                      <div
                        className={`font-mono text-xs font-extrabold px-2 py-0.5 rounded border ${
                          market.outcome
                            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                            : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                        }`}
                      >
                        {market.outcome ? 'YES (1.00)' : 'NO (0.00)'}
                      </div>
                    ) : (
                      <>
                        <div className="font-mono text-xs font-extrabold text-indigo-400">
                          {yesProb}%
                        </div>
                        <div className="text-[9px] text-slate-400">Prob</div>
                      </>
                    )}
                  </div>
                </div>

                {/* Mini Probability Bar */}
                {!market.resolved && (
                  <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden mb-2">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isShortLived
                          ? 'bg-gradient-to-r from-amber-500 to-emerald-400'
                          : 'bg-gradient-to-r from-indigo-500 to-emerald-400'
                      }`}
                      style={{ width: `${yesProb}%` }}
                    />
                  </div>
                )}

                {/* Price Buttons & 1-Click Bet */}
                {!market.resolved ? (
                  <div className="flex items-center justify-between gap-2">
                    {/* Buy YES Button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onQuickTrade(market.id, 'BUY_YES', quickBetAmount);
                      }}
                      className="flex-1 py-1.5 px-2 rounded-lg bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700/80 text-emerald-300 font-mono text-[11px] font-bold flex items-center justify-between transition-all shadow-sm active:scale-95"
                    >
                      <div className="flex items-center gap-1">
                        <span className="text-[10px] text-emerald-400 font-sans">YES</span>
                        <span className="text-[9px] text-emerald-500/70 font-mono">${quickBetAmount}</span>
                      </div>
                      <span>${market.currentYesPrice.toFixed(2)}</span>
                    </button>

                    {/* Buy NO Button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onQuickTrade(market.id, 'BUY_NO', quickBetAmount);
                      }}
                      className="flex-1 py-1.5 px-2 rounded-lg bg-rose-950/80 hover:bg-rose-900 border border-rose-700/80 text-rose-300 font-mono text-[11px] font-bold flex items-center justify-between transition-all shadow-sm active:scale-95"
                    >
                      <div className="flex items-center gap-1">
                        <span className="text-[10px] text-rose-400 font-sans">NO</span>
                        <span className="text-[9px] text-rose-500/70 font-mono">${quickBetAmount}</span>
                      </div>
                      <span>${market.currentNoPrice.toFixed(2)}</span>
                    </button>
                  </div>
                ) : (
                  <div className="text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-800/60 flex items-center justify-between">
                    <span>{market.resolutionText || 'Settled at Showdown'}</span>
                    <span className="text-emerald-400">Completed</span>
                  </div>
                )}

                {/* Position Tag if user holds contracts */}
                {pos && (
                  <div className="mt-2 pt-1.5 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-300">
                    <span className="flex items-center gap-1 text-indigo-400 font-bold">
                      <Layers className="w-2.5 h-2.5" />
                      Your Pos: {pos.yesContracts > 0 ? `${pos.yesContracts} YES` : `${pos.noContracts} NO`}
                    </span>
                    <span
                      className={`font-bold ${
                        pos.realizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      PnL: {pos.realizedPnl >= 0 ? '+' : ''}${pos.realizedPnl}
                    </span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
