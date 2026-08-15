import React, { useState } from 'react';
import { Market, OrderSide, OrderType, Player, Position } from '../types/game';
import { getOrderBookDepth } from '../engine/market';
import { PriceChart } from './PriceChart';
import { X, TrendingUp, DollarSign, Layers, ArrowRight, ShieldCheck } from 'lucide-react';

interface OrderBookModalProps {
  market: Market;
  userPlayer: Player;
  position?: Position;
  initialSide?: OrderSide;
  onClose: () => void;
  onPlaceOrder: (
    marketId: string,
    side: OrderSide,
    type: OrderType,
    price: number,
    quantity: number
  ) => void;
}

export const OrderBookModal: React.FC<OrderBookModalProps> = ({
  market,
  userPlayer,
  position,
  initialSide = 'BUY_YES',
  onClose,
  onPlaceOrder,
}) => {
  const [side, setSide] = useState<OrderSide>(initialSide);
  const [orderType, setOrderType] = useState<OrderType>('MARKET');
  const [quantity, setQuantity] = useState<number>(50);
  const [limitPrice, setLimitPrice] = useState<number>(
    side.includes('YES') ? market.currentYesPrice : market.currentNoPrice
  );

  const depth = getOrderBookDepth(market);
  const spread = Number(Math.max(0.01, market.bestAsk - market.bestBid).toFixed(2));

  const effectivePrice = orderType === 'MARKET'
    ? (side.includes('YES') ? market.currentYesPrice : market.currentNoPrice)
    : limitPrice;

  const totalCost = Number((quantity * effectivePrice).toFixed(2));
  const maxPayout = quantity * 1.0;
  const maxProfit = Number((maxPayout - totalCost).toFixed(2));

  const handleExecute = () => {
    if (quantity <= 0 || totalCost > userPlayer.cashBalance) return;
    onPlaceOrder(market.id, side, orderType, effectivePrice, quantity);
    onClose();
  };

  const presetQuantities = [25, 50, 100, 250, 500];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
      <div className="bg-slate-950 border border-slate-800 rounded-3xl w-full max-w-4xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 font-mono font-bold">
              {market.ticker.slice(0, 3)}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold">
                  {market.ticker}
                </span>
                <h2 className="text-base sm:text-lg font-bold text-white">
                  {market.name}
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {market.description}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body: Grid of Chart & Order Book vs Order Ticket */}
        <div className="flex-1 overflow-y-auto p-5 grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left Column: Price Chart & Depth Ladder (7 Cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            {/* Price Timeline Chart */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 shadow-inner">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
                  Price Discovery Timeline
                </span>
                <span className="font-mono text-xs font-bold text-emerald-400">
                  Spot: ${(market.currentYesPrice).toFixed(2)} ({Math.round(market.currentYesPrice * 100)}%)
                </span>
              </div>
              <PriceChart priceHistory={market.priceHistory} height={150} />
            </div>

            {/* Depth Ladder */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4">
              <div className="flex items-center justify-between mb-3 text-xs font-bold uppercase tracking-wider text-slate-400">
                <span>Order Book Depth</span>
                <span className="font-mono text-[11px] text-amber-400">
                  Spread: ${spread.toFixed(2)}
                </span>
              </div>

              {/* Asks (Sell Orders) */}
              <div className="space-y-1 mb-2">
                <div className="text-[10px] text-rose-400 font-semibold uppercase">
                  Asks (Sell Orders)
                </div>
                {depth.asks.slice(0, 3).map((a, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs font-mono bg-rose-950/20 px-2 py-1 rounded border border-rose-900/30"
                  >
                    <span className="text-rose-400 font-bold">${a.price.toFixed(2)}</span>
                    <span className="text-slate-400">{a.quantity} contracts</span>
                  </div>
                ))}
              </div>

              {/* Current Midpoint */}
              <div className="py-1.5 my-1 bg-slate-950 px-3 rounded border border-slate-800 flex items-center justify-between font-mono text-xs text-indigo-400 font-extrabold">
                <span>MIDPOINT SPOT</span>
                <span>${market.currentYesPrice.toFixed(2)}</span>
              </div>

              {/* Bids (Buy Orders) */}
              <div className="space-y-1 mt-2">
                <div className="text-[10px] text-emerald-400 font-semibold uppercase">
                  Bids (Buy Orders)
                </div>
                {depth.bids.slice(0, 3).map((b, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs font-mono bg-emerald-950/20 px-2 py-1 rounded border border-emerald-900/30"
                  >
                    <span className="text-emerald-400 font-bold">${b.price.toFixed(2)}</span>
                    <span className="text-slate-400">{b.quantity} contracts</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Order Execution Ticket (5 Cols) */}
          <div className="lg:col-span-5 bg-slate-900/90 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-xl">
            <div>
              {/* Buy YES vs Buy NO Toggle */}
              <div className="grid grid-cols-2 gap-2 p-1 bg-slate-950 rounded-xl border border-slate-800 mb-4">
                <button
                  onClick={() => {
                    setSide('BUY_YES');
                    setLimitPrice(market.currentYesPrice);
                  }}
                  className={`py-2 rounded-lg font-bold text-xs transition-all ${
                    side === 'BUY_YES'
                      ? 'bg-emerald-600 text-slate-950 shadow-md'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  BUY YES (${market.currentYesPrice.toFixed(2)})
                </button>
                <button
                  onClick={() => {
                    setSide('BUY_NO');
                    setLimitPrice(market.currentNoPrice);
                  }}
                  className={`py-2 rounded-lg font-bold text-xs transition-all ${
                    side === 'BUY_NO'
                      ? 'bg-rose-600 text-white shadow-md'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  BUY NO (${market.currentNoPrice.toFixed(2)})
                </button>
              </div>

              {/* Order Type Toggle: Market vs Limit */}
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs text-slate-400 font-medium">Order Type</span>
                <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px] font-semibold">
                  <button
                    onClick={() => setOrderType('MARKET')}
                    className={`px-3 py-1 rounded ${
                      orderType === 'MARKET' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                    }`}
                  >
                    Market
                  </button>
                  <button
                    onClick={() => setOrderType('LIMIT')}
                    className={`px-3 py-1 rounded ${
                      orderType === 'LIMIT' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                    }`}
                  >
                    Limit
                  </button>
                </div>
              </div>

              {/* Limit Price Input (if Limit order selected) */}
              {orderType === 'LIMIT' && (
                <div className="mb-4">
                  <label className="text-xs text-slate-400 font-medium mb-1 block">
                    Limit Price ($0.01 - $0.99)
                  </label>
                  <input
                    type="number"
                    min={0.01}
                    max={0.99}
                    step={0.01}
                    value={limitPrice}
                    onChange={(e) => setLimitPrice(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm font-mono text-white focus:outline-none focus:border-indigo-500"
                  />
                </div>
              )}

              {/* Quantity Input */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-slate-400 font-medium">
                    Contracts Quantity
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">
                    Avail Cash: ${userPlayer.cashBalance.toLocaleString()}
                  </span>
                </div>
                <input
                  type="number"
                  min={1}
                  max={10000}
                  value={quantity}
                  onChange={(e) => setQuantity(Math.max(1, Number(e.target.value)))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm font-mono text-white focus:outline-none focus:border-indigo-500"
                />

                {/* Quick Presets */}
                <div className="flex gap-1.5 mt-2">
                  {presetQuantities.map((q) => (
                    <button
                      key={q}
                      onClick={() => setQuantity(q)}
                      className="flex-1 py-1 rounded bg-slate-950 border border-slate-800 text-[10px] font-mono text-slate-400 hover:text-white hover:bg-slate-800"
                    >
                      {q}
                    </button>
                  ))}
                  <button
                    onClick={() => {
                      const maxQ = Math.floor(userPlayer.cashBalance / effectivePrice);
                      setQuantity(Math.max(1, maxQ));
                    }}
                    className="flex-1 py-1 rounded bg-indigo-950 border border-indigo-800 text-[10px] font-mono text-indigo-300 hover:text-white"
                  >
                    Max
                  </button>
                </div>
              </div>

              {/* Order Cost & Payout Breakdown */}
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5 text-xs font-mono mb-4">
                <div className="flex justify-between text-slate-400">
                  <span>Execution Cost:</span>
                  <span className="text-white font-bold">${totalCost.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Payout if YES:</span>
                  <span className="text-emerald-400 font-bold">${maxPayout.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Max Profit:</span>
                  <span className="text-emerald-400 font-bold">+${maxProfit.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Max Loss:</span>
                  <span className="text-rose-400 font-bold">-${totalCost.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Submit Execution Button */}
            <button
              onClick={handleExecute}
              disabled={totalCost > userPlayer.cashBalance || totalCost <= 0}
              className={`w-full py-3 rounded-xl font-bold text-xs sm:text-sm flex items-center justify-center gap-2 shadow-lg transition-all cursor-pointer ${
                side === 'BUY_YES'
                  ? 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-emerald-500/20'
                  : 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/20'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {totalCost > userPlayer.cashBalance ? (
                'Insufficient Cash Balance'
              ) : (
                <>
                  <span>
                    Execute {orderType} {side} ({quantity} @ ${effectivePrice.toFixed(2)})
                  </span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
