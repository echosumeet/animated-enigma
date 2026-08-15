import React, { useState } from 'react';
import { Landmark, DollarSign, Percent, ShieldAlert, ArrowDownLeft, ArrowUpRight, X, Sparkles, CheckCircle2 } from 'lucide-react';
import { Player } from '../types/game';

interface BorrowLoanModalProps {
  userPlayer: Player;
  isOpen: boolean;
  onClose: () => void;
  onBorrow: (amount: number) => void;
  onRepay: (amount: number) => void;
}

export const BorrowLoanModal: React.FC<BorrowLoanModalProps> = ({
  userPlayer,
  isOpen,
  onClose,
  onBorrow,
  onRepay,
}) => {
  const [borrowAmount, setBorrowAmount] = useState<number>(2500);
  const [repayAmount, setRepayAmount] = useState<number>(userPlayer.loanBalance || 0);
  const [activeTab, setActiveTab] = useState<'BORROW' | 'REPAY'>('BORROW');

  if (!isOpen) return null;

  const currentLoan = userPlayer.loanBalance || 0;
  const currentCash = userPlayer.cashBalance || 0;
  const interestPerWin = Math.max(1, Math.round(currentLoan * 0.04));
  const totalInterestPaid = userPlayer.totalInterestPaid || 0;

  const quickBorrowPresets = [1000, 2500, 5000, 10000];

  const handleExecuteBorrow = () => {
    if (borrowAmount > 0) {
      onBorrow(borrowAmount);
      onClose();
    }
  };

  const handleExecuteRepay = () => {
    if (repayAmount > 0) {
      onRepay(repayAmount);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
      <div className="bg-slate-950 border border-slate-800 rounded-3xl w-full max-w-lg shadow-2xl overflow-hidden flex flex-col my-auto max-h-[95vh] relative">
        {/* Header */}
        <div className="p-5 sm:p-6 border-b border-slate-800 bg-gradient-to-b from-amber-950/30 via-slate-950 to-slate-950 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center justify-center shadow-lg">
              <Landmark className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg sm:text-xl font-black text-white tracking-wide">
                  Market Liquidity Loan
                </h2>
                <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[10px] font-mono font-bold">
                  4% Fixed Interest
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Borrow emergency funds directly from the market protocol
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Selector */}
        <div className="px-6 pt-4 flex gap-2">
          <button
            onClick={() => setActiveTab('BORROW')}
            className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
              activeTab === 'BORROW'
                ? 'bg-amber-500 text-slate-950 font-black shadow-md shadow-amber-500/20'
                : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
          >
            <ArrowDownLeft className="w-3.5 h-3.5" />
            <span>Borrow Liquidity</span>
          </button>
          <button
            onClick={() => {
              setActiveTab('REPAY');
              setRepayAmount(Math.min(currentCash, currentLoan));
            }}
            disabled={currentLoan <= 0}
            className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
              activeTab === 'REPAY'
                ? 'bg-emerald-500 text-slate-950 font-black shadow-md shadow-emerald-500/20'
                : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
          >
            <ArrowUpRight className="w-3.5 h-3.5" />
            <span>Repay Principal ({currentLoan > 0 ? `$${currentLoan.toLocaleString()}` : '$0'})</span>
          </button>
        </div>

        {/* Status Metrics Bar */}
        <div className="p-6 space-y-4">
          <div className="grid grid-cols-3 gap-2.5">
            <div className="p-2.5 rounded-2xl bg-slate-900/90 border border-slate-800 text-center">
              <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
                Liquid Cash
              </span>
              <span className="font-mono text-sm sm:text-base font-extrabold text-white">
                ${currentCash.toLocaleString()}
              </span>
            </div>

            <div className="p-2.5 rounded-2xl bg-amber-950/40 border border-amber-500/30 text-center">
              <span className="text-[10px] uppercase font-bold text-amber-400 block mb-1">
                Active Debt
              </span>
              <span className="font-mono text-sm sm:text-base font-extrabold text-amber-300">
                ${currentLoan.toLocaleString()}
              </span>
            </div>

            <div className="p-2.5 rounded-2xl bg-slate-900/90 border border-slate-800 text-center">
              <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
                Auto-Pay / Win
              </span>
              <span className="font-mono text-sm sm:text-base font-extrabold text-rose-400">
                {currentLoan > 0 ? `-$${interestPerWin}` : '$0'} (4%)
              </span>
            </div>
          </div>

          {/* BORROW TAB */}
          {activeTab === 'BORROW' && (
            <div className="space-y-4">
              {/* Presets */}
              <div>
                <label className="text-xs font-bold text-slate-300 block mb-2">
                  Select Quick Borrow Amount:
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {quickBorrowPresets.map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => setBorrowAmount(preset)}
                      className={`py-2 rounded-xl text-xs font-mono font-bold transition-all border cursor-pointer ${
                        borrowAmount === preset
                          ? 'bg-amber-500/20 border-amber-400 text-amber-300 ring-2 ring-amber-400/30'
                          : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800 hover:text-white'
                      }`}
                    >
                      ${preset.toLocaleString()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Slider & Custom Input */}
              <div className="p-3.5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-medium">Custom Amount:</span>
                  <div className="flex items-center gap-1">
                    <span className="font-mono text-sm font-black text-amber-400">$</span>
                    <input
                      type="number"
                      min={500}
                      max={50000}
                      step={250}
                      value={borrowAmount}
                      onChange={(e) => setBorrowAmount(Math.max(0, Number(e.target.value)))}
                      className="w-28 bg-slate-950 border border-slate-700 rounded-lg px-2 py-1 font-mono text-sm font-bold text-white text-right focus:outline-none focus:border-amber-400"
                    />
                  </div>
                </div>
                <input
                  type="range"
                  min={500}
                  max={25000}
                  step={500}
                  value={borrowAmount}
                  onChange={(e) => setBorrowAmount(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>

              {/* Terms Callout */}
              <div className="p-3.5 rounded-2xl bg-indigo-950/30 border border-indigo-500/20 space-y-2 text-xs text-slate-300">
                <div className="flex items-center gap-1.5 font-bold text-indigo-300">
                  <Percent className="w-4 h-4 text-amber-400" />
                  <span>How the 4% Auto-Pay Rule Works:</span>
                </div>
                <ul className="space-y-1 text-[11px] text-slate-400 list-disc list-inside">
                  <li>
                    When you win a poker pot or prediction market payout, <strong>4% interest (${Math.round((currentLoan + borrowAmount) * 0.04)})</strong> is automatically deducted from the winning cash.
                  </li>
                  <li>
                    If you lose a hand, <strong>no interest is deducted</strong>.
                  </li>
                  <li>
                    You can voluntarily repay the loan principal anytime from this menu to stop future interest deductions.
                  </li>
                </ul>
              </div>

              {/* Action Button */}
              <button
                onClick={handleExecuteBorrow}
                className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-amber-500 via-amber-400 to-amber-500 hover:from-amber-400 hover:to-amber-300 text-slate-950 font-black text-sm flex items-center justify-center gap-2 shadow-xl shadow-amber-500/20 transition-all cursor-pointer hover:scale-[1.02] active:scale-98"
              >
                <Landmark className="w-4 h-4" />
                <span>Confirm Borrow +${borrowAmount.toLocaleString()} @ 4% Interest</span>
              </button>
            </div>
          )}

          {/* REPAY TAB */}
          {activeTab === 'REPAY' && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-medium">Repay Principal Amount:</span>
                  <div className="flex items-center gap-1">
                    <span className="font-mono text-sm font-black text-emerald-400">$</span>
                    <input
                      type="number"
                      min={1}
                      max={Math.min(currentCash, currentLoan)}
                      value={repayAmount}
                      onChange={(e) => setRepayAmount(Math.max(0, Number(e.target.value)))}
                      className="w-28 bg-slate-950 border border-slate-700 rounded-lg px-2 py-1 font-mono text-sm font-bold text-white text-right focus:outline-none focus:border-emerald-400"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setRepayAmount(Math.min(currentCash, Math.round(currentLoan / 2)))}
                    className="flex-1 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-300 transition-colors"
                  >
                    50% Repay
                  </button>
                  <button
                    type="button"
                    onClick={() => setRepayAmount(Math.min(currentCash, currentLoan))}
                    className="flex-1 py-1.5 rounded-lg bg-emerald-950 border border-emerald-800 hover:bg-emerald-900 text-xs font-bold text-emerald-300 transition-colors"
                  >
                    Repay All (${Math.min(currentCash, currentLoan).toLocaleString()})
                  </button>
                </div>
              </div>

              {totalInterestPaid > 0 && (
                <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400">Total 4% Interest Paid So Far:</span>
                  <span className="text-amber-400 font-bold">${totalInterestPaid.toLocaleString()}</span>
                </div>
              )}

              {/* Action Button */}
              <button
                onClick={handleExecuteRepay}
                disabled={repayAmount <= 0 || currentCash < repayAmount}
                className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-black text-sm flex items-center justify-center gap-2 shadow-xl shadow-emerald-500/20 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed hover:scale-[1.02] active:scale-98"
              >
                <CheckCircle2 className="w-4 h-4" />
                <span>Repay ${repayAmount.toLocaleString()} Principal</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
