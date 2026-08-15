import React, { useState } from 'react';
import { Layers, X, Sparkles, Trophy, Shield, Zap, Flame, Check } from 'lucide-react';

interface NewTableModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSpinUp: (
    tableName: string,
    theme: 'emerald' | 'sapphire' | 'amber' | 'ruby',
    startingLevel: number,
    botPreset: 'default' | 'high_roller'
  ) => void;
}

export const NewTableModal: React.FC<NewTableModalProps> = ({ isOpen, onClose, onSpinUp }) => {
  const [tableName, setTableName] = useState<string>('Table 2: High Roller Alpha');
  const [theme, setTheme] = useState<'emerald' | 'sapphire' | 'amber' | 'ruby'>('sapphire');
  const [startingLevel, setStartingLevel] = useState<number>(2);
  const [botPreset, setBotPreset] = useState<'default' | 'high_roller'>('high_roller');

  if (!isOpen) return null;

  const presets = [
    {
      id: 'high_roller',
      name: 'High Roller Alpha',
      theme: 'sapphire' as const,
      level: 2,
      preset: 'high_roller' as const,
      description: 'Viktor (Deep Quant), Elena (The Apex Shark), Jax (Momentum Scalper), Bruno (The Whale). Escalated blinds & +30% Dual Synergy.',
    },
    {
      id: 'apex_arena',
      name: 'Table 2: Apex Diamond',
      theme: 'ruby' as const,
      level: 3,
      preset: 'high_roller' as const,
      description: 'Deep stack pro tournament table with aggressive bots, rapid street questions, and +50% Dual Synergy bounty.',
    },
    {
      id: 'turbo_classic',
      name: 'Table 2: Turbo Scalpers',
      theme: 'amber' as const,
      level: 1,
      preset: 'default' as const,
      description: 'Fast-paced standard tournament with Alice, Marcus, Vance, and Rex.',
    },
  ];

  const handleSelectPreset = (p: (typeof presets)[0]) => {
    setTableName(p.name);
    setTheme(p.theme);
    setStartingLevel(p.level);
    setBotPreset(p.preset);
  };

  const handleConfirm = () => {
    onSpinUp(tableName, theme, startingLevel, botPreset);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-slate-950 border border-slate-800 rounded-3xl max-w-lg w-full p-6 shadow-2xl relative overflow-hidden flex flex-col gap-5 animate-in fade-in zoom-in-95 duration-200">
        {/* Glow accent */}
        <div className="absolute -top-24 -right-24 w-60 h-60 bg-indigo-500/20 rounded-full blur-3xl pointer-events-none" />

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-extrabold text-lg text-white">Spin Up 2nd Poker Table</h3>
              <p className="text-xs text-slate-400">Play two independent tables simultaneously with dual view</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Quick Presets */}
        <div className="flex flex-col gap-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Quick Table Presets
          </label>
          <div className="grid grid-cols-1 gap-2">
            {presets.map((p) => {
              const isSelected = tableName === p.name;
              return (
                <button
                  key={p.id}
                  onClick={() => handleSelectPreset(p)}
                  className={`p-3 rounded-2xl border text-left transition-all flex items-start justify-between gap-3 ${
                    isSelected
                      ? 'bg-indigo-950/70 border-indigo-500 ring-1 ring-indigo-500/40 shadow-lg'
                      : 'bg-slate-900/60 border-slate-800 hover:bg-slate-900 hover:border-slate-700'
                  }`}
                >
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-xs text-white">{p.name}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300">
                        Level {p.level}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">{p.description}</p>
                  </div>
                  {isSelected && <Check className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Customization Details */}
        <div className="grid grid-cols-2 gap-3">
          {/* Theme Selector */}
          <div>
            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5 block">
              Felt Theme
            </label>
            <div className="grid grid-cols-4 gap-1.5">
              {(
                [
                  { id: 'sapphire', label: 'Blue', color: 'bg-blue-600' },
                  { id: 'emerald', label: 'Green', color: 'bg-emerald-600' },
                  { id: 'ruby', label: 'Ruby', color: 'bg-rose-600' },
                  { id: 'amber', label: 'Gold', color: 'bg-amber-600' },
                ] as const
              ).map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTheme(t.id)}
                  className={`py-1.5 rounded-xl border text-[10px] font-bold flex flex-col items-center gap-1 transition-all ${
                    theme === t.id
                      ? 'bg-slate-800 border-white text-white'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  <span className={`w-3 h-3 rounded-full ${t.color}`} />
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* Starting Level */}
          <div>
            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5 block">
              Blinds & Bounty Level
            </label>
            <select
              value={startingLevel}
              onChange={(e) => setStartingLevel(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2 text-xs font-mono text-white focus:outline-none focus:border-indigo-500"
            >
              <option value={1}>Level 1: 50/100 (+15% Synergy)</option>
              <option value={2}>Level 2: 100/200 (+30% Synergy)</option>
              <option value={3}>Level 3: 200/400 (+50% Synergy)</option>
              <option value={4}>Level 4: 400/800 (+75% Synergy)</option>
            </select>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-slate-800/80">
          <button
            onClick={onClose}
            className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-bold text-xs transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 via-indigo-600 to-amber-500 hover:from-indigo-400 hover:to-amber-400 text-white font-extrabold text-xs flex items-center gap-2 shadow-lg shadow-indigo-500/25 transition-all cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            Spin Up Table 2
          </button>
        </div>
      </div>
    </div>
  );
};
