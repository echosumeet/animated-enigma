import React, { useState } from 'react';
import {
  Layers,
  Cpu,
  BrainCircuit,
  Bot,
  Compass,
  ArrowRight,
  ShieldCheck,
  Zap,
  Globe2,
  Database,
  Radio,
  FileCode2,
  Download,
  X,
  CheckCircle2,
  Terminal,
  Activity,
  Sparkles,
  ExternalLink,
} from 'lucide-react';

interface ShowcaseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ShowcaseModal: React.FC<ShowcaseModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'ARCHITECTURE' | 'FOUR_SKILLS' | 'AI_PIPELINE' | 'MARKDOWN'>('ARCHITECTURE');
  const [selectedSkill, setSelectedSkill] = useState<number>(0);

  if (!isOpen) return null;

  const fourSkills = [
    {
      id: 'building-deploying',
      title: '1. Building & Deploying AI Applications',
      subtitle: 'Google Gemini 3.7 Flash + Live Search Grounding',
      icon: BrainCircuit,
      color: 'emerald',
      bgGradient: 'from-emerald-950/60 to-slate-900',
      borderColor: 'border-emerald-500/40',
      textColor: 'text-emerald-400',
      bullets: [
        'Google Search Grounding tool (googleSearch: {}) bound to Gemini 3.7 Flash for zero-latency live score extraction from global cricket matches.',
        'Strict JSON schema formatting with runtime code-block sanitization and error fallback recovery.',
        'Preservation of Search Grounding chunks & source URLs for auditable transparency.',
        'Deterministic client-side fallback simulation to ensure zero gameplay disruption if network is offline.',
      ],
      codeSnippet: `// server.ts - Gemini 3.7 Flash Search Grounding
const response = await ai.models.generateContent({
  model: 'gemini-3.7-flash',
  contents: prompt,
  config: {
    tools: [{ googleSearch: {} }],
    temperature: 0.2,
  },
});`,
    },
    {
      id: 'software-fundamentals',
      title: '2. Software Engineering Fundamentals',
      subtitle: 'Deterministic Ledgers, Order Books & Express Bundling',
      icon: Database,
      color: 'indigo',
      bgGradient: 'from-indigo-950/60 to-slate-900',
      borderColor: 'border-indigo-500/40',
      textColor: 'text-indigo-400',
      bullets: [
        'Separation of Concerns: Poker state machine (Texas Hold\'em), continuous order book matching engine, and sports micro-settlement engine.',
        'Double-Entry Accounting: Immutable ledger tracking chips, liquid cash, contracts, and zero-sum synergy bounty distribution.',
        'Web Audio API: Pure mathematical audio synthesis (sound.ts) with zero external audio assets or load latency.',
        'Standalone CommonJS Server Build: esbuild compiling server.ts into dist/server.cjs for containerized Cloud Run execution.',
      ],
      codeSnippet: `// ledger.ts - Double Entry Financial Invariant
export class LedgerEngine {
  public static calculatePlayerEquity(
    player: Player,
    positions: Position[],
    markets: Market[]
  ): PlayerEquityData {
    const portfolioValue = positions.reduce((acc, pos) => {
      const m = markets.find(m => m.id === pos.marketId);
      return acc + (pos.shares * (m?.lastPrice || 0.5));
    }, 0);
    return {
      liquidCash: player.cashBalance,
      chipStack: player.chips,
      portfolioValue,
      totalEquity: player.cashBalance + player.chips + portfolioValue
    };
  }
}`,
    },
    {
      id: 'coding-agents',
      title: '3. Using Coding Agents',
      subtitle: 'Disciplined Context, Evals & Closed-Loop Verification',
      icon: Bot,
      color: 'purple',
      bgGradient: 'from-purple-950/60 to-slate-900',
      borderColor: 'border-purple-500/40',
      textColor: 'text-purple-400',
      bullets: [
        'Rigorous Read-Modify-Write workflow preventing stale context or hallucinated file structures.',
        'Continuous automated verification through lint_applet (tsc --noEmit) and compile_applet (Vite build) at every milestone.',
        'Modular architectural boundaries (cricket.ts, game.ts, gameManager.ts) avoiding token limit bottlenecks.',
        'Non-blocking background dev server restarts and reactive status tracking.',
      ],
      codeSnippet: `// Verification Command Loop
// 1. lint_applet -> verifies full TypeScript typing & zero runtime errors
// 2. compile_applet -> executes production Vite build
// 3. restart_dev_server -> applies hot backend changes cleanly`,
    },
    {
      id: 'shaping-build',
      title: '4. Shaping the Build',
      subtitle: 'Game Theory Synergy, Micro-Betting UX & Product Instinct',
      icon: Compass,
      color: 'amber',
      bgGradient: 'from-amber-950/60 to-slate-900',
      borderColor: 'border-amber-500/40',
      textColor: 'text-amber-400',
      bullets: [
        'Dual Synergy Bounty: Incentivizes simultaneous mastery of poker hand equity and market prediction contracts (+15% to +100% pool from losers).',
        '1-Click Micro-Betting: Rapid $25/$50/$100/$250 quick bet tickets tailored to tight 15-second street trading windows.',
        'Tournament Blind Escalation: Deep structure with escalating small/big blinds every 3 hands for escalating tournament drama.',
        'Live Sports & Poker Fusion: Real-time cricket score ticker integrated directly into the tournament HUD.',
      ],
      codeSnippet: `// gameManager.ts - Dual Synergy Bounty Extraction
if (isPokerWinner && userHasWinningMarketPosition) {
  const bountyAmount = Math.round(pot * (synergyMultiplierPct / 100));
  // Extract bonus directly from losing players' chip stacks
  losers.forEach(loser => {
    const contribution = Math.min(loser.chips, bountyAmount / losers.length);
    loser.chips -= contribution;
  });
  winner.chips += bountyAmount;
}`,
    },
  ];

  const handleDownloadMarkdown = () => {
    const element = document.createElement('a');
    const file = new Blob([
      `# Market Poker & Live Sports Arena: AI Engineering Showcase\n\nMapping to Andrew Ng's AI Engineering Skills Map (August 2026)\n\nAvailable in repository root at: /AI_ENGINEERING_SHOWCASE.md\nArtifact path: /home/workdir/artifacts/market-poker-ai-engineering-showcase.md`
    ], { type: 'text/markdown' });
    element.href = URL.createObjectURL(file);
    element.download = 'market-poker-ai-engineering-showcase.md';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-slate-950/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-slate-950 border border-slate-800 w-full max-w-5xl h-[88vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-indigo-500/20 to-emerald-500/20 border border-indigo-500/30 text-indigo-400">
              <Cpu className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base sm:text-lg font-bold text-white tracking-wide">
                  AI Engineering Architecture & Showcase
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-indigo-500/20 border border-indigo-500/30 text-indigo-300">
                  Andrew Ng Skills Map
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Tournament Texas Hold'em, Continuous Prediction Markets & Google-Grounded Live Cricket Scanner
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadMarkdown}
              className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-semibold transition-all border border-slate-700 cursor-pointer"
              title="Download Markdown Showcase"
            >
              <Download className="w-3.5 h-3.5 text-emerald-400" />
              <span>Download Showcase (.md)</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="px-5 pt-3 border-b border-slate-800/80 bg-slate-950 flex items-center gap-2 overflow-x-auto no-scrollbar">
          <button
            onClick={() => setActiveTab('ARCHITECTURE')}
            className={`pb-2.5 px-3 text-xs font-bold transition-all flex items-center gap-2 border-b-2 ${
              activeTab === 'ARCHITECTURE'
                ? 'border-indigo-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-4 h-4 text-indigo-400" />
            <span>Visual System Architecture</span>
          </button>
          <button
            onClick={() => setActiveTab('FOUR_SKILLS')}
            className={`pb-2.5 px-3 text-xs font-bold transition-all flex items-center gap-2 border-b-2 ${
              activeTab === 'FOUR_SKILLS'
                ? 'border-emerald-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <BrainCircuit className="w-4 h-4 text-emerald-400" />
            <span>Andrew Ng's 4 AI Skills</span>
          </button>
          <button
            onClick={() => setActiveTab('AI_PIPELINE')}
            className={`pb-2.5 px-3 text-xs font-bold transition-all flex items-center gap-2 border-b-2 ${
              activeTab === 'AI_PIPELINE'
                ? 'border-amber-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Radio className="w-4 h-4 text-amber-400" />
            <span>Live Grounding Pipeline</span>
          </button>
          <button
            onClick={() => setActiveTab('MARKDOWN')}
            className={`pb-2.5 px-3 text-xs font-bold transition-all flex items-center gap-2 border-b-2 ${
              activeTab === 'MARKDOWN'
                ? 'border-purple-500 text-white'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileCode2 className="w-4 h-4 text-purple-400" />
            <span>Showcase Document Spec</span>
          </button>
        </div>

        {/* Tab Content Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6 custom-scrollbar bg-slate-950/60">
          {/* TAB 1: VISUAL ARCHITECTURE */}
          {activeTab === 'ARCHITECTURE' && (
            <div className="space-y-6">
              {/* Architecture Blueprint Card */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Layers className="w-5 h-5 text-indigo-400" />
                    <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                      End-to-End System Architecture
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400">
                    Full-Stack Express + React 18 + Gemini 3.7
                  </span>
                </div>

                {/* Interactive Diagram Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Layer 1: Client Front-End */}
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-indigo-500/30 flex flex-col gap-3">
                    <div className="flex items-center gap-2 text-indigo-400 font-bold text-xs font-mono uppercase">
                      <Sparkles className="w-4 h-4" />
                      <span>1. Client Tier (React 18)</span>
                    </div>
                    <ul className="space-y-2 text-xs text-slate-300">
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Poker Table & HUD</strong>: 5-seat tournament engine, escalating blinds, pot tracking.</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Markets Panel</strong>: Continuous order books, 1-click preset trades ($25-$250).</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Live Sports Arena</strong>: Google Cricket score card, ball ticker & win probabilities.</span>
                      </li>
                    </ul>
                  </div>

                  {/* Layer 2: Deterministic Core & Server */}
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-emerald-500/30 flex flex-col gap-3">
                    <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs font-mono uppercase">
                      <Database className="w-4 h-4" />
                      <span>2. Backend & Ledger (Express)</span>
                    </div>
                    <ul className="space-y-2 text-xs text-slate-300">
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Double-Entry Ledger</strong>: Immutable transaction ledger with cash/chip segregation.</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Order Book Matching</strong>: Binary contract matching with dynamic bid/ask spreads.</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Synergy Extraction</strong>: Zero-sum bounty pool deducted directly from losing stacks.</span>
                      </li>
                    </ul>
                  </div>

                  {/* Layer 3: AI Grounding & External APIs */}
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-amber-500/30 flex flex-col gap-3">
                    <div className="flex items-center gap-2 text-amber-400 font-bold text-xs font-mono uppercase">
                      <Globe2 className="w-4 h-4" />
                      <span>3. AI Grounding Tier</span>
                    </div>
                    <ul className="space-y-2 text-xs text-slate-300">
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Gemini 3.7 Flash</strong>: Server-side SDK instance (@google/genai) with zero browser leaks.</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Google Search Tool</strong>: Real-time search grounding for live international cricket matches.</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                        <span><strong>Fallback Simulator</strong>: Deterministic next-ball generator for offline continuity.</span>
                      </li>
                    </ul>
                  </div>
                </div>

                {/* Flow Arrows Diagram */}
                <div className="mt-5 p-3 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-400 flex flex-col md:flex-row items-center justify-between gap-3 text-center">
                  <div className="flex items-center gap-1.5 text-indigo-300">
                    <span>Poker Street Action</span>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                  <div className="flex items-center gap-1.5 text-amber-300">
                    <span>Micro-Market Generation</span>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                  <div className="flex items-center gap-1.5 text-emerald-300">
                    <span>Google Search Grounding</span>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                  <div className="flex items-center gap-1.5 text-purple-300">
                    <span>Dual Synergy Settlement</span>
                  </div>
                </div>
              </div>

              {/* Key Architectural Metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="text-[10px] uppercase font-mono text-slate-400">Model Engine</div>
                  <div className="text-sm font-bold text-white mt-0.5 font-mono">Gemini 3.7 Flash</div>
                  <div className="text-[10px] text-emerald-400 font-mono">Google Search Grounded</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="text-[10px] uppercase font-mono text-slate-400">Financial Ledger</div>
                  <div className="text-sm font-bold text-white mt-0.5 font-mono">Double-Entry</div>
                  <div className="text-[10px] text-indigo-400 font-mono">Zero-Sum Invariant</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="text-[10px] uppercase font-mono text-slate-400">Tournament Escalation</div>
                  <div className="text-sm font-bold text-white mt-0.5 font-mono">5 Dynamic Levels</div>
                  <div className="text-[10px] text-amber-400 font-mono">3 Hands / Level</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="text-[10px] uppercase font-mono text-slate-400">Synergy Bounty</div>
                  <div className="text-sm font-bold text-white mt-0.5 font-mono">+15% to +100%</div>
                  <div className="text-[10px] text-purple-400 font-mono">From Losers Stack</div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: FOUR SKILLS BREAKDOWN */}
          {activeTab === 'FOUR_SKILLS' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                {fourSkills.map((skill, idx) => {
                  const Icon = skill.icon;
                  const isSelected = selectedSkill === idx;
                  return (
                    <button
                      key={skill.id}
                      onClick={() => setSelectedSkill(idx)}
                      className={`p-3 rounded-xl border text-left transition-all cursor-pointer ${
                        isSelected
                          ? `bg-slate-900 ${skill.borderColor} shadow-lg shadow-${skill.color}-500/10`
                          : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <Icon className={`w-5 h-5 ${skill.textColor} mb-2`} />
                      <div className="text-xs font-bold text-white line-clamp-1">{skill.title}</div>
                      <div className="text-[10px] text-slate-400 mt-1 line-clamp-2">{skill.subtitle}</div>
                    </button>
                  );
                })}
              </div>

              {/* Selected Skill Deep Dive */}
              {fourSkills[selectedSkill] && (
                <div className={`p-5 rounded-2xl bg-gradient-to-br ${fourSkills[selectedSkill].bgGradient} border ${fourSkills[selectedSkill].borderColor} space-y-4`}>
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <span>{fourSkills[selectedSkill].title}</span>
                    </h3>
                    <p className="text-xs text-slate-300 mt-1 font-mono">
                      {fourSkills[selectedSkill].subtitle}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* Implementation Bullets */}
                    <div className="space-y-2">
                      <div className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
                        Key Implementation Highlights
                      </div>
                      <ul className="space-y-2">
                        {fourSkills[selectedSkill].bullets.map((b, bIdx) => (
                          <li key={bIdx} className="text-xs text-slate-300 flex items-start gap-2 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80">
                            <CheckCircle2 className={`w-4 h-4 ${fourSkills[selectedSkill].textColor} mt-0.5 flex-shrink-0`} />
                            <span>{b}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Code Snippet */}
                    <div>
                      <div className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono mb-2 flex items-center justify-between">
                        <span>Code Reference</span>
                        <Terminal className="w-3.5 h-3.5 text-slate-400" />
                      </div>
                      <pre className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-emerald-300 overflow-x-auto custom-scrollbar h-[180px]">
                        <code>{fourSkills[selectedSkill].codeSnippet}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: LIVE AI PIPELINE */}
          {activeTab === 'AI_PIPELINE' && (
            <div className="space-y-4">
              <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
                    <h3 className="text-xs font-bold text-white font-mono uppercase">
                      Google Cricket Search Grounding Inspector
                    </h3>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    Endpoint: POST /api/cricket/scan
                  </span>
                </div>

                <p className="text-xs text-slate-300">
                  Inspect the live inference request payload sent to <strong>Gemini 3.7 Flash</strong> with Google Search Grounding to extract international cricket match scores and generate prediction propositions.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px]">
                    <div className="text-slate-400 text-[10px] uppercase font-bold mb-1">
                      Prompt Schema & Config
                    </div>
                    <div className="text-slate-300 space-y-1">
                      <div><strong className="text-indigo-400">Model:</strong> gemini-3.7-flash</div>
                      <div><strong className="text-indigo-400">Tool:</strong> googleSearch: &#123;&#125;</div>
                      <div><strong className="text-indigo-400">Temperature:</strong> 0.2 (Low for strict JSON output)</div>
                      <div><strong className="text-indigo-400">Target Queries:</strong> International ODI, T20I, Test Series</div>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px]">
                    <div className="text-slate-400 text-[10px] uppercase font-bold mb-1">
                      Output Extraction & Grounding
                    </div>
                    <div className="text-slate-300 space-y-1">
                      <div><strong className="text-emerald-400">Scorecard:</strong> Team runs, wickets, overs, CRR/RRR</div>
                      <div><strong className="text-emerald-400">Batsmen/Bowler:</strong> Runs, balls, strike rates, maidens</div>
                      <div><strong className="text-emerald-400">Grounding Chunks:</strong> Source links & articles stored</div>
                      <div><strong className="text-emerald-400">Fallback:</strong> Local Ball-by-Ball Poisson Simulator</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: MARKDOWN SPEC */}
          {activeTab === 'MARKDOWN' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-slate-400">
                  File: /AI_ENGINEERING_SHOWCASE.md & /home/workdir/artifacts/market-poker-ai-engineering-showcase.md
                </span>
                <button
                  onClick={handleDownloadMarkdown}
                  className="flex items-center gap-1 text-xs font-mono text-emerald-400 hover:text-emerald-300 hover:underline"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Markdown</span>
                </button>
              </div>

              <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300 overflow-y-auto max-h-[400px] custom-scrollbar whitespace-pre-wrap">
                {`# Market Poker & Live Sports Arena: AI Engineering Showcase

Mapping to Andrew Ng's AI Engineering Skills Map (August 2026)

1. Building and Deploying AI Applications
- Gemini 3.7 Flash with Google Search Grounding for real-time international cricket match scores.
- Strict JSON Schema extraction and prompt steering.
- Deterministic simulation fallback loop.

2. Software Engineering Fundamentals
- Zero-sum double-entry ledger engine.
- Continuous prediction order book matching engine ($0.01 - $0.99).
- Clean separation between poker state machine, market pricing, and sports micro-settlement.
- esbuild CommonJS bundling for Cloud Run server runtime.

3. Using Coding Agents
- Read-Modify-Write verification workflows.
- Continuous linting (tsc --noEmit) and compilation checks.
- Clean modular file organization.

4. Shaping the Build
- Dual Victory Synergy Bounty (+15% to +100% pool extracted from losing players).
- 1-Click micro-betting presets ($25, $50, $100, $250).
- Escalating tournament blind levels.`}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
