import React, { useState, useEffect } from 'react';
import {
  LiveCricketMatch,
  CricketSportsMarket,
  UserSportsBet,
} from '../types/cricket';
import { CricketEngine } from '../engine/cricketEngine';
import { sound } from '../engine/sound';
import confetti from 'canvas-confetti';
import {
  Zap,
  Flame,
  TrendingUp,
  RefreshCw,
  Globe2,
  Trophy,
  Activity,
  CheckCircle2,
  XCircle,
  Play,
  Layers,
  ArrowUpRight,
  ShieldCheck,
  Radio,
  ExternalLink,
  ChevronRight,
} from 'lucide-react';

interface LiveCricketPanelProps {
  userCash: number;
  onPlaceSportsBet: (bet: UserSportsBet) => void;
  onSettleSportsPayout: (payout: number, description: string) => void;
  userSportsBets: UserSportsBet[];
}

export const LiveCricketPanel: React.FC<LiveCricketPanelProps> = ({
  userCash,
  onPlaceSportsBet,
  onSettleSportsPayout,
  userSportsBets,
}) => {
  const [matches, setMatches] = useState<LiveCricketMatch[]>(() =>
    CricketEngine.getInitialMatches()
  );
  const [selectedMatchId, setSelectedMatchId] = useState<string>('cric-ind-aus-2026');
  const [activeTab, setActiveTab] = useState<'MARKETS' | 'MY_BETS' | 'SCORECARD'>('MARKETS');
  const [quickBetAmount, setQuickBetAmount] = useState<number>(50);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [lastScannedTime, setLastScannedTime] = useState<string>('Just now');
  const [groundingSources, setGroundingSources] = useState<{ title: string; url: string }[]>([]);
  const [recentBallNotice, setRecentBallNotice] = useState<string | null>(null);

  const currentMatch =
    matches.find((m) => m.id === selectedMatchId) || matches[0];

  // Google Cricket Scores Live Scanner via Server-Side Gemini API
  const handleScanGoogleScores = async () => {
    setIsScanning(true);
    try {
      sound.playQuickBet();
      const res = await fetch('/api/cricket/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();

      if (data.sources && Array.isArray(data.sources)) {
        setGroundingSources(data.sources);
      }

      if (data.matches && Array.isArray(data.matches) && data.matches.length > 0) {
        // Merge scanned matches with local state
        setMatches(data.matches);
      } else {
        // Simulate minor realistic live ball update if no new match payload
        const simResult = CricketEngine.simulateNextBall(currentMatch, userSportsBets);
        setMatches((prev) =>
          prev.map((m) => (m.id === currentMatch.id ? simResult.updatedMatch : m))
        );
        if (simResult.settledPnl > 0) {
          onSettleSportsPayout(simResult.settledPnl, `Sports Bet Win: ${simResult.ballEventText}`);
          confetti({ particleCount: 60, spread: 60 });
        }
        setRecentBallNotice(simResult.ballEventText);
      }

      setLastScannedTime(new Date().toLocaleTimeString());
    } catch (err) {
      console.warn('Scan request error, fallback to local match engine:', err);
      const simResult = CricketEngine.simulateNextBall(currentMatch, userSportsBets);
      setMatches((prev) =>
        prev.map((m) => (m.id === currentMatch.id ? simResult.updatedMatch : m))
      );
      if (simResult.settledPnl > 0) {
        onSettleSportsPayout(simResult.settledPnl, `Sports Bet Win: ${simResult.ballEventText}`);
      }
    } finally {
      setIsScanning(false);
    }
  };

  // Simulate Next Ball Interaction
  const handleSimulateBall = () => {
    sound.playCardDeal();
    const simResult = CricketEngine.simulateNextBall(currentMatch, userSportsBets);
    setMatches((prev) =>
      prev.map((m) => (m.id === currentMatch.id ? simResult.updatedMatch : m))
    );
    setRecentBallNotice(simResult.ballEventText);

    if (simResult.settledPnl > 0) {
      onSettleSportsPayout(simResult.settledPnl, `Cricket Bet Won: ${simResult.ballEventText}`);
      confetti({ particleCount: 70, spread: 70 });
      sound.playWin();
    }
  };

  // Place Quick Sports Bet
  const handleBet = (market: CricketSportsMarket, side: 'BUY_YES' | 'BUY_NO') => {
    if (userCash < quickBetAmount) {
      sound.playFold();
      return;
    }

    const price = side === 'BUY_YES' ? market.yesPrice : market.noPrice;
    const odds = side === 'BUY_YES' ? market.yesOddsMultiplier : market.noOddsMultiplier;
    const potentialPayout = Math.round(quickBetAmount * odds);
    const shares = Math.floor(quickBetAmount / Math.max(0.01, price));

    const newBet: UserSportsBet = {
      id: `cric-bet-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      matchId: currentMatch.id,
      matchTitle: currentMatch.matchTitle,
      marketId: market.id,
      marketTitle: market.title,
      side,
      stake: quickBetAmount,
      price,
      shares,
      potentialPayout,
      oddsMultiplier: odds,
      status: 'OPEN',
      pnl: 0,
      timestamp: Date.now(),
    };

    sound.playChipBet();
    onPlaceSportsBet(newBet);
  };

  const openBets = userSportsBets.filter((b) => b.status === 'OPEN');
  const settledBets = userSportsBets.filter((b) => b.status !== 'OPEN');
  const totalBetsPnl = userSportsBets.reduce((acc, b) => acc + b.pnl, 0);

  return (
    <div className="bg-slate-950/95 border border-slate-800/90 rounded-2xl p-3 sm:p-4 flex flex-col h-full shadow-2xl backdrop-blur-md">
      {/* Top Header & Live Google Search Grounding Scanner */}
      <div className="pb-3 border-b border-slate-800/80 mb-2.5">
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/25 text-emerald-400">
              <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <h2 className="text-xs sm:text-sm font-bold text-white tracking-wide">
                  Live Cricket & Sports
                </h2>
                <span className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 flex items-center gap-1">
                  <Globe2 className="w-2.5 h-2.5" />
                  Google Grounded
                </span>
              </div>
              <p className="text-[10px] text-slate-400">
                Scanned Google Cricket Scores & Live Micro-Betting
              </p>
            </div>
          </div>

          {/* Google Cricket Live Scan Trigger */}
          <button
            onClick={handleScanGoogleScores}
            disabled={isScanning}
            className="px-2.5 py-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-mono text-[11px] font-bold flex items-center gap-1.5 shadow-md shadow-emerald-500/20 active:scale-95 transition-all cursor-pointer disabled:opacity-50"
          >
            <RefreshCw className={`w-3 h-3 ${isScanning ? 'animate-spin' : ''}`} />
            <span>{isScanning ? 'Scanning...' : 'Scan Google'}</span>
          </button>
        </div>

        {/* Match Selector Tabs */}
        <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar pt-1">
          {matches.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedMatchId(m.id)}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all whitespace-nowrap flex items-center gap-1.5 ${
                m.id === selectedMatchId
                  ? 'bg-slate-800 text-white border border-emerald-500/50 shadow-sm'
                  : 'bg-slate-900/60 text-slate-400 hover:bg-slate-800/80 hover:text-slate-200'
              }`}
            >
              <span>{m.team1.shortName} vs {m.team2.shortName}</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            </button>
          ))}
        </div>
      </div>

      {/* Live Match Scoreboard Card */}
      {currentMatch && (
        <div className="bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 border border-slate-800 rounded-xl p-3 mb-2.5 shadow-inner">
          <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono mb-1.5">
            <span className="truncate max-w-[200px]">{currentMatch.seriesName}</span>
            <span className="text-emerald-400 font-bold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              LIVE • {currentMatch.matchType}
            </span>
          </div>

          {/* Teams and Scores */}
          <div className="grid grid-cols-2 gap-2 my-1.5 bg-slate-950/60 p-2 rounded-lg border border-slate-800/60">
            {/* Team 1 */}
            <div className={`p-1.5 rounded ${currentMatch.team1.isBatting ? 'bg-slate-900 border border-slate-700' : ''}`}>
              <div className="flex items-center gap-1.5 text-xs font-bold text-white">
                <span>{currentMatch.team1.flag}</span>
                <span>{currentMatch.team1.name}</span>
              </div>
              <div className="font-mono text-sm font-extrabold text-amber-400 mt-0.5">
                {currentMatch.team1.score}
              </div>
            </div>

            {/* Team 2 */}
            <div className={`p-1.5 rounded ${currentMatch.team2.isBatting ? 'bg-slate-900 border border-slate-700' : ''}`}>
              <div className="flex items-center gap-1.5 text-xs font-bold text-white">
                <span>{currentMatch.team2.flag}</span>
                <span>{currentMatch.team2.name}</span>
              </div>
              <div className="font-mono text-sm font-extrabold text-emerald-400 mt-0.5">
                {currentMatch.team2.score}
              </div>
            </div>
          </div>

          {/* Match Status & Live Batsmen / Bowler */}
          <div className="text-[11px] font-semibold text-slate-300 my-1 bg-slate-900/40 px-2 py-1 rounded border border-slate-800/50">
            {currentMatch.statusText}
          </div>

          {/* Recent Balls String & Live Ball Simulation Button */}
          <div className="flex items-center justify-between gap-2 mt-2 pt-2 border-t border-slate-800/60">
            <div className="flex items-center gap-1">
              <span className="text-[10px] uppercase font-mono text-slate-400">Over {currentMatch.currentOverNumber}:</span>
              <div className="flex items-center gap-1">
                {currentMatch.recentBalls.map((b, idx) => (
                  <span
                    key={idx}
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono font-bold ${
                      b === 'W'
                        ? 'bg-rose-600 text-white animate-bounce'
                        : b === '6'
                        ? 'bg-purple-600 text-white font-extrabold'
                        : b === '4'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    {b}
                  </span>
                ))}
              </div>
            </div>

            <button
              onClick={handleSimulateBall}
              className="px-2 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-300 hover:text-white text-[10px] font-mono font-bold flex items-center gap-1 transition-all active:scale-95 cursor-pointer"
            >
              <Play className="w-2.5 h-2.5 fill-indigo-300" />
              <span>Sim Ball</span>
            </button>
          </div>

          {/* Live Recent Ball Notice */}
          {recentBallNotice && (
            <div className="mt-1.5 text-[10px] font-mono text-amber-300 bg-amber-950/40 border border-amber-500/30 px-2 py-0.5 rounded animate-pulse">
              {recentBallNotice}
            </div>
          )}

          {/* Win Probability Bar */}
          <div className="mt-2">
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1">
              <span>{currentMatch.team1.shortName} {currentMatch.winProbability.team1Pct}%</span>
              <span>Win Probability</span>
              <span>{currentMatch.team2.shortName} {currentMatch.winProbability.team2Pct}%</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
              <div
                className="bg-amber-500 h-full transition-all duration-500"
                style={{ width: `${currentMatch.winProbability.team1Pct}%` }}
              />
              <div
                className="bg-emerald-400 h-full transition-all duration-500"
                style={{ width: `${currentMatch.winProbability.team2Pct}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Navigation Sub-Tabs (Markets vs My Sports Bets) */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2">
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setActiveTab('MARKETS')}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
              activeTab === 'MARKETS'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white bg-slate-900/60'
            }`}
          >
            Live Markets ({currentMatch?.markets.length || 0})
          </button>
          <button
            onClick={() => setActiveTab('MY_BETS')}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all flex items-center gap-1 ${
              activeTab === 'MY_BETS'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white bg-slate-900/60'
            }`}
          >
            My Bets ({userSportsBets.length})
            {openBets.length > 0 && (
              <span className="px-1.5 py-0.2 rounded-full bg-indigo-400 text-slate-950 text-[9px] font-mono font-black">
                {openBets.length}
              </span>
            )}
          </button>
        </div>

        {/* Bet Size Quick Selector */}
        <div className="flex items-center gap-1 bg-slate-900/90 px-2 py-0.5 rounded-lg border border-slate-800 text-[10px] font-mono">
          <span className="text-slate-400">Stake:</span>
          {[25, 50, 100, 250].map((amt) => (
            <button
              key={amt}
              onClick={() => setQuickBetAmount(amt)}
              className={`px-1 py-0.2 rounded font-bold transition-all ${
                quickBetAmount === amt
                  ? 'bg-emerald-500 text-slate-950'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              ${amt}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
        {activeTab === 'MARKETS' && (
          <>
            {currentMatch?.markets.map((market) => {
              const yesOdds = market.yesOddsMultiplier;
              const noOdds = market.noOddsMultiplier;
              const potentialYesWin = Math.round(quickBetAmount * yesOdds);
              const potentialNoWin = Math.round(quickBetAmount * noOdds);

              return (
                <div
                  key={market.id}
                  className={`p-2.5 rounded-xl border transition-all ${
                    market.resolved
                      ? 'bg-slate-900/40 border-slate-800/50 opacity-80'
                      : 'bg-slate-900/70 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-1.5">
                    <div>
                      <div className="text-xs font-bold text-white flex items-center gap-1.5">
                        <Flame className="w-3 h-3 text-amber-400" />
                        <span>{market.title}</span>
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5 line-clamp-1">
                        {market.description}
                      </div>
                    </div>

                    {market.resolved && (
                      <span
                        className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                          market.outcome
                            ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                            : 'bg-rose-950 text-rose-300 border-rose-800'
                        }`}
                      >
                        {market.outcome ? 'YES WON' : 'NO WON'}
                      </span>
                    )}
                  </div>

                  {/* Bet Action Buttons */}
                  {!market.resolved ? (
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      {/* BUY YES */}
                      <button
                        onClick={() => handleBet(market, 'BUY_YES')}
                        className="py-1.5 px-2 rounded-lg bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700/80 text-emerald-300 font-mono text-[11px] font-bold flex items-center justify-between transition-all shadow-sm active:scale-95 cursor-pointer"
                      >
                        <div className="flex flex-col text-left">
                          <span className="text-[10px] text-emerald-400 font-sans font-bold">
                            YES (${quickBetAmount})
                          </span>
                          <span className="text-[9px] text-slate-400 font-normal">
                            Pays ${potentialYesWin}
                          </span>
                        </div>
                        <span className="text-xs font-extrabold text-emerald-300">
                          {yesOdds.toFixed(2)}x
                        </span>
                      </button>

                      {/* BUY NO */}
                      <button
                        onClick={() => handleBet(market, 'BUY_NO')}
                        className="py-1.5 px-2 rounded-lg bg-rose-950/80 hover:bg-rose-900 border border-rose-700/80 text-rose-300 font-mono text-[11px] font-bold flex items-center justify-between transition-all shadow-sm active:scale-95 cursor-pointer"
                      >
                        <div className="flex flex-col text-left">
                          <span className="text-[10px] text-rose-400 font-sans font-bold">
                            NO (${quickBetAmount})
                          </span>
                          <span className="text-[9px] text-slate-400 font-normal">
                            Pays ${potentialNoWin}
                          </span>
                        </div>
                        <span className="text-xs font-extrabold text-rose-300">
                          {noOdds.toFixed(2)}x
                        </span>
                      </button>
                    </div>
                  ) : (
                    <div className="text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-800/60">
                      {market.resolutionText || 'Settled'}
                    </div>
                  )}
                </div>
              );
            })}
          </>
        )}

        {activeTab === 'MY_BETS' && (
          <div className="space-y-2">
            {userSportsBets.length === 0 ? (
              <div className="text-center py-8 text-slate-500 text-xs">
                No active cricket bets placed yet.
              </div>
            ) : (
              userSportsBets.map((bet) => (
                <div
                  key={bet.id}
                  className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs font-mono"
                >
                  <div className="flex items-center justify-between text-slate-300 font-sans font-bold mb-1">
                    <span className="truncate max-w-[200px]">{bet.marketTitle}</span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[10px] font-mono ${
                        bet.status === 'WON'
                          ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          : bet.status === 'LOST'
                          ? 'bg-rose-950 text-rose-300 border border-rose-800'
                          : 'bg-amber-950 text-amber-300 border border-amber-800'
                      }`}
                    >
                      {bet.status}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>
                      Side: <strong className="text-white">{bet.side === 'BUY_YES' ? 'YES' : 'NO'}</strong> @ {bet.oddsMultiplier}x
                    </span>
                    <span>
                      Stake: <strong className="text-white">${bet.stake}</strong>
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] mt-1 pt-1 border-t border-slate-800/60">
                    <span className="text-slate-400">
                      Potential Payout: <strong className="text-emerald-400">${bet.potentialPayout}</strong>
                    </span>
                    {bet.status !== 'OPEN' && (
                      <span
                        className={`font-bold ${
                          bet.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        PnL: {bet.pnl >= 0 ? '+' : ''}${bet.pnl}
                      </span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Google Search Grounding Sources Bar (if any) */}
      {groundingSources.length > 0 && (
        <div className="mt-2 pt-2 border-t border-slate-800/60">
          <div className="text-[9px] uppercase font-mono text-slate-500 mb-1 flex items-center gap-1">
            <Globe2 className="w-2.5 h-2.5" />
            <span>Google Cricket Search Grounding:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {groundingSources.slice(0, 3).map((src, idx) => (
              <a
                key={idx}
                href={src.url}
                target="_blank"
                rel="noreferrer"
                className="text-[9px] font-mono text-emerald-400 hover:text-emerald-300 hover:underline bg-emerald-950/40 px-1.5 py-0.5 rounded border border-emerald-800/50 flex items-center gap-1"
              >
                <span className="truncate max-w-[140px]">{src.title}</span>
                <ExternalLink className="w-2 h-2 flex-shrink-0" />
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
