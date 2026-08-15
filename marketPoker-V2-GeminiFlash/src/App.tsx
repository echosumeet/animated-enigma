import React, { useState, useEffect, useRef } from 'react';
import { GamePhase, GameTableState, OrderSide, OrderType } from './types/game';
import { UserSportsBet } from './types/cricket';
import { createInitialTableState, GameManager } from './engine/gameManager';
import { sound } from './engine/sound';
import { Header } from './components/Header';
import { PokerTable } from './components/PokerTable';
import { PokerControls } from './components/PokerControls';
import { MarketsPanel } from './components/MarketsPanel';
import { LiveCricketPanel } from './components/LiveCricketPanel';
import { OrderBookModal } from './components/OrderBookModal';
import { PortfolioDrawer } from './components/PortfolioDrawer';
import { PostHandModal } from './components/PostHandModal';
import { DevInspector } from './components/DevInspector';
import { SimulationModal } from './components/SimulationModal';
import { TutorialModal } from './components/TutorialModal';
import { ShowcaseModal } from './components/ShowcaseModal';
import { NewTableModal } from './components/NewTableModal';
import { BorrowLoanModal } from './components/BorrowLoanModal';
import {
  Play,
  Sparkles,
  Trophy,
  Zap,
  Globe2,
  Radio,
  Layers,
  Plus,
  Grid,
  Maximize2,
  X,
  FastForward,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';

export default function App() {
  // Multi-Table State: supports 1 or 2 simultaneous tables
  const [tables, setTables] = useState<GameTableState[]>(() => [
    createInitialTableState('You', 'table-1', 'Table 1: Main Arena', 'emerald', 1, 'default'),
  ]);
  const [activeTableId, setActiveTableId] = useState<string>('table-1');
  const [viewMode, setViewMode] = useState<'single' | 'dual'>('single');
  const [isNewTableModalOpen, setIsNewTableModalOpen] = useState<boolean>(false);

  // Modals & Panels State
  const [selectedMarketId, setSelectedMarketId] = useState<string>('market-player-1');
  const [selectedTableForMarket, setSelectedTableForMarket] = useState<string>('table-1');
  const [isOrderBookOpen, setIsOrderBookOpen] = useState<boolean>(false);
  const [orderBookInitialSide, setOrderBookInitialSide] = useState<OrderSide>('BUY_YES');
  const [isSoundOn, setIsSoundOn] = useState<boolean>(true);
  const [isTutorialOpen, setIsTutorialOpen] = useState<boolean>(false);
  const [isSimulationOpen, setIsSimulationOpen] = useState<boolean>(false);
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(false);
  const [isShowcaseOpen, setIsShowcaseOpen] = useState<boolean>(false);
  const [rightPanelTab, setRightPanelTab] = useState<'POKER_MARKETS' | 'LIVE_SPORTS'>('POKER_MARKETS');
  const [userSportsBets, setUserSportsBets] = useState<UserSportsBet[]>([]);

  const tablesRef = useRef<GameTableState[]>(tables);
  tablesRef.current = tables;

  // Main game loop: manages turn timers, bot decisions, auto-actions on timeout, and ticking sounds
  useEffect(() => {
    const timer = setInterval(() => {
      setTables((currentTables) => {
        let hasChanges = false;
        const updated = currentTables.map((t) => {
          if (t.phase === 'WAITING' || t.phase === 'HAND_RESULTS' || t.isPaused) {
            return t;
          }

          // 1. Trading phase auto-countdown
          if (t.phase.endsWith('_TRADING')) {
            const newPhaseTime = (t.phaseTimeRemaining ?? 15) - 1;
            hasChanges = true;
            if (newPhaseTime <= 0) {
              let processed = GameManager.processBotActions(t);
              if (processed.phase === 'PREFLOP_TRADING') processed.phase = 'PREFLOP_BETTING';
              else if (processed.phase === 'FLOP_TRADING') processed.phase = 'FLOP_BETTING';
              else if (processed.phase === 'TURN_TRADING') processed.phase = 'TURN_BETTING';
              else if (processed.phase === 'RIVER_TRADING') processed.phase = 'RIVER_BETTING';
              processed.turnTimeRemaining = processed.currentTurnIndex === 0 ? 15 : 2;
              processed.totalTurnTime = processed.currentTurnIndex === 0 ? 15 : 2;
              if (processed.currentTurnIndex === 0) {
                sound.playTurnAlert();
              }
              return processed;
            }
            return { ...t, phaseTimeRemaining: newPhaseTime };
          }

          // 2. Betting phase turn timers
          if (t.phase.endsWith('_BETTING')) {
            const currentTurnPlayer = t.players[t.currentTurnIndex];
            if (!currentTurnPlayer || currentTurnPlayer.status !== 'active') {
              // If current turn player is inactive or all-in, advance turn
              hasChanges = true;
              return GameManager.handlePokerAction(t, currentTurnPlayer?.id || 'player-1', 'CHECK');
            }

            const isUser = t.currentTurnIndex === 0;
            const currentRem = t.turnTimeRemaining !== undefined ? t.turnTimeRemaining : (isUser ? 15 : 2);
            const nextRem = currentRem - 1;
            hasChanges = true;

            // Clock ticking sound for user's turn
            if (isUser) {
              if (nextRem <= 5 && nextRem >= 0) {
                sound.playClockTick(true);
              } else if (nextRem > 5 && nextRem % 2 === 0) {
                sound.playClockTick(false);
              }
            }

            // If time expires
            if (nextRem <= 0) {
              if (isUser) {
                // User timeout -> Auto-Check if no call diff, otherwise Auto-Fold
                const callDiff = Math.max(0, t.currentHighestBet - currentTurnPlayer.currentBet);
                if (callDiff === 0) {
                  return GameManager.handlePokerAction(t, 'player-1', 'CHECK');
                } else {
                  return GameManager.handlePokerAction(t, 'player-1', 'FOLD');
                }
              } else {
                // Bot turn -> Process Bot AI action
                return GameManager.processBotActions(t);
              }
            }

            // Otherwise decrement turn timer
            return {
              ...t,
              turnTimeRemaining: nextRem,
              totalTurnTime: t.totalTurnTime || (isUser ? 15 : 2),
            };
          }

          return t;
        });
        return hasChanges ? updated : currentTables;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Helper to update a single table by tableId
  const updateTable = (tableId: string, updater: (t: GameTableState) => GameTableState) => {
    setTables((prev) =>
      prev.map((t) => (t.tableId === tableId ? updater(t) : t))
    );
  };

  // Handlers for active tables
  const handleStartGame = (tableId: string) => {
    updateTable(tableId, (t) => GameManager.startNewHand(t));
  };

  const handlePokerAction = (
    tableId: string,
    action: 'CHECK' | 'CALL' | 'BET' | 'RAISE' | 'FOLD',
    amount?: number
  ) => {
    updateTable(tableId, (t) => GameManager.handlePokerAction(t, 'player-1', action, amount));
  };

  const handleFastForward = (tableId: string) => {
    updateTable(tableId, (t) => GameManager.fastForwardHand(t));
  };

  const handleAdvanceToBetting = (tableId: string) => {
    updateTable(tableId, (t) => {
      let processed = GameManager.processBotActions(t);
      if (processed.phase === 'PREFLOP_TRADING') processed.phase = 'PREFLOP_BETTING';
      else if (processed.phase === 'FLOP_TRADING') processed.phase = 'FLOP_BETTING';
      else if (processed.phase === 'TURN_TRADING') processed.phase = 'TURN_BETTING';
      else if (processed.phase === 'RIVER_TRADING') processed.phase = 'RIVER_BETTING';
      return processed;
    });
  };

  const handlePlaceOrder = (
    tableId: string,
    marketId: string,
    side: OrderSide,
    type: OrderType,
    price: number,
    quantity: number
  ) => {
    updateTable(tableId, (t) =>
      GameManager.placeMarketOrder(t, 'player-1', marketId, side, type, price, quantity)
    );
  };

  const handleQuickTrade = (
    tableId: string,
    marketId: string,
    side: 'BUY_YES' | 'BUY_NO',
    dollars?: number
  ) => {
    if (dollars) {
      updateTable(tableId, (t) =>
        GameManager.quickBet(t, 'player-1', marketId, side, dollars)
      );
    } else {
      setSelectedTableForMarket(tableId);
      setSelectedMarketId(marketId);
      setOrderBookInitialSide(side);
      setIsOrderBookOpen(true);
    }
  };

  const handleClosePosition = (
    tableId: string,
    marketId: string,
    side: 'SELL_YES' | 'SELL_NO',
    qty: number
  ) => {
    const currentTable = tables.find((t) => t.tableId === tableId);
    if (!currentTable) return;
    const market = currentTable.markets.find((m) => m.id === marketId);
    if (!market) return;
    const price = side === 'SELL_YES' ? market.currentYesPrice : market.currentNoPrice;
    handlePlaceOrder(tableId, marketId, side, 'MARKET', price, qty);
  };

  // Spin up a 2nd Table
  const handleSpinUpTable = (
    tableName: string,
    theme: 'emerald' | 'sapphire' | 'amber' | 'ruby',
    startingLevel: number,
    botPreset: 'default' | 'high_roller'
  ) => {
    const tableId = `table-${Date.now()}`;
    const newTable = createInitialTableState(
      'You',
      tableId,
      tableName,
      theme,
      startingLevel,
      botPreset
    );
    setTables((prev) => [...prev, newTable]);
    setActiveTableId(tableId);
    setViewMode('dual'); // automatically suggest dual view when 2nd table is opened
  };

  // Close a Table
  const handleCloseTable = (tableId: string) => {
    if (tables.length <= 1) return;
    setTables((prev) => prev.filter((t) => t.tableId !== tableId));
    const remaining = tables.filter((t) => t.tableId !== tableId);
    if (remaining.length > 0) {
      setActiveTableId(remaining[0].tableId);
    }
    setViewMode('single');
  };

  // Sports Betting Integration
  const handlePlaceSportsBet = (bet: UserSportsBet) => {
    // Debit from first table or active table
    updateTable(activeTableId, (prev) => {
      const players = prev.players.map((p) => {
        if (p.id === 'player-1') {
          return { ...p, cashBalance: Number((p.cashBalance - bet.stake).toFixed(2)) };
        }
        return p;
      });
      const user = prev.players.find((p) => p.id === 'player-1');
      const newCash = user ? user.cashBalance - bet.stake : 10000;

      const newLedgerEntry = {
        id: `led-cric-${Date.now()}`,
        timestamp: Date.now(),
        handNumber: prev.handNumber,
        playerId: 'player-1',
        category: 'MARKET_BUY' as const,
        amount: -bet.stake,
        balanceAfter: newCash,
        description: `Live Sports Bet: ${bet.marketTitle} ($${bet.stake} on ${
          bet.side === 'BUY_YES' ? 'YES' : 'NO'
        })`,
      };

      const newEvent = {
        id: `evt-cric-${Date.now()}`,
        timestamp: Date.now(),
        phase: prev.phase,
        type: 'MARKET' as const,
        text: `🏏 Sports Bet Placed: $${bet.stake} on ${bet.marketTitle} (${bet.oddsMultiplier}x)`,
        highlight: true,
      };

      return {
        ...prev,
        players,
        ledger: [newLedgerEntry, ...prev.ledger],
        events: [newEvent, ...prev.events],
      };
    });

    setUserSportsBets((prev) => [bet, ...prev]);
  };

  const handleSettleSportsPayout = (payout: number, description: string) => {
    updateTable(activeTableId, (prev) => {
      const players = prev.players.map((p) => {
        if (p.id === 'player-1') {
          return { ...p, cashBalance: Number((p.cashBalance + payout).toFixed(2)) };
        }
        return p;
      });
      const user = prev.players.find((p) => p.id === 'player-1');
      const newCash = user ? user.cashBalance + payout : 10000;

      const newLedgerEntry = {
        id: `led-cric-settle-${Date.now()}`,
        timestamp: Date.now(),
        handNumber: prev.handNumber,
        playerId: 'player-1',
        category: 'MARKET_SETTLE' as const,
        amount: payout,
        balanceAfter: newCash,
        description,
      };

      const newEvent = {
        id: `evt-cric-settle-${Date.now()}`,
        timestamp: Date.now(),
        phase: prev.phase,
        type: 'MARKET' as const,
        text: `🏆 Live Sports Payout Won: +$${payout} (${description})`,
        highlight: true,
      };

      return {
        ...prev,
        players,
        ledger: [newLedgerEntry, ...prev.ledger],
        events: [newEvent, ...prev.events],
      };
    });
  };

  const handleToggleSound = () => {
    const enabled = sound.toggleSound();
    setIsSoundOn(enabled);
  };

  const activeTable = tables.find((t) => t.tableId === activeTableId) || tables[0];
  const userPlayer = activeTable.players.find((p) => p.id === 'player-1') || activeTable.players[0];

  const marketTable = tables.find((t) => t.tableId === selectedTableForMarket) || activeTable;
  const selectedMarket =
    marketTable.markets.find((m) => m.id === selectedMarketId) || marketTable.markets[0];
  const userPosition = marketTable.positions.find(
    (p) => p.playerId === 'player-1' && p.marketId === selectedMarketId
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white antialiased">
      {/* Top Bar Navigation & Header */}
      <Header
        handNumber={activeTable.handNumber}
        tournamentLevel={activeTable.tournamentLevel}
        handsInCurrentLevel={activeTable.handsInCurrentLevel}
        handsPerLevel={activeTable.handsPerLevel}
        synergyMultiplierPct={activeTable.synergyMultiplierPct}
        smallBlind={activeTable.smallBlind}
        bigBlind={activeTable.bigBlind}
        phase={activeTable.phase}
        userPlayer={userPlayer}
        positions={activeTable.positions}
        markets={activeTable.markets}
        isSoundOn={isSoundOn}
        onToggleSound={handleToggleSound}
        onOpenTutorial={() => setIsTutorialOpen(true)}
        onOpenSimulation={() => setIsSimulationOpen(true)}
        onOpenInspector={() => setIsInspectorOpen(true)}
        onOpenShowcase={() => setIsShowcaseOpen(true)}
      />

      {/* Multi-Table Controller Strip */}
      <div className="bg-slate-950/95 border-b border-slate-800/80 px-3 sm:px-4 py-2 flex flex-wrap items-center justify-between gap-2 z-20">
        {/* Table Selector Tabs */}
        <div className="flex items-center gap-2 overflow-x-auto">
          {tables.map((t, idx) => {
            const isTabActive = t.tableId === activeTableId;
            const isPlaying = t.phase !== 'WAITING' && t.phase !== 'HAND_RESULTS';
            const userInTable = t.players.find((p) => p.id === 'player-1');
            const isFolded = userInTable?.status === 'folded';
            const isUserTurnOnThis =
              t.currentTurnIndex === 0 &&
              t.phase.endsWith('_BETTING') &&
              userInTable?.status === 'active';

            return (
              <div key={t.tableId} className="flex items-center">
                <button
                  onClick={() => {
                    setActiveTableId(t.tableId);
                    setSelectedTableForMarket(t.tableId);
                  }}
                  className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-2 border cursor-pointer ${
                    isUserTurnOnThis
                      ? 'bg-amber-500/20 border-amber-400 text-amber-300 ring-2 ring-amber-400/50 shadow-lg shadow-amber-500/20 animate-pulse'
                      : isTabActive
                      ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-lg ring-1 ring-indigo-500/30'
                      : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/70'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      isUserTurnOnThis
                        ? 'bg-amber-400 animate-ping'
                        : isPlaying
                        ? isFolded
                          ? 'bg-amber-400'
                          : 'bg-emerald-400 animate-pulse'
                        : 'bg-slate-500'
                    }`}
                  />
                  <span>{t.tableName || `Table ${idx + 1}`}</span>
                  {isUserTurnOnThis ? (
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-amber-400 text-slate-950 font-black">
                      🔔 {t.turnTimeRemaining}s
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-amber-300 font-semibold">
                      {t.phase === 'WAITING'
                        ? 'Waiting'
                        : `Pot: $${t.mainPot.toLocaleString()}`}
                    </span>
                  )}
                </button>

                {/* Close 2nd Table Button */}
                {tables.length > 1 && idx > 0 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCloseTable(t.tableId);
                    }}
                    title="Close Table"
                    className="p-1 -ml-2 rounded-full bg-slate-800/80 hover:bg-rose-900 border border-slate-700 text-slate-400 hover:text-rose-200 transition-colors cursor-pointer"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            );
          })}

          {/* Spin Up 2nd Table Button */}
          {tables.length < 2 && (
            <button
              onClick={() => setIsNewTableModalOpen(true)}
              className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-amber-500/20 to-indigo-500/20 hover:from-amber-500/30 hover:to-indigo-500/30 border border-amber-500/40 text-amber-300 font-bold text-xs flex items-center gap-1.5 shadow-sm transition-all cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5 text-amber-400" />
              <span>Spin Up 2nd Table</span>
            </button>
          )}
        </div>

        {/* View Mode Toggle: Dual Play vs Single Tab */}
        {tables.length > 1 && (
          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setViewMode('dual')}
              className={`px-2.5 py-1 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
                viewMode === 'dual'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Grid className="w-3.5 h-3.5" />
              <span>Dual Play (2 Tables)</span>
            </button>
            <button
              onClick={() => setViewMode('single')}
              className={`px-2.5 py-1 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer ${
                viewMode === 'single'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Maximize2 className="w-3.5 h-3.5" />
              <span>Single Table</span>
            </button>
          </div>
        )}
      </div>

      {/* Main Game Arena */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        {/* Left / Center Column: Poker Tables & Controls */}
        <div className="lg:col-span-8 flex flex-col gap-5">
          {/* Attention Banner if another table needs action in Single View Mode */}
          {viewMode === 'single' && tables.some((t) => t.tableId !== activeTableId && t.currentTurnIndex === 0 && t.phase.endsWith('_BETTING') && t.players[0]?.status === 'active') && (
            <div className="bg-amber-500/15 border-2 border-amber-400 rounded-2xl p-3 flex items-center justify-between shadow-xl shadow-amber-500/10 animate-pulse">
              <div className="flex items-center gap-2 text-xs font-bold text-amber-300">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping" />
                <span>
                  Attention: Action is required on{' '}
                  <strong className="text-white">
                    {tables.find((t) => t.tableId !== activeTableId && t.currentTurnIndex === 0)?.tableName}
                  </strong>{' '}
                  ({tables.find((t) => t.tableId !== activeTableId && t.currentTurnIndex === 0)?.turnTimeRemaining}s remaining)!
                </span>
              </div>
              <button
                onClick={() => {
                  const target = tables.find((t) => t.tableId !== activeTableId && t.currentTurnIndex === 0);
                  if (target) {
                    setActiveTableId(target.tableId);
                    setSelectedTableForMarket(target.tableId);
                  }
                }}
                className="px-3.5 py-1.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-black text-xs flex items-center gap-1 shadow-md cursor-pointer"
              >
                <span>Switch to Table</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Dual Table Mode */}
          {viewMode === 'dual' && tables.length >= 2 ? (
            <div className="grid grid-cols-1 gap-5">
              {tables.map((t) => {
                const uPlayer = t.players.find((p) => p.id === 'player-1') || t.players[0];
                const isUserTurnOnThis =
                  t.currentTurnIndex === 0 &&
                  t.phase.endsWith('_BETTING') &&
                  uPlayer.status === 'active';

                return (
                  <div
                    key={t.tableId}
                    className={`bg-slate-950/80 rounded-3xl p-3 sm:p-4 border transition-all ${
                      isUserTurnOnThis
                        ? 'border-amber-400 ring-4 ring-amber-400/40 shadow-[0_0_35px_rgba(245,158,11,0.35)]'
                        : t.tableId === activeTableId
                        ? 'border-indigo-500/60 ring-2 ring-indigo-500/20 shadow-xl'
                        : 'border-slate-800/80 hover:border-slate-700'
                    }`}
                    onClick={() => {
                      setActiveTableId(t.tableId);
                      setSelectedTableForMarket(t.tableId);
                    }}
                  >
                    {/* Table Title Bar */}
                    <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/60">
                      <div className="flex items-center gap-2">
                        <span className="font-extrabold text-sm text-white">{t.tableName}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300">
                          Level {t.tournamentLevel} ({t.smallBlind}/{t.bigBlind})
                        </span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300">
                          Dual Bounty: +{t.synergyMultiplierPct}%
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {isUserTurnOnThis && (
                          <span className="px-2.5 py-0.5 rounded-full bg-amber-500 text-slate-950 font-black text-xs animate-bounce">
                            🔔 YOUR TURN ({t.turnTimeRemaining}s)
                          </span>
                        )}
                        {t.phase === 'WAITING' && (
                          <button
                            onClick={() => handleStartGame(t.tableId)}
                            className="px-3 py-1 rounded-xl bg-gradient-to-r from-amber-500 to-emerald-400 hover:from-amber-400 hover:to-emerald-300 text-slate-950 font-black text-xs flex items-center gap-1.5 shadow-md cursor-pointer"
                          >
                            <Play className="w-3 h-3 fill-slate-950" />
                            Start Hand #1
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Poker Felt Table */}
                    <PokerTable
                      players={t.players}
                      communityCards={t.communityCards}
                      mainPot={t.mainPot}
                      currentTurnIndex={t.currentTurnIndex}
                      dealerIndex={t.dealerIndex}
                      phase={t.phase}
                      tableName={t.tableName}
                      tableTheme={t.tableTheme}
                      handNumber={t.handNumber}
                      tournamentLevel={t.tournamentLevel}
                      synergyMultiplierPct={t.synergyMultiplierPct}
                      turnTimeRemaining={t.turnTimeRemaining}
                      totalTurnTime={t.totalTurnTime}
                      compact={true}
                    />

                    {/* Poker Controls */}
                    {t.phase !== 'WAITING' && t.phase !== 'HAND_RESULTS' && (
                      <div className="mt-3">
                        <PokerControls
                          userPlayer={uPlayer}
                          currentTurnIndex={t.currentTurnIndex}
                          currentHighestBet={t.currentHighestBet}
                          minRaise={t.minRaise}
                          mainPot={t.mainPot}
                          phase={t.phase}
                          turnTimeRemaining={t.turnTimeRemaining}
                          onAction={(act, amt) => handlePokerAction(t.tableId, act, amt)}
                          onAdvanceToBetting={() => handleAdvanceToBetting(t.tableId)}
                          onFastForward={() => handleFastForward(t.tableId)}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            /* Single Table Mode */
            <div className="flex flex-col gap-4">
              {/* Start First Hand Hero Banner (if Waiting) */}
              {activeTable.phase === 'WAITING' && (
                <div className="bg-gradient-to-r from-indigo-950 via-slate-900 to-slate-950 border border-indigo-500/30 rounded-3xl p-6 sm:p-8 text-center shadow-2xl flex flex-col items-center justify-center">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 mb-3">
                    <Trophy className="w-6 h-6" />
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black text-white tracking-wide">
                    {activeTable.tableName} • Live Tournament Arena
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-400 max-w-lg mt-1 mb-5">
                    Escalating blinds, real-time Kalshi/Polymarket style live questions on each street, Google Live Cricket scores scanner & micro-betting, and <strong>Dual Victory Synergy Bounties</strong>.
                  </p>
                  <button
                    onClick={() => handleStartGame(activeTable.tableId)}
                    className="px-8 py-3.5 rounded-2xl bg-gradient-to-r from-amber-500 via-emerald-400 to-teal-400 hover:from-amber-400 hover:to-teal-300 text-slate-950 font-black text-sm flex items-center gap-2.5 shadow-xl shadow-emerald-500/25 transition-all cursor-pointer hover:scale-105 active:scale-95"
                  >
                    <Play className="w-4 h-4 fill-slate-950" />
                    <span>Start Tournament (Hand #1 • Level {activeTable.tournamentLevel})</span>
                  </button>
                </div>
              )}

              {/* 5-Seat Poker Felt Table */}
              <PokerTable
                players={activeTable.players}
                communityCards={activeTable.communityCards}
                mainPot={activeTable.mainPot}
                currentTurnIndex={activeTable.currentTurnIndex}
                dealerIndex={activeTable.dealerIndex}
                phase={activeTable.phase}
                tableName={activeTable.tableName}
                tableTheme={activeTable.tableTheme}
                handNumber={activeTable.handNumber}
                tournamentLevel={activeTable.tournamentLevel}
                synergyMultiplierPct={activeTable.synergyMultiplierPct}
                turnTimeRemaining={activeTable.turnTimeRemaining}
                totalTurnTime={activeTable.totalTurnTime}
              />

              {/* Poker Controls Bar */}
              {activeTable.phase !== 'WAITING' && activeTable.phase !== 'HAND_RESULTS' && (
                <PokerControls
                  userPlayer={userPlayer}
                  currentTurnIndex={activeTable.currentTurnIndex}
                  currentHighestBet={activeTable.currentHighestBet}
                  minRaise={activeTable.minRaise}
                  mainPot={activeTable.mainPot}
                  phase={activeTable.phase}
                  turnTimeRemaining={activeTable.turnTimeRemaining}
                  onAction={(act, amt) => handlePokerAction(activeTable.tableId, act, amt)}
                  onAdvanceToBetting={() => handleAdvanceToBetting(activeTable.tableId)}
                  onFastForward={() => handleFastForward(activeTable.tableId)}
                />
              )}
            </div>
          )}

          {/* Active Portfolio Drawer */}
          <PortfolioDrawer
            positions={activeTable.positions.filter((p) => p.playerId === 'player-1')}
            markets={activeTable.markets}
            onClosePosition={(mId, side, qty) =>
              handleClosePosition(activeTable.tableId, mId, side, qty)
            }
          />
        </div>

        {/* Right Column: Toggleable Prediction Markets & Live Sports Arena */}
        <div className="lg:col-span-4 h-[calc(100vh-130px)] sticky top-28 flex flex-col gap-2">
          {/* Top Toggle Switcher: Poker Markets vs Live Sports */}
          <div className="bg-slate-900/90 p-1.5 rounded-2xl border border-slate-800 flex items-center gap-1 shadow-lg backdrop-blur-md">
            <button
              onClick={() => setRightPanelTab('POKER_MARKETS')}
              className={`flex-1 py-2 px-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
                rightPanelTab === 'POKER_MARKETS'
                  ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Poker Markets</span>
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300">
                {marketTable.markets.filter((m) => !m.resolved).length}
              </span>
            </button>

            <button
              onClick={() => setRightPanelTab('LIVE_SPORTS')}
              className={`flex-1 py-2 px-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer relative ${
                rightPanelTab === 'LIVE_SPORTS'
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
              <span>Live Sports Bet</span>
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-400/20 text-emerald-300 font-extrabold">
                CRICKET
              </span>
            </button>
          </div>

          {/* If 2 tables are active, show indicator of which table's prediction markets are displayed */}
          {tables.length > 1 && rightPanelTab === 'POKER_MARKETS' && (
            <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/60 rounded-xl border border-slate-800/60 text-xs">
              <span className="text-slate-400">Markets for:</span>
              <div className="flex items-center gap-1">
                {tables.map((t) => (
                  <button
                    key={t.tableId}
                    onClick={() => setSelectedTableForMarket(t.tableId)}
                    className={`px-2 py-0.5 rounded-lg text-[10px] font-bold font-mono transition-colors ${
                      selectedTableForMarket === t.tableId
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    {t.tableName}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Panel Views */}
          <div className="flex-1 min-h-0">
            {rightPanelTab === 'POKER_MARKETS' ? (
              <MarketsPanel
                markets={marketTable.markets}
                positions={marketTable.positions.filter((p) => p.playerId === 'player-1')}
                selectedMarketId={selectedMarketId}
                synergyMultiplierPct={marketTable.synergyMultiplierPct || 15}
                onSelectMarket={(id) => {
                  setSelectedMarketId(id);
                  setIsOrderBookOpen(true);
                }}
                onQuickTrade={(mId, side, dlrs) =>
                  handleQuickTrade(marketTable.tableId, mId, side, dlrs)
                }
              />
            ) : (
              <LiveCricketPanel
                userCash={userPlayer.cashBalance}
                onPlaceSportsBet={handlePlaceSportsBet}
                onSettleSportsPayout={handleSettleSportsPayout}
                userSportsBets={userSportsBets}
              />
            )}
          </div>
        </div>
      </main>

      {/* Spin Up New Table Modal */}
      {isNewTableModalOpen && (
        <NewTableModal
          isOpen={isNewTableModalOpen}
          onClose={() => setIsNewTableModalOpen(false)}
          onSpinUp={handleSpinUpTable}
        />
      )}

      {/* Order Book Depth & Execution Ticket Modal */}
      {isOrderBookOpen && selectedMarket && (
        <OrderBookModal
          market={selectedMarket}
          userPlayer={userPlayer}
          position={userPosition}
          initialSide={orderBookInitialSide}
          onClose={() => setIsOrderBookOpen(false)}
          onPlaceOrder={(mId, side, type, price, qty) =>
            handlePlaceOrder(marketTable.tableId, mId, side, type, price, qty)
          }
        />
      )}

      {/* Post-Hand Analytics Modal */}
      {activeTable.phase === 'HAND_RESULTS' && activeTable.lastHandAnalytics && (
        <PostHandModal
          analytics={activeTable.lastHandAnalytics}
          onNextHand={() => handleStartGame(activeTable.tableId)}
        />
      )}

      {/* 100-to-1,000 Hand Simulation Modal */}
      {isSimulationOpen && (
        <SimulationModal onClose={() => setIsSimulationOpen(false)} />
      )}

      {/* Dev Inspector Modal */}
      {isInspectorOpen && (
        <DevInspector state={activeTable} onClose={() => setIsInspectorOpen(false)} />
      )}

      {/* Tutorial Modal */}
      {isTutorialOpen && (
        <TutorialModal onClose={() => setIsTutorialOpen(false)} />
      )}

      {/* AI Engineering Showcase & Architecture Modal */}
      {isShowcaseOpen && (
        <ShowcaseModal isOpen={isShowcaseOpen} onClose={() => setIsShowcaseOpen(false)} />
      )}
    </div>
  );
}
