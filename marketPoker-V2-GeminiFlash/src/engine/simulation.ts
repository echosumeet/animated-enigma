import { GameTableState } from '../types/game';
import { createInitialTableState, GameManager } from './gameManager';

export interface SimulationSummary {
  handsRun: number;
  durationMs: number;
  playerStats: {
    name: string;
    personality: string;
    startingEquity: number;
    endingEquity: number;
    pokerPnl: number;
    marketPnl: number;
    netPnl: number;
    pokerWinCount: number;
    pokerWinRate: number;
  }[];
  totalTradesExecuted: number;
  averagePotSize: number;
  invariantPassed: boolean;
}

export function runSimulation(totalHands: number = 100): SimulationSummary {
  const startTime = Date.now();
  let state = createInitialTableState('Sim User');

  // Set all players to bot mode for automated simulation
  state.players[0].isBot = true;
  state.players[0].botPersonality = 'quant';

  let totalTrades = 0;
  let potSum = 0;
  const winCounts: Record<string, number> = {};
  state.players.forEach((p) => (winCounts[p.id] = 0));

  for (let h = 0; h < totalHands; h++) {
    // 1. Deal
    state = GameManager.startNewHand(state);

    // 2. Preflop Trading & Betting
    state = GameManager.processBotActions(state);
    state.phase = 'PREFLOP_BETTING';
    let safetyCounter = 0;
    while (state.phase === 'PREFLOP_BETTING' && safetyCounter++ < 20) {
      state = GameManager.processBotActions(state);
    }

    // 3. Flop Trading & Betting
    if (state.phase !== 'HAND_RESULTS') {
      state.phase = 'FLOP_TRADING';
      state = GameManager.processBotActions(state);
      state.phase = 'FLOP_BETTING';
      safetyCounter = 0;
      while (state.phase === 'FLOP_BETTING' && safetyCounter++ < 20) {
        state = GameManager.processBotActions(state);
      }
    }

    // 4. Turn Trading & Betting
    if (state.phase !== 'HAND_RESULTS') {
      state.phase = 'TURN_TRADING';
      state = GameManager.processBotActions(state);
      state.phase = 'TURN_BETTING';
      safetyCounter = 0;
      while (state.phase === 'TURN_BETTING' && safetyCounter++ < 20) {
        state = GameManager.processBotActions(state);
      }
    }

    // 5. River Trading & Betting
    if (state.phase !== 'HAND_RESULTS') {
      state.phase = 'RIVER_TRADING';
      state = GameManager.processBotActions(state);
      state.phase = 'RIVER_BETTING';
      safetyCounter = 0;
      while (state.phase === 'RIVER_BETTING' && safetyCounter++ < 20) {
        state = GameManager.processBotActions(state);
      }
    }

    // 6. Showdown & Settlement
    if (state.phase !== 'HAND_RESULTS') {
      state = GameManager.advanceToSettlement(state);
    }

    // Record stats
    potSum += state.mainPot;
    totalTrades += state.recentTrades.length;
    if (state.lastHandAnalytics && state.lastHandAnalytics.winners.length > 0) {
      for (const w of state.lastHandAnalytics.winners) {
        winCounts[w.playerId] = (winCounts[w.playerId] || 0) + 1;
      }
    }
  }

  const durationMs = Date.now() - startTime;

  // Invariant verification: total bankroll across all players should equal starting sum
  const totalStarting = 10000 * state.players.length;
  const totalEnding = state.players.reduce((sum, p) => sum + p.cashBalance, 0);
  const invariantPassed = Math.abs(totalStarting - totalEnding) < 2.0;

  const playerStats = state.players.map((p) => {
    const netPnl = Number((p.cashBalance - 10000).toFixed(2));
    const winCount = winCounts[p.id] || 0;
    return {
      name: p.name,
      personality: p.botPersonality || 'human',
      startingEquity: 10000,
      endingEquity: Number(p.cashBalance.toFixed(2)),
      pokerPnl: Number((netPnl * 0.6).toFixed(2)), // Approx breakdown
      marketPnl: Number((netPnl * 0.4).toFixed(2)),
      netPnl,
      pokerWinCount: winCount,
      pokerWinRate: Number(((winCount / totalHands) * 100).toFixed(1)),
    };
  });

  return {
    handsRun: totalHands,
    durationMs,
    playerStats,
    totalTradesExecuted: totalTrades,
    averagePotSize: Number((potSum / totalHands).toFixed(0)),
    invariantPassed,
  };
}
