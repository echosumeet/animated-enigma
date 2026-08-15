import {
  Card,
  GameEvent,
  GamePhase,
  GameTableState,
  Order,
  OrderSide,
  OrderType,
  Player,
  Position,
  Trade,
} from '../types/game';
import { BotEngine, BotObservation } from './bots';
import { LedgerEngine } from './ledger';
import { createInitialMarkets, executeOrder, generateStreetMarkets, updatePosition } from './market';
import { createDeck, evaluateHand, shuffleDeck } from './poker';
import { settleHandAndMarkets, settleStreetMarkets } from './settlement';
import { sound } from './sound';

export const INITIAL_BANKROLL = 10000;
export const HANDS_PER_LEVEL = 3;

export function getTournamentLevelConfig(level: number): {
  smallBlind: number;
  bigBlind: number;
  synergyMultiplierPct: number;
} {
  switch (level) {
    case 1:
      return { smallBlind: 50, bigBlind: 100, synergyMultiplierPct: 15 };
    case 2:
      return { smallBlind: 100, bigBlind: 200, synergyMultiplierPct: 30 };
    case 3:
      return { smallBlind: 200, bigBlind: 400, synergyMultiplierPct: 50 };
    case 4:
      return { smallBlind: 400, bigBlind: 800, synergyMultiplierPct: 75 };
    case 5:
    default:
      return {
        smallBlind: 800 * Math.pow(2, Math.max(0, level - 5)),
        bigBlind: 1600 * Math.pow(2, Math.max(0, level - 5)),
        synergyMultiplierPct: 100,
      };
  }
}

export function createInitialTableState(
  userName: string = 'You',
  tableId: string = 'table-1',
  tableName: string = 'Table 1: Main Arena',
  theme: 'emerald' | 'sapphire' | 'amber' | 'ruby' = 'emerald',
  startingLevel: number = 1,
  botPreset: 'default' | 'high_roller' = 'default'
): GameTableState {
  const isHighRoller = botPreset === 'high_roller';
  const players: Player[] = [
    {
      id: 'player-1',
      name: userName,
      avatar: 'user',
      isBot: false,
      seatNumber: 1,
      cashBalance: INITIAL_BANKROLL,
      reservedCash: 0,
      currentBet: 0,
      totalHandBet: 0,
      status: 'active',
      cards: [],
      showCards: true,
      realizedHandPnl: 0,
    },
    {
      id: 'player-2',
      name: isHighRoller ? 'Viktor (Deep Quant)' : 'Alice (The Quant)',
      avatar: 'quant',
      isBot: true,
      botPersonality: 'quant',
      botDescription: isHighRoller
        ? 'High stakes neural EV modeller and Bayesian probability specialist.'
        : 'Rigorous EV calculations, Kelly sizing, targets mispriced prediction markets.',
      seatNumber: 2,
      cashBalance: INITIAL_BANKROLL,
      reservedCash: 0,
      currentBet: 0,
      totalHandBet: 0,
      status: 'active',
      cards: [],
      showCards: false,
      realizedHandPnl: 0,
    },
    {
      id: 'player-3',
      name: isHighRoller ? 'Jax (Momentum Scalper)' : 'Marcus (The Trader)',
      avatar: 'trader',
      isBot: true,
      botPersonality: 'trader',
      botDescription: isHighRoller
        ? 'High frequency liquidity aggressor, drives breakout contracts.'
        : 'Momentum breakout follower, fast scalper, drives rapid price discovery.',
      seatNumber: 3,
      cashBalance: INITIAL_BANKROLL,
      reservedCash: 0,
      currentBet: 0,
      totalHandBet: 0,
      status: 'active',
      cards: [],
      showCards: false,
      realizedHandPnl: 0,
    },
    {
      id: 'player-4',
      name: isHighRoller ? 'Elena (The Apex Shark)' : 'Vance (The Shark)',
      avatar: 'shark',
      isBot: true,
      botPersonality: 'poker_shark',
      botDescription: isHighRoller
        ? 'WSOP bracelet winner, exploits fear with multi-street bluff barrels.'
        : 'High-stakes poker pro, strategic bluffing via market manipulation.',
      seatNumber: 4,
      cashBalance: INITIAL_BANKROLL,
      reservedCash: 0,
      currentBet: 0,
      totalHandBet: 0,
      status: 'active',
      cards: [],
      showCards: false,
      realizedHandPnl: 0,
    },
    {
      id: 'player-5',
      name: isHighRoller ? 'Bruno (The Whale)' : 'Rex (The Degen)',
      avatar: 'degen',
      isBot: true,
      botPersonality: 'degen',
      botDescription: isHighRoller
        ? 'Billionaire risk-junkie, fearless all-in shoves and longshot punts.'
        : 'High volatility risk-taker, bets big on longshots and aggressive draws.',
      seatNumber: 5,
      cashBalance: INITIAL_BANKROLL,
      reservedCash: 0,
      currentBet: 0,
      totalHandBet: 0,
      status: 'active',
      cards: [],
      showCards: false,
      realizedHandPnl: 0,
    },
  ];

  const markets = createInitialMarkets(players.map((p) => p.name));
  const levelConfig = getTournamentLevelConfig(startingLevel);

  return {
    tableId,
    tableName,
    tableTheme: theme,
    raiseCount: 0,
    handNumber: 0,
    tournamentLevel: startingLevel,
    handsInCurrentLevel: 0,
    handsPerLevel: HANDS_PER_LEVEL,
    synergyMultiplierPct: levelConfig.synergyMultiplierPct,
    dealerIndex: 0,
    currentTurnIndex: 1,
    phase: 'WAITING',
    phaseTimeRemaining: 0,
    turnTimeRemaining: 15,
    totalTurnTime: 15,
    smallBlind: levelConfig.smallBlind,
    bigBlind: levelConfig.bigBlind,
    currentHighestBet: 0,
    minRaise: levelConfig.bigBlind,
    mainPot: 0,
    sidePots: [],
    deck: [],
    communityCards: [],
    players,
    markets,
    positions: [],
    openOrders: [],
    recentTrades: [],
    events: [
      {
        id: `evt-init-${tableId}`,
        timestamp: Date.now(),
        phase: 'WAITING',
        type: 'SYSTEM',
        text: `${tableName} initialized: Level ${startingLevel} (${levelConfig.smallBlind}/${levelConfig.bigBlind} Blinds, +${levelConfig.synergyMultiplierPct}% Dual Synergy). Ready for Hand #1.`,
      },
    ],
    ledger: [],
    isPaused: false,
    autoPlayNextHand: false,
    eliminatedPlayerIds: [],
  };
}

export class GameManager {
  // Start a new poker hand with tournament level progression
  public static startNewHand(prevState: GameTableState): GameTableState {
    const handNum = prevState.handNumber + 1;
    let tournamentLevel = prevState.tournamentLevel;
    let handsInCurrentLevel = prevState.handsInCurrentLevel + 1;

    // Check if tournament blinds escalate
    let levelUpEvent: GameEvent | undefined;
    if (handsInCurrentLevel > HANDS_PER_LEVEL) {
      tournamentLevel += 1;
      handsInCurrentLevel = 1;
      sound.playLevelUp();
      const newCfg = getTournamentLevelConfig(tournamentLevel);
      levelUpEvent = {
        id: `evt-lvl-${Date.now()}`,
        timestamp: Date.now(),
        phase: 'DEALING',
        type: 'TOURNAMENT',
        text: `⚡ TOURNAMENT BLINDS UP! Reached Level ${tournamentLevel}: Blinds ${newCfg.smallBlind}/${newCfg.bigBlind} • Dual Synergy Bonus increased to +${newCfg.synergyMultiplierPct}%!`,
        highlight: true,
      };
    }

    const { smallBlind, bigBlind, synergyMultiplierPct } = getTournamentLevelConfig(tournamentLevel);

    const deck = shuffleDeck(createDeck());
    const dealerIndex = (prevState.dealerIndex + 1) % prevState.players.length;

    // Identify eliminated players
    const eliminatedPlayerIds = prevState.players
      .filter((p) => p.cashBalance < bigBlind)
      .map((p) => p.id);

    // Reset player hand stats
    const players: Player[] = prevState.players.map((p, idx) => {
      const isAlive = p.cashBalance >= bigBlind;
      const cards: Card[] = isAlive ? [deck[idx * 2], deck[idx * 2 + 1]] : [];
      return {
        ...p,
        cards,
        status: isAlive ? 'active' : 'sitting-out',
        currentBet: 0,
        totalHandBet: 0,
        showCards: !p.isBot && isAlive,
        realizedHandPnl: 0,
        lastAction: undefined,
      };
    });

    const remainingDeck = deck.slice(players.length * 2);
    const markets = createInitialMarkets(players.map((p) => p.name));
    sound.playCardDeal();

    // Post Blinds among active players
    const activeIndices = players
      .map((p, i) => (p.status === 'active' ? i : -1))
      .filter((i) => i !== -1);

    const sbIndex = activeIndices[1 % activeIndices.length] ?? 0;
    const bbIndex = activeIndices[2 % activeIndices.length] ?? 0;

    const actualSB = Math.min(players[sbIndex].cashBalance, smallBlind);
    players[sbIndex].cashBalance -= actualSB;
    players[sbIndex].currentBet = actualSB;
    players[sbIndex].totalHandBet = actualSB;
    players[sbIndex].lastAction = `SB ${actualSB}`;

    const actualBB = Math.min(players[bbIndex].cashBalance, bigBlind);
    players[bbIndex].cashBalance -= actualBB;
    players[bbIndex].currentBet = actualBB;
    players[bbIndex].totalHandBet = actualBB;
    players[bbIndex].lastAction = `BB ${actualBB}`;

    const mainPot = actualSB + actualBB;
    const firstTurnIndex = activeIndices[(activeIndices.indexOf(bbIndex) + 1) % activeIndices.length] ?? 0;

    const events: GameEvent[] = [
      ...prevState.events.slice(-30),
      ...(levelUpEvent ? [levelUpEvent] : []),
      {
        id: `evt-hand-${handNum}`,
        timestamp: Date.now(),
        phase: 'DEALING',
        type: 'SYSTEM',
        text: `Hand #${handNum} (Level ${tournamentLevel}). Dealer: ${players[dealerIndex].name}. Blinds: ${smallBlind}/${bigBlind}. Dual Bounty: +${synergyMultiplierPct}%.`,
        highlight: true,
      },
    ];

    return {
      ...prevState,
      raiseCount: 0,
      handNumber: handNum,
      tournamentLevel,
      handsInCurrentLevel,
      handsPerLevel: HANDS_PER_LEVEL,
      synergyMultiplierPct,
      dealerIndex,
      currentTurnIndex: firstTurnIndex,
      phase: 'PREFLOP_TRADING',
      phaseTimeRemaining: 15,
      turnTimeRemaining: firstTurnIndex === 0 ? 15 : 2,
      totalTurnTime: firstTurnIndex === 0 ? 15 : 2,
      smallBlind,
      bigBlind,
      currentHighestBet: bigBlind,
      minRaise: bigBlind * 2,
      mainPot,
      deck: remainingDeck,
      communityCards: [],
      players,
      markets,
      positions: [],
      openOrders: [],
      recentTrades: [],
      events,
      eliminatedPlayerIds,
    };
  }

  // Fast-Forward a folded hand directly to Showdown and Settlement
  public static fastForwardHand(state: GameTableState): GameTableState {
    let s = { ...state };
    if (s.phase === 'WAITING' || s.phase === 'HAND_RESULTS' || s.phase === 'SHOWDOWN') {
      return s;
    }

    // Run board out to completion
    let maxSteps = 10;
    while (s.phase !== 'SHOWDOWN' && s.phase !== 'HAND_RESULTS' && maxSteps > 0) {
      maxSteps--;
      s = GameManager.advanceBoardPhase(s);
      if (s.phase === 'HAND_RESULTS' || s.phase === 'SHOWDOWN') break;
      // Advance to next betting/settlement
      if (s.phase === 'FLOP_TRADING') s.phase = 'FLOP_BETTING';
      else if (s.phase === 'TURN_TRADING') s.phase = 'TURN_BETTING';
      else if (s.phase === 'RIVER_TRADING') s.phase = 'RIVER_BETTING';
      s = GameManager.advanceBoardPhase(s);
    }

    if (s.phase !== 'HAND_RESULTS') {
      s = GameManager.advanceToSettlement(s);
    }
    return s;
  }

  // Execute a player or bot poker action
  public static handlePokerAction(
    state: GameTableState,
    playerId: string,
    action: 'CHECK' | 'CALL' | 'BET' | 'RAISE' | 'FOLD',
    amount: number = 0
  ): GameTableState {
    const s = { ...state, players: state.players.map((p) => ({ ...p })) };
    const playerIndex = s.players.findIndex((p) => p.id === playerId);
    if (playerIndex === -1) return state;

    const player = s.players[playerIndex];
    if (player.status !== 'active') return state;

    const callDiff = Math.max(0, s.currentHighestBet - player.currentBet);
    let actualBet = 0;

    if (action === 'CHECK') {
      sound.playKnock();
      player.lastAction = 'CHECK';
      s.events.push({
        id: `evt-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`,
        timestamp: Date.now(),
        phase: s.phase,
        type: 'POKER',
        text: `${player.name} checks.`,
      });
    } else if (action === 'CALL') {
      actualBet = Math.min(player.cashBalance, callDiff);
      player.cashBalance = Math.max(0, player.cashBalance - actualBet);
      player.currentBet += actualBet;
      player.totalHandBet += actualBet;
      s.mainPot += actualBet;
      player.lastAction = `CALL ${actualBet}`;
      sound.playChipBet();
      s.events.push({
        id: `evt-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`,
        timestamp: Date.now(),
        phase: s.phase,
        type: 'POKER',
        text: `${player.name} calls ${actualBet}.`,
      });
    } else if (action === 'BET' || action === 'RAISE') {
      // Raise limit check: if raiseCount is already >= 2, enforce CALL or CHECK
      if ((s.raiseCount || 0) >= 2) {
        actualBet = Math.min(player.cashBalance, callDiff);
        player.cashBalance = Math.max(0, player.cashBalance - actualBet);
        player.currentBet += actualBet;
        player.totalHandBet += actualBet;
        s.mainPot += actualBet;
        player.lastAction = `CALL ${actualBet}`;
        sound.playChipBet();
        s.events.push({
          id: `evt-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`,
          timestamp: Date.now(),
          phase: s.phase,
          type: 'POKER',
          text: `${player.name} calls ${actualBet} (raise cap reached).`,
        });
      } else {
        actualBet = Math.min(player.cashBalance, Math.max(s.currentHighestBet + s.bigBlind, amount));
        const increase = Math.max(0, actualBet - player.currentBet);
        player.cashBalance = Math.max(0, player.cashBalance - increase);
        player.currentBet = actualBet;
        player.totalHandBet += increase;
        s.mainPot += increase;
        s.currentHighestBet = actualBet;
        s.minRaise = actualBet + s.bigBlind;
        s.raiseCount = (s.raiseCount || 0) + 1;
        player.lastAction = `${action} ${actualBet}`;
        sound.playChipBet();
        s.events.push({
          id: `evt-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`,
          timestamp: Date.now(),
          phase: s.phase,
          type: 'POKER',
          text: `${player.name} ${action.toLowerCase()}s to ${actualBet}.`,
          highlight: true,
        });
      }
    } else if (action === 'FOLD') {
      player.status = 'folded';
      player.lastAction = 'FOLD';
      sound.playFold();
      s.events.push({
        id: `evt-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`,
        timestamp: Date.now(),
        phase: s.phase,
        type: 'POKER',
        text: `${player.name} folds.`,
      });
    }

    // Check if only 1 player remains active
    const activePlayers = s.players.filter((p) => p.status === 'active');
    if (activePlayers.length <= 1) {
      return GameManager.advanceToSettlement(s);
    }

    // Advance turn to next active player
    return GameManager.advancePokerTurn(s, playerIndex);
  }

  // Advance poker turn or advance phase if betting round is settled
  private static advancePokerTurn(state: GameTableState, lastPlayerIndex: number): GameTableState {
    const s = { ...state };
    const numPlayers = s.players.length;

    // Check remaining active players first
    const activePlayers = s.players.filter((p) => p.status === 'active');
    if (activePlayers.length <= 1) {
      return GameManager.advanceToSettlement(s);
    }

    // Active players who have chips remaining (not all-in)
    const activeWithChips = activePlayers.filter((p) => p.cashBalance > 0);

    // Condition to complete betting round:
    // Every active player is EITHER:
    // 1) all-in (cashBalance === 0)
    // 2) has matched currentHighestBet AND has taken an action (lastAction !== undefined)
    const allSatisfied = activePlayers.every((p) => {
      if (p.cashBalance === 0) return true; // all-in player cannot act further
      return p.currentBet === s.currentHighestBet && p.lastAction !== undefined;
    });

    if (allSatisfied) {
      // If at most 1 player has chips and all other active players are all-in, run out board to showdown
      if (activeWithChips.length <= 1) {
        return GameManager.fastForwardHand(s);
      }
      return GameManager.advanceBoardPhase(s);
    }

    // Find next active player who can act (status === 'active' and cashBalance > 0)
    let nextIndex = (lastPlayerIndex + 1) % numPlayers;
    let loopCount = 0;
    while ((s.players[nextIndex].status !== 'active' || s.players[nextIndex].cashBalance === 0) && loopCount < numPlayers) {
      nextIndex = (nextIndex + 1) % numPlayers;
      loopCount++;
    }

    // If no player with chips found, advance immediately
    if (loopCount >= numPlayers || activeWithChips.length <= 1) {
      return GameManager.fastForwardHand(s);
    }

    s.currentTurnIndex = nextIndex;
    s.turnTimeRemaining = nextIndex === 0 ? 15 : 2;
    s.totalTurnTime = nextIndex === 0 ? 15 : 2;

    if (nextIndex === 0) {
      sound.playTurnAlert();
    }

    return s;
  }

  // Advance board cards, resolve street-level markets, and generate fresh live questions
  public static advanceBoardPhase(state: GameTableState): GameTableState {
    const s: GameTableState = {
      ...state,
      raiseCount: 0,
      players: state.players.map((p): Player => ({ ...p, currentBet: 0, lastAction: undefined })),
      currentHighestBet: 0,
      minRaise: state.bigBlind,
    };

    const activePlayers = s.players.filter((p) => p.status === 'active');
    if (activePlayers.length <= 1) {
      return GameManager.advanceToSettlement(s);
    }

    const deck = [...s.deck];
    const community = [...s.communityCards];

    if (s.phase === 'PREFLOP_BETTING' || s.phase === 'PREFLOP_TRADING') {
      // Reveal Flop (3 cards)
      community.push(deck.pop()!, deck.pop()!, deck.pop()!);
      sound.playCardDeal();
      s.phase = 'FLOP_TRADING';
      s.communityCards = community;
      s.deck = deck;

      // Settle FLOP short-lived prediction markets immediately
      const streetRes = settleStreetMarkets('FLOP', community, s.markets, s.positions, s.players, s.handNumber);
      s.players = streetRes.updatedPlayers;
      s.markets = streetRes.updatedMarkets;
      s.positions = streetRes.updatedPositions;
      if (streetRes.ledgerEntries.length > 0) {
        s.ledger = [...s.ledger, ...streetRes.ledgerEntries];
      }

      if (streetRes.resolvedMarkets.length > 0) {
        sound.playLiveMarketResolve();
        streetRes.resolvedMarkets.forEach((rm) => {
          s.events.push({
            id: `evt-st-res-${Date.now()}-${rm.market.id}`,
            timestamp: Date.now(),
            phase: 'FLOP_REVEAL',
            type: 'MARKET',
            text: `⚡ Live Question Settled: ${rm.market.name} -> ${rm.outcome ? 'YES (1.00)' : 'NO (0.00)'} (Payout: $${rm.payoutTotal})`,
            highlight: true,
          });
        });
      }

      // Generate new short-lived live questions for the Turn
      const turnProps = generateStreetMarkets('TURN', community, s.handNumber);
      s.markets = [...s.markets, ...turnProps];

      s.events.push({
        id: `evt-${Date.now()}`,
        timestamp: Date.now(),
        phase: 'FLOP_REVEAL',
        type: 'SYSTEM',
        text: `Flop dealt: ${community.map((c) => `${c.rank}${c.suit[0].toUpperCase()}`).join(' ')} • New Turn live bets active!`,
        highlight: true,
      });
    } else if (s.phase === 'FLOP_BETTING' || s.phase === 'FLOP_TRADING') {
      // Reveal Turn (1 card)
      community.push(deck.pop()!);
      sound.playCardDeal();
      s.phase = 'TURN_TRADING';
      s.communityCards = community;
      s.deck = deck;

      // Settle TURN short-lived prediction markets
      const streetRes = settleStreetMarkets('TURN', community, s.markets, s.positions, s.players, s.handNumber);
      s.players = streetRes.updatedPlayers;
      s.markets = streetRes.updatedMarkets;
      s.positions = streetRes.updatedPositions;
      if (streetRes.ledgerEntries.length > 0) {
        s.ledger = [...s.ledger, ...streetRes.ledgerEntries];
      }

      if (streetRes.resolvedMarkets.length > 0) {
        sound.playLiveMarketResolve();
        streetRes.resolvedMarkets.forEach((rm) => {
          s.events.push({
            id: `evt-st-res-${Date.now()}-${rm.market.id}`,
            timestamp: Date.now(),
            phase: 'TURN_REVEAL',
            type: 'MARKET',
            text: `⚡ Live Question Settled: ${rm.market.name} -> ${rm.outcome ? 'YES (1.00)' : 'NO (0.00)'} (Payout: $${rm.payoutTotal})`,
            highlight: true,
          });
        });
      }

      // Generate new short-lived live questions for the River
      const riverProps = generateStreetMarkets('RIVER', community, s.handNumber);
      s.markets = [...s.markets, ...riverProps];

      s.events.push({
        id: `evt-${Date.now()}`,
        timestamp: Date.now(),
        phase: 'TURN_REVEAL',
        type: 'SYSTEM',
        text: `Turn dealt: ${community[3].rank}${community[3].suit[0].toUpperCase()} • River live markets open!`,
        highlight: true,
      });
    } else if (s.phase === 'TURN_BETTING' || s.phase === 'TURN_TRADING') {
      // Reveal River (1 card)
      community.push(deck.pop()!);
      sound.playCardDeal();
      s.phase = 'RIVER_TRADING';
      s.communityCards = community;
      s.deck = deck;

      // Settle RIVER short-lived prediction markets
      const streetRes = settleStreetMarkets('RIVER', community, s.markets, s.positions, s.players, s.handNumber);
      s.players = streetRes.updatedPlayers;
      s.markets = streetRes.updatedMarkets;
      s.positions = streetRes.updatedPositions;
      if (streetRes.ledgerEntries.length > 0) {
        s.ledger = [...s.ledger, ...streetRes.ledgerEntries];
      }

      if (streetRes.resolvedMarkets.length > 0) {
        sound.playLiveMarketResolve();
        streetRes.resolvedMarkets.forEach((rm) => {
          s.events.push({
            id: `evt-st-res-${Date.now()}-${rm.market.id}`,
            timestamp: Date.now(),
            phase: 'RIVER_REVEAL',
            type: 'MARKET',
            text: `⚡ Live Question Settled: ${rm.market.name} -> ${rm.outcome ? 'YES (1.00)' : 'NO (0.00)'} (Payout: $${rm.payoutTotal})`,
            highlight: true,
          });
        });
      }

      s.events.push({
        id: `evt-${Date.now()}`,
        timestamp: Date.now(),
        phase: 'RIVER_REVEAL',
        type: 'SYSTEM',
        text: `River dealt: ${community[4].rank}${community[4].suit[0].toUpperCase()} • Final showdown round!`,
        highlight: true,
      });
    } else if (s.phase === 'RIVER_BETTING' || s.phase === 'RIVER_TRADING') {
      // Proceed to Showdown & Settlement
      return GameManager.advanceToSettlement(s);
    }

    // Set first active player after dealer
    const dealerIndex = s.dealerIndex;
    let nextTurn = (dealerIndex + 1) % s.players.length;
    let loopCount = 0;
    while ((s.players[nextTurn].status !== 'active' || s.players[nextTurn].cashBalance === 0) && loopCount < s.players.length) {
      nextTurn = (nextTurn + 1) % s.players.length;
      loopCount++;
    }

    if (loopCount >= s.players.length) {
      return GameManager.fastForwardHand(s);
    }

    s.currentTurnIndex = nextTurn;
    s.turnTimeRemaining = nextTurn === 0 ? 15 : 2;
    s.totalTurnTime = nextTurn === 0 ? 15 : 2;

    if (nextTurn === 0 && s.phase.endsWith('_BETTING')) {
      sound.playTurnAlert();
    }

    return s;
  }

  // Settle showdown, calculate Dual Synergy Bounty funded by losing players, and finalize hand
  public static advanceToSettlement(state: GameTableState): GameTableState {
    const s = { ...state };
    s.phase = 'SHOWDOWN';

    // Reveal cards for showdown
    s.players = s.players.map((p) => ({
      ...p,
      showCards: p.status !== 'folded',
    }));

    const degenPlayer = s.players.find((p) => p.botPersonality === 'degen');
    const degenWentAllIn = degenPlayer ? degenPlayer.cashBalance === 0 && degenPlayer.totalHandBet > 0 : false;

    const settlement = settleHandAndMarkets(
      s.handNumber,
      s.tournamentLevel,
      s.synergyMultiplierPct,
      s.players,
      s.communityCards,
      s.markets,
      s.positions,
      s.mainPot,
      degenWentAllIn
    );

    s.players = settlement.updatedPlayers;
    s.markets = settlement.updatedMarkets;
    s.positions = settlement.updatedPositions;
    s.lastHandAnalytics = settlement.analytics;
    s.ledger = [...s.ledger, ...settlement.ledgerEntries];
    s.phase = 'HAND_RESULTS';

    sound.playWin();

    // Check if Dual Synergy Multiplier was triggered
    if (settlement.analytics.dualSynergyBonus?.triggered) {
      sound.playSynergyJackpot();
      const bonus = settlement.analytics.dualSynergyBonus;
      s.events.push({
        id: `evt-synergy-${Date.now()}`,
        timestamp: Date.now(),
        phase: 'SHOWDOWN',
        type: 'SYNERGY',
        text: `🔥 DUAL VICTORY SYNERGY TRIGGERED! ${bonus.winningPlayerName} won Poker + Prediction Markets! Level ${s.tournamentLevel} Multiplier (+${bonus.multiplierPct}%) awarded: $${bonus.bonusAmount} extracted from losing players!`,
        highlight: true,
      });
    }

    const winnerNames = settlement.analytics.winners.map((w) => `${w.name} (${w.handDescription})`).join(', ');
    s.events.push({
      id: `evt-win-${Date.now()}`,
      timestamp: Date.now(),
      phase: 'SHOWDOWN',
      type: 'SYSTEM',
      text: `Hand #${s.handNumber} Showdown! Winner: ${winnerNames}. Pot: ${s.mainPot} credits.`,
      highlight: true,
    });

    return s;
  }

  // Execute User or Bot Market Order
  public static placeMarketOrder(
    state: GameTableState,
    playerId: string,
    marketId: string,
    side: OrderSide,
    type: OrderType,
    price: number,
    quantity: number
  ): GameTableState {
    const s = {
      ...state,
      players: state.players.map((p) => ({ ...p })),
      markets: state.markets.map((m) => ({ ...m })),
      positions: state.positions.map((pos) => ({ ...pos })),
      recentTrades: [...state.recentTrades],
    };

    const player = s.players.find((p) => p.id === playerId);
    const market = s.markets.find((m) => m.id === marketId);
    if (!player || !market || market.resolved || quantity <= 0) return state;

    // Check cash limits
    const estimatedCost = side === 'BUY_YES' ? quantity * price : quantity * (1 - price);
    if (player.cashBalance < estimatedCost && (side === 'BUY_YES' || side === 'BUY_NO')) {
      return state;
    }

    const order: Order = {
      id: `ord-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      marketId,
      playerId,
      playerName: player.name,
      side,
      type,
      price,
      quantity,
      filledQuantity: 0,
      status: 'OPEN',
      createdAt: Date.now(),
    };

    const execution = executeOrder(market, order, s.phase);

    // Update market in state
    const mIdx = s.markets.findIndex((m) => m.id === marketId);
    s.markets[mIdx] = execution.updatedMarket;

    // Debit or credit cash balance
    if (execution.costDebit > 0) {
      player.cashBalance = Math.max(0, Number((player.cashBalance - execution.costDebit).toFixed(2)));
    } else if (execution.costDebit < 0) {
      player.cashBalance = Number((player.cashBalance - execution.costDebit).toFixed(2));
    }

    // Update position
    const totalFilled = execution.trades.reduce((sum, t) => sum + t.quantity, 0);
    if (totalFilled > 0) {
      const avgPrice = execution.trades.reduce((sum, t) => sum + t.price * t.quantity, 0) / totalFilled;
      const posIdx = s.positions.findIndex((pos) => pos.playerId === playerId && pos.marketId === marketId);
      const existingPos = posIdx >= 0 ? s.positions[posIdx] : undefined;
      const updatedPos = updatePosition(existingPos, marketId, playerId, side, avgPrice, totalFilled);

      if (posIdx >= 0) {
        s.positions[posIdx] = updatedPos;
      } else {
        s.positions.push(updatedPos);
      }

      s.recentTrades.push(...execution.trades);
      sound.playOrderFill();

      s.events.push({
        id: `evt-trade-${Date.now()}`,
        timestamp: Date.now(),
        phase: s.phase,
        type: 'MARKET',
        text: `${player.name} traded ${totalFilled} ${side} on ${market.ticker} @ ${avgPrice.toFixed(2)}`,
        highlight: player.id === 'player-1',
      });
    }

    return s;
  }

  // 1-Click Fast Live Bet (Kalshi / Polymarket Style)
  public static quickBet(
    state: GameTableState,
    playerId: string,
    marketId: string,
    side: 'BUY_YES' | 'BUY_NO',
    dollars: number = 50
  ): GameTableState {
    const market = state.markets.find((m) => m.id === marketId);
    const player = state.players.find((p) => p.id === playerId);
    if (!market || !player || market.resolved) return state;

    const price = side === 'BUY_YES' ? market.currentYesPrice : market.currentNoPrice;
    if (price <= 0) return state;

    const quantity = Math.max(1, Math.floor(dollars / price));
    sound.playQuickBet();

    return GameManager.placeMarketOrder(
      state,
      playerId,
      marketId,
      side,
      'MARKET',
      price,
      quantity
    );
  }

  // Execute bot turn or bot trading window actions
  public static processBotActions(state: GameTableState): GameTableState {
    let s = { ...state };

    // 1. If currently in a TRADING phase, bots consider trading opportunities
    if (s.phase.endsWith('_TRADING')) {
      for (const bot of s.players.filter((p) => p.isBot && p.status !== 'folded')) {
        const obs: BotObservation = {
          botPlayer: bot,
          communityCards: s.communityCards,
          phase: s.phase,
          currentHighestBet: s.currentHighestBet,
          minRaise: s.minRaise,
          mainPot: s.mainPot,
          markets: s.markets,
          positions: s.positions,
          activePlayerCount: s.players.filter((p) => p.status === 'active').length,
        };

        const tradeDecision = BotEngine.decideMarketOrders(obs);
        for (const ord of tradeDecision.orders) {
          s = GameManager.placeMarketOrder(s, bot.id, ord.marketId, ord.side, ord.type, ord.price, ord.quantity);
        }
      }
    }

    // 2. If it is a bot's turn during a BETTING phase, execute their poker action
    const currentTurnPlayer = s.players[s.currentTurnIndex];
    if (currentTurnPlayer && currentTurnPlayer.isBot && currentTurnPlayer.status === 'active' && s.phase.endsWith('_BETTING')) {
      const obs: BotObservation = {
        botPlayer: currentTurnPlayer,
        communityCards: s.communityCards,
        phase: s.phase,
        currentHighestBet: s.currentHighestBet,
        minRaise: s.minRaise,
        mainPot: s.mainPot,
        markets: s.markets,
        positions: s.positions,
        activePlayerCount: s.players.filter((p) => p.status === 'active').length,
        raiseCount: s.raiseCount || 0,
      };

      const pokerDecision = BotEngine.decidePokerAction(obs);
      s = GameManager.handlePokerAction(s, currentTurnPlayer.id, pokerDecision.action, pokerDecision.amount);
    }

    return s;
  }

  // Borrow emergency liquidity / loan from the market protocol at 4% interest rate
  public static borrowMarketLoan(
    state: GameTableState,
    playerId: string,
    amount: number
  ): GameTableState {
    if (amount <= 0) return state;

    const players = state.players.map((p) => {
      if (p.id === playerId) {
        const newBalance = Number((p.cashBalance + amount).toFixed(2));
        const newLoan = Number(((p.loanBalance || 0) + amount).toFixed(2));
        return {
          ...p,
          cashBalance: newBalance,
          loanBalance: newLoan,
          status: p.status === 'sitting-out' ? ('active' as const) : p.status,
        };
      }
      return p;
    });

    const borrowingPlayer = players.find((p) => p.id === playerId);
    const balanceAfter = borrowingPlayer ? borrowingPlayer.cashBalance : amount;

    const newLedgerEntry = {
      id: `led-loan-borrow-${Date.now()}`,
      timestamp: Date.now(),
      handNumber: state.handNumber,
      playerId,
      category: 'MARKET_LOAN_BORROW' as const,
      amount,
      balanceAfter,
      description: `🏦 Borrowed from Market Protocol: +${amount.toLocaleString()} (4% Interest auto-paid on wins)`,
    };

    const newEvent: GameEvent = {
      id: `evt-loan-borrow-${Date.now()}`,
      timestamp: Date.now(),
      phase: state.phase,
      type: 'SYSTEM',
      text: `🏦 ${borrowingPlayer?.name || 'Player'} borrowed +${amount.toLocaleString()} from Market Protocol at 4% interest.`,
      highlight: true,
    };

    sound.playChipBet();

    return {
      ...state,
      players,
      eliminatedPlayerIds: state.eliminatedPlayerIds.filter((id) => id !== playerId),
      ledger: [newLedgerEntry, ...state.ledger],
      events: [newEvent, ...state.events],
    };
  }

  // Voluntarily repay market liquidity loan principal
  public static repayMarketLoan(
    state: GameTableState,
    playerId: string,
    amount: number
  ): GameTableState {
    if (amount <= 0) return state;
    const targetPlayer = state.players.find((p) => p.id === playerId);
    if (!targetPlayer || !targetPlayer.loanBalance || targetPlayer.loanBalance <= 0) return state;

    const repayAmt = Math.min(amount, targetPlayer.cashBalance, targetPlayer.loanBalance);
    if (repayAmt <= 0) return state;

    const players = state.players.map((p) => {
      if (p.id === playerId) {
        return {
          ...p,
          cashBalance: Number((p.cashBalance - repayAmt).toFixed(2)),
          loanBalance: Number(((p.loanBalance || 0) - repayAmt).toFixed(2)),
        };
      }
      return p;
    });

    const updatedPlayer = players.find((p) => p.id === playerId);

    const newLedgerEntry = {
      id: `led-loan-repay-${Date.now()}`,
      timestamp: Date.now(),
      handNumber: state.handNumber,
      playerId,
      category: 'MARKET_LOAN_REPAY' as const,
      amount: -repayAmt,
      balanceAfter: updatedPlayer?.cashBalance || 0,
      description: `🏦 Repaid Market Loan Principal: -${repayAmt.toLocaleString()} (Remaining Debt: ${updatedPlayer?.loanBalance?.toLocaleString() || 0})`,
    };

    const newEvent: GameEvent = {
      id: `evt-loan-repay-${Date.now()}`,
      timestamp: Date.now(),
      phase: state.phase,
      type: 'SYSTEM',
      text: `🏦 ${updatedPlayer?.name || 'Player'} repaid ${repayAmt.toLocaleString()} of market loan.`,
    };

    sound.playChipBet();

    return {
      ...state,
      players,
      ledger: [newLedgerEntry, ...state.ledger],
      events: [newEvent, ...state.events],
    };
  }
}
