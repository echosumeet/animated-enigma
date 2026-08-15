import {
  Card,
  DualSynergyBonus,
  HandCategory,
  HandEvaluation,
  LedgerEntry,
  Market,
  Player,
  Position,
  PostHandAnalytics,
} from '../types/game';
import { evaluateHand } from './poker';

export interface SettlementResult {
  updatedPlayers: Player[];
  updatedMarkets: Market[];
  updatedPositions: Position[];
  analytics: PostHandAnalytics;
  ledgerEntries: LedgerEntry[];
}

export interface StreetSettlementResult {
  updatedPlayers: Player[];
  updatedMarkets: Market[];
  updatedPositions: Position[];
  resolvedMarkets: { market: Market; outcome: boolean; payoutTotal: number }[];
  ledgerEntries: LedgerEntry[];
}

// Settle intermediate street-level propositions (e.g. FLOP_REVEAL, TURN_REVEAL, RIVER_REVEAL)
export function settleStreetMarkets(
  street: 'FLOP' | 'TURN' | 'RIVER',
  communityCards: Card[],
  markets: Market[],
  positions: Position[],
  players: Player[],
  handNumber: number
): StreetSettlementResult {
  const updatedPlayers = players.map((p) => ({ ...p }));
  const updatedMarkets = markets.map((m) => ({ ...m }));
  const updatedPositions = positions.map((pos) => ({ ...pos }));
  const resolvedMarkets: { market: Market; outcome: boolean; payoutTotal: number }[] = [];
  const ledgerEntries: LedgerEntry[] = [];

  for (const m of updatedMarkets) {
    if (m.resolved || m.lifespan !== 'SHORT_LIVED' || m.expiryStreet !== street) {
      continue;
    }

    let outcome = false;
    let resolutionDesc = '';

    if (m.ticker === 'FLOP_ACE_KING' && street === 'FLOP') {
      const flop = communityCards.slice(0, 3);
      outcome = flop.some((c) => c.rank === 'A' || c.rank === 'K');
      resolutionDesc = outcome
        ? `YES: Flop contains ${flop.filter((c) => c.rank === 'A' || c.rank === 'K').map((c) => c.rank).join(', ')}`
        : 'NO: No Ace or King in Flop';
    } else if (m.ticker === 'NEXT_CARD_RED' && street === 'FLOP') {
      const firstCard = communityCards[0];
      outcome = firstCard ? firstCard.suit === 'hearts' || firstCard.suit === 'diamonds' : false;
      resolutionDesc = outcome
        ? `YES: 1st Flop card is Red (${firstCard?.suit})`
        : `NO: 1st Flop card is Black (${firstCard?.suit})`;
    } else if (m.ticker === 'TURN_SPADE' && street === 'TURN') {
      const turnCard = communityCards[3];
      outcome = turnCard ? turnCard.suit === 'spades' : false;
      resolutionDesc = outcome ? `YES: Turn card is Spade (${turnCard?.rank}♠)` : `NO: Turn is ${turnCard?.suit}`;
    } else if (m.ticker === 'TURN_HIGH_CARD' && street === 'TURN') {
      const turnCard = communityCards[3];
      const highRanks = ['10', 'J', 'Q', 'K', 'A'];
      outcome = turnCard ? highRanks.includes(turnCard.rank) : false;
      resolutionDesc = outcome ? `YES: Turn is High Card (${turnCard?.rank})` : `NO: Turn is Low Card (${turnCard?.rank})`;
    } else if (m.ticker === 'RIVER_BLACK' && street === 'RIVER') {
      const riverCard = communityCards[4];
      outcome = riverCard ? riverCard.suit === 'spades' || riverCard.suit === 'clubs' : false;
      resolutionDesc = outcome ? `YES: River is Black (${riverCard?.suit})` : `NO: River is Red (${riverCard?.suit})`;
    } else if (m.ticker === 'RIVER_PAIRS_BOARD' && street === 'RIVER') {
      const riverCard = communityCards[4];
      const priorCards = communityCards.slice(0, 4);
      outcome = riverCard ? priorCards.some((c) => c.rank === riverCard.rank) : false;
      resolutionDesc = outcome
        ? `YES: River (${riverCard?.rank}) paired board!`
        : `NO: River (${riverCard?.rank}) did not pair board`;
    }

    m.resolved = true;
    m.outcome = outcome;
    m.currentYesPrice = outcome ? 1.0 : 0.0;
    m.currentNoPrice = outcome ? 0.0 : 1.0;
    m.resolutionText = resolutionDesc;

    // Settle positions for this street market
    let marketPayoutTotal = 0;
    for (const pos of updatedPositions) {
      if (pos.marketId !== m.id) continue;
      const player = updatedPlayers.find((p) => p.id === pos.playerId);
      if (!player) continue;

      let payoutCash = 0;
      let netPnl = pos.realizedPnl;

      if (outcome === true) {
        if (pos.yesContracts > 0) {
          payoutCash = pos.yesContracts * 1.0;
          netPnl += payoutCash - pos.yesContracts * pos.avgYesPrice;
        }
        if (pos.noContracts > 0) {
          netPnl -= pos.noContracts * pos.avgNoPrice;
        }
      } else {
        if (pos.noContracts > 0) {
          payoutCash = pos.noContracts * 1.0;
          netPnl += payoutCash - pos.noContracts * pos.avgNoPrice;
        }
        if (pos.yesContracts > 0) {
          netPnl -= pos.yesContracts * pos.avgYesPrice;
        }
      }

      if (payoutCash > 0) {
        player.cashBalance += payoutCash;
        marketPayoutTotal += payoutCash;
        ledgerEntries.push({
          id: `ledger-street-${m.id}-${player.id}-${Date.now()}`,
          timestamp: Date.now(),
          handNumber,
          playerId: player.id,
          category: 'MARKET_SETTLE',
          amount: payoutCash,
          balanceAfter: Number(player.cashBalance.toFixed(2)),
          description: `Live Bet Payout: ${m.name} (${outcome ? 'YES' : 'NO'})`,
        });
      }

      pos.yesContracts = 0;
      pos.noContracts = 0;
      pos.realizedPnl = Number(netPnl.toFixed(2));
    }

    resolvedMarkets.push({ market: m, outcome, payoutTotal: marketPayoutTotal });
  }

  return {
    updatedPlayers,
    updatedMarkets,
    updatedPositions,
    resolvedMarkets,
    ledgerEntries,
  };
}

export function settleHandAndMarkets(
  handNumber: number,
  tournamentLevel: number,
  synergyMultiplierPct: number,
  players: Player[],
  communityCards: Card[],
  markets: Market[],
  positions: Position[],
  pot: number,
  degenWentAllIn: boolean = false
): SettlementResult {
  const updatedPlayers = players.map((p) => ({ ...p }));
  const updatedMarkets = markets.map((m) => ({ ...m }));
  const updatedPositions = positions.map((pos) => ({ ...pos }));
  const ledgerEntries: LedgerEntry[] = [];

  // 1. Determine Poker Winners
  const activePlayers = updatedPlayers.filter((p) => p.status !== 'folded');
  const evaluations: { player: Player; evaluation: HandEvaluation }[] = [];

  for (const p of activePlayers) {
    const evaluation = evaluateHand(p.cards, communityCards);
    evaluations.push({ player: p, evaluation });
  }

  // Sort descending by rank score
  evaluations.sort((a, b) => b.evaluation.rankScore - a.evaluation.rankScore);

  const winners: { playerId: string; name: string; amount: number; handDescription: string }[] = [];
  let winningCategory: HandCategory = 'High Card';

  if (evaluations.length > 0) {
    const topScore = evaluations[0].evaluation.rankScore;
    winningCategory = evaluations[0].evaluation.category;
    const topWinners = evaluations.filter((e) => e.evaluation.rankScore === topScore);
    const splitAmount = Number((pot / topWinners.length).toFixed(2));

    for (const w of topWinners) {
      const p = updatedPlayers.find((pl) => pl.id === w.player.id);
      if (p) {
        p.cashBalance += splitAmount;
        p.realizedHandPnl += splitAmount - p.totalHandBet;
        ledgerEntries.push({
          id: `ledger-poker-win-${p.id}-${Date.now()}`,
          timestamp: Date.now(),
          handNumber,
          playerId: p.id,
          category: 'POKER_WIN',
          amount: splitAmount,
          balanceAfter: Number(p.cashBalance.toFixed(2)),
          description: `Poker Pot Won: ${w.evaluation.description}`,
        });
      }
      winners.push({
        playerId: w.player.id,
        name: w.player.name,
        amount: splitAmount,
        handDescription: w.evaluation.description,
      });
    }
  } else {
    // If only 1 player remained active
    const lastPlayer = updatedPlayers.find((p) => p.status !== 'folded') || updatedPlayers[0];
    lastPlayer.cashBalance += pot;
    lastPlayer.realizedHandPnl += pot - lastPlayer.totalHandBet;
    winners.push({
      playerId: lastPlayer.id,
      name: lastPlayer.name,
      amount: pot,
      handDescription: 'Opponents Folded',
    });
    ledgerEntries.push({
      id: `ledger-poker-win-${lastPlayer.id}-${Date.now()}`,
      timestamp: Date.now(),
      handNumber,
      playerId: lastPlayer.id,
      category: 'POKER_WIN',
      amount: pot,
      balanceAfter: Number(lastPlayer.cashBalance.toFixed(2)),
      description: 'Poker Pot Won (Opponents Folded)',
    });
  }

  const winningPlayerIds = new Set(winners.map((w) => w.playerId));

  // 2. Resolve Remaining Prediction Markets (Showdown, Hand Types, Macro Props)
  const marketOutcomes: {
    marketId: string;
    marketName: string;
    outcome: boolean;
    settlementPrice: number;
    lifespan?: any;
  }[] = [];

  for (const m of updatedMarkets) {
    if (m.resolved) {
      marketOutcomes.push({
        marketId: m.id,
        marketName: m.name,
        outcome: !!m.outcome,
        settlementPrice: m.outcome ? 1.0 : 0.0,
        lifespan: m.lifespan,
      });
      continue;
    }

    let outcome = false;

    if (m.category === 'PLAYER_WIN') {
      outcome = m.targetPlayerId ? winningPlayerIds.has(m.targetPlayerId) : false;
    } else if (m.category === 'HAND_TYPE' && m.ticker === 'STRAIGHT_PLUS') {
      const straightOrBetter: HandCategory[] = [
        'Straight',
        'Flush',
        'Full House',
        'Four of a Kind',
        'Straight Flush',
        'Royal Flush',
      ];
      outcome = straightOrBetter.includes(winningCategory);
    } else if (m.category === 'BOARD_EVENT' && m.ticker === 'BOARD_PAIRS') {
      const ranks = communityCards.map((c) => c.rank);
      const uniqueRanks = new Set(ranks);
      outcome = communityCards.length >= 5 && uniqueRanks.size < 5;
    } else if (m.ticker === 'POT_OVER_2K') {
      outcome = pot >= 2000;
    } else if (m.ticker === 'REX_ALL_IN') {
      outcome = degenWentAllIn;
    } else if (m.ticker === 'YOU_TOP_2') {
      // Tournament prop survives while user is alive
      const user = updatedPlayers.find((p) => p.id === 'player-1');
      outcome = user ? user.cashBalance > 0 : false;
    }

    m.resolved = true;
    m.outcome = outcome;
    m.currentYesPrice = outcome ? 1.0 : 0.0;
    m.currentNoPrice = outcome ? 0.0 : 1.0;

    marketOutcomes.push({
      marketId: m.id,
      marketName: m.name,
      outcome,
      settlementPrice: outcome ? 1.0 : 0.0,
      lifespan: m.lifespan,
    });
  }

  // 3. Settle Remaining Market Positions & Payouts
  let userMarketPnl = 0;
  const user = updatedPlayers.find((p) => p.id === 'player-1') || updatedPlayers[0];
  const playerProfitablePositions: Record<string, number> = {};

  for (const pos of updatedPositions) {
    const market = updatedMarkets.find((m) => m.id === pos.marketId);
    if (!market || !market.resolved) continue;

    const player = updatedPlayers.find((p) => p.id === pos.playerId);
    if (!player) continue;

    let settlementCash = 0;
    let netPositionPnl = pos.realizedPnl;

    if (market.outcome === true) {
      if (pos.yesContracts > 0) {
        const payout = pos.yesContracts * 1.0;
        const cost = pos.yesContracts * pos.avgYesPrice;
        settlementCash += payout;
        netPositionPnl += payout - cost;
        playerProfitablePositions[player.id] = (playerProfitablePositions[player.id] || 0) + pos.yesContracts;
      }
      if (pos.noContracts > 0) {
        const cost = pos.noContracts * pos.avgNoPrice;
        netPositionPnl -= cost;
      }
    } else {
      if (pos.noContracts > 0) {
        const payout = pos.noContracts * 1.0;
        const cost = pos.noContracts * pos.avgNoPrice;
        settlementCash += payout;
        netPositionPnl += payout - cost;
        playerProfitablePositions[player.id] = (playerProfitablePositions[player.id] || 0) + pos.noContracts;
      }
      if (pos.yesContracts > 0) {
        const cost = pos.yesContracts * pos.avgYesPrice;
        netPositionPnl -= cost;
      }
    }

    // Credit player's cash balance with settlement payout
    if (settlementCash > 0) {
      player.cashBalance += settlementCash;
      ledgerEntries.push({
        id: `ledger-market-win-${market.id}-${player.id}-${Date.now()}`,
        timestamp: Date.now(),
        handNumber,
        playerId: player.id,
        category: 'MARKET_SETTLE',
        amount: settlementCash,
        balanceAfter: Number(player.cashBalance.toFixed(2)),
        description: `Market Settlement: ${market.name} (${market.outcome ? 'YES' : 'NO'})`,
      });
    }

    pos.yesContracts = 0;
    pos.noContracts = 0;
    pos.realizedPnl = Number(netPositionPnl.toFixed(2));

    if (player.id === user.id) {
      userMarketPnl += Number(netPositionPnl.toFixed(2));
    }
  }

  // 4. Calculate DUAL SYNERGY BOUNTY (Poker Win + Prediction Market Win) funded by Losing Players!
  let dualSynergyBonus: DualSynergyBonus | undefined;
  const primaryWinner = winners[0];

  if (primaryWinner) {
    const winningPlayer = updatedPlayers.find((p) => p.id === primaryWinner.playerId);
    const contractsWon = winningPlayer ? playerProfitablePositions[winningPlayer.id] || 0 : 0;

    // Triggered if the poker winner also won live prediction contracts or held positive market PnL
    if (winningPlayer && contractsWon > 0) {
      const rawBonus = Math.round(pot * (synergyMultiplierPct / 100));
      const losingPlayers = updatedPlayers.filter((p) => p.id !== winningPlayer.id && p.cashBalance > 0);

      if (rawBonus > 0 && losingPlayers.length > 0) {
        let collectedBonus = 0;
        const fundedBy: { playerId: string; name: string; amount: number }[] = [];
        const perPlayerShare = Math.ceil(rawBonus / losingPlayers.length);

        for (const lp of losingPlayers) {
          const deduction = Math.min(lp.cashBalance, perPlayerShare);
          if (deduction > 0) {
            lp.cashBalance -= deduction;
            collectedBonus += deduction;
            fundedBy.push({ playerId: lp.id, name: lp.name, amount: deduction });

            ledgerEntries.push({
              id: `ledger-synergy-penalty-${lp.id}-${Date.now()}`,
              timestamp: Date.now(),
              handNumber,
              playerId: lp.id,
              category: 'SYNERGY_PENALTY',
              amount: -deduction,
              balanceAfter: Number(lp.cashBalance.toFixed(2)),
              description: `Dual Synergy Bounty Penalty paid to ${winningPlayer.name} (+${synergyMultiplierPct}%)`,
            });
          }
        }

        if (collectedBonus > 0) {
          winningPlayer.cashBalance += collectedBonus;
          winningPlayer.realizedHandPnl += collectedBonus;

          ledgerEntries.push({
            id: `ledger-synergy-bounty-${winningPlayer.id}-${Date.now()}`,
            timestamp: Date.now(),
            handNumber,
            playerId: winningPlayer.id,
            category: 'SYNERGY_BOUNTY',
            amount: collectedBonus,
            balanceAfter: Number(winningPlayer.cashBalance.toFixed(2)),
            description: `🔥 DUAL SYNERGY BOUNTY: +${synergyMultiplierPct}% Level Multiplier (${winningPlayer.name} won Poker + Markets)`,
          });

          dualSynergyBonus = {
            triggered: true,
            multiplierPct: synergyMultiplierPct,
            bonusAmount: collectedBonus,
            winningPlayerId: winningPlayer.id,
            winningPlayerName: winningPlayer.name,
            winningContractsCount: contractsWon,
            fundedBy,
          };
        }
      }
    }
  }

  // 5. AUTO-PAY 4% MARKET LOAN INTEREST ON HAND WINNINGS
  let userLoanInterestDeducted = 0;
  for (const p of updatedPlayers) {
    if (p.loanBalance && p.loanBalance > 0) {
      // Determine if player won any payout in this hand
      const wonPoker = winners.find((w) => w.playerId === p.id)?.amount || 0;
      const wonSynergy = dualSynergyBonus?.winningPlayerId === p.id ? dualSynergyBonus.bonusAmount : 0;
      const wonMarket = p.id === user.id && userMarketPnl > 0 ? userMarketPnl : 0;
      const totalWinningsThisHand = wonPoker + wonSynergy + wonMarket;

      if (totalWinningsThisHand > 0) {
        const interestDue = Math.max(1, Math.round(p.loanBalance * 0.04));
        const interestToDeduct = Math.min(p.cashBalance, Math.min(totalWinningsThisHand, interestDue));

        if (interestToDeduct > 0) {
          p.cashBalance = Number((p.cashBalance - interestToDeduct).toFixed(2));
          p.totalInterestPaid = (p.totalInterestPaid || 0) + interestToDeduct;

          ledgerEntries.push({
            id: `ledger-loan-interest-${p.id}-${Date.now()}`,
            timestamp: Date.now(),
            handNumber,
            playerId: p.id,
            category: 'MARKET_LOAN_INTEREST',
            amount: -interestToDeduct,
            balanceAfter: Number(p.cashBalance.toFixed(2)),
            description: `💸 Auto-Paid 4% Market Interest: -${interestToDeduct} (Active Loan: ${p.loanBalance.toLocaleString()})`,
          });

          if (p.id === user.id) {
            userLoanInterestDeducted = interestToDeduct;
          }
        }
      }
    }
  }

  // 6. Calculate Analytics
  const userPokerPnl = winners.some((w) => w.playerId === user.id)
    ? winners.find((w) => w.playerId === user.id)!.amount - user.totalHandBet
    : -user.totalHandBet;

  const userSynergyPnl = dualSynergyBonus?.winningPlayerId === user.id ? dualSynergyBonus.bonusAmount : 0;
  const userTotalPnl = Number((userPokerPnl + userMarketPnl + userSynergyPnl - userLoanInterestDeducted).toFixed(2));
  const userEndingEquity = Number(user.cashBalance.toFixed(2));

  const analytics: PostHandAnalytics = {
    handNumber,
    tournamentLevel,
    synergyMultiplierPct,
    winners,
    pokerPot: pot,
    userPokerPnl,
    userMarketPnl,
    userTotalPnl,
    userEndingEquity,
    loanInterestDeducted: userLoanInterestDeducted,
    loanBalance: user.loanBalance,
    dualSynergyBonus,
    marketOutcomes,
    skillDeltas: [
      { category: 'cardSkill', delta: userPokerPnl > 0 ? +3 : -1 },
      { category: 'marketSkill', delta: userMarketPnl > 0 ? +4 : -1 },
      { category: 'forecastingSkill', delta: userTotalPnl > 0 ? +3 : -1 },
      { category: 'riskSkill', delta: userEndingEquity > 9500 ? +2 : -2 },
      { category: 'psychologySkill', delta: dualSynergyBonus?.triggered ? +5 : +1 },
    ],
  };

  return {
    updatedPlayers,
    updatedMarkets,
    updatedPositions,
    analytics,
    ledgerEntries,
  };
}
