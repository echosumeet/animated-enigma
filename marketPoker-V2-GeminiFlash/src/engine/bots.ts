import {
  Card,
  GamePhase,
  Market,
  Order,
  OrderSide,
  Player,
  Position,
} from '../types/game';
import { estimatePokerEquity } from './poker';

export interface BotObservation {
  botPlayer: Player;
  communityCards: Card[];
  phase: GamePhase;
  currentHighestBet: number;
  minRaise: number;
  mainPot: number;
  markets: Market[];
  positions: Position[];
  activePlayerCount: number;
  raiseCount?: number;
}

export interface BotPokerDecision {
  action: 'CHECK' | 'CALL' | 'BET' | 'RAISE' | 'FOLD';
  amount?: number;
  reasoning: string;
}

export interface BotTradingDecision {
  orders: {
    marketId: string;
    side: OrderSide;
    type: 'LIMIT' | 'MARKET';
    price: number;
    quantity: number;
  }[];
  reasoning: string;
}

export class BotEngine {
  // Decide Poker Action based on Bot Personality and Private Hole Cards only
  public static decidePokerAction(obs: BotObservation): BotPokerDecision {
    const { botPlayer, communityCards, currentHighestBet, minRaise, mainPot, activePlayerCount, raiseCount = 0 } = obs;
    const callAmount = Math.max(0, currentHighestBet - botPlayer.currentBet);
    const availableCash = botPlayer.cashBalance;
    const personality = botPlayer.botPersonality || 'quant';

    // Calculate private equity
    const equity = estimatePokerEquity(botPlayer.cards, communityCards, activePlayerCount - 1, 150);

    // Pot odds = callAmount / (mainPot + callAmount)
    const potOdds = callAmount > 0 ? callAmount / (mainPot + callAmount) : 0;
    const canRaise = raiseCount < 2 && availableCash > callAmount + minRaise;

    switch (personality) {
      case 'quant': {
        // Strict EV calculation
        if (callAmount === 0) {
          if (equity > 0.40 && availableCash > 100) {
            const betAmount = Math.min(availableCash, Math.max(minRaise, Math.round(mainPot * 0.5)));
            return {
              action: 'BET',
              amount: betAmount,
              reasoning: `Equity ${(equity * 100).toFixed(0)}% > 40% threshold. Value betting half pot.`,
            };
          }
          return { action: 'CHECK', reasoning: `Checking with ${(equity * 100).toFixed(0)}% equity.` };
        }

        if (canRaise && equity > potOdds + 0.10 && availableCash >= callAmount * 2 && equity > 0.58) {
          const raiseAmount = Math.min(availableCash, currentHighestBet + Math.max(minRaise, callAmount + 100));
          return {
            action: 'RAISE',
            amount: raiseAmount,
            reasoning: `Strong equity (${(equity * 100).toFixed(0)}%) justifies value raise over pot odds (${(potOdds * 100).toFixed(0)}%).`,
          };
        } else if (equity >= potOdds) {
          return {
            action: 'CALL',
            amount: Math.min(availableCash, callAmount),
            reasoning: `Equity (${(equity * 100).toFixed(0)}%) meets required pot odds (${(potOdds * 100).toFixed(0)}%). Calling ${callAmount}.`,
          };
        } else {
          return {
            action: 'FOLD',
            reasoning: `Negative EV: equity (${(equity * 100).toFixed(0)}%) < pot odds (${(potOdds * 100).toFixed(0)}%). Folding.`,
          };
        }
      }

      case 'poker_shark': {
        // Position-aware with strategic bluff frequency
        const shouldBluff = Math.random() < 0.20;
        if (callAmount === 0) {
          if (equity > 0.35 || shouldBluff) {
            const bet = Math.min(availableCash, Math.max(minRaise, Math.round(mainPot * 0.65)));
            return {
              action: 'BET',
              amount: Math.max(50, bet),
              reasoning: shouldBluff ? 'Executing positional probe bluff.' : 'Betting strong range.',
            };
          }
          return { action: 'CHECK', reasoning: 'Pot control check.' };
        }

        if (canRaise && shouldBluff && availableCash > callAmount * 2.5) {
          const raiseAmount = Math.min(availableCash, currentHighestBet + Math.max(minRaise, callAmount * 2));
          return {
            action: 'RAISE',
            amount: raiseAmount,
            reasoning: 'Aggressive semi-bluff raise to apply maximum pressure.',
          };
        }

        if (equity > potOdds - 0.05 && availableCash >= callAmount) {
          return { action: 'CALL', amount: callAmount, reasoning: 'Defending range with playable equity.' };
        }
        return { action: 'FOLD', reasoning: 'Disciplined fold against opponent aggression.' };
      }

      case 'trader': {
        // Active, loves action and raising on flop
        if (callAmount === 0) {
          if (equity > 0.30 && availableCash > 100) {
            return {
              action: 'BET',
              amount: Math.min(availableCash, Math.max(minRaise, Math.round(mainPot * 0.4))),
              reasoning: 'Probing bet for price discovery.',
            };
          }
          return { action: 'CHECK', reasoning: 'Checking range.' };
        }
        if (equity > 0.25 || Math.random() < 0.4) {
          return { action: 'CALL', amount: Math.min(availableCash, callAmount), reasoning: 'Calling to see next card.' };
        }
        return { action: 'FOLD', reasoning: 'Folding weak holding.' };
      }

      case 'degen': {
        // High aggression, loves huge bets & all-ins
        if (callAmount === 0) {
          if (Math.random() < 0.65 && availableCash > 150) {
            return {
              action: 'BET',
              amount: Math.min(availableCash, Math.max(minRaise, Math.round(mainPot * 0.85))),
              reasoning: 'Heavy pot-sized blast to drive action.',
            };
          }
          return { action: 'CHECK', reasoning: 'Waiting for big pot.' };
        }
        if (canRaise && Math.random() < 0.35 && availableCash >= callAmount * 2) {
          const raiseAmount = Math.min(availableCash, currentHighestBet + Math.max(minRaise, callAmount * 2));
          return {
            action: 'RAISE',
            amount: raiseAmount,
            reasoning: 'Re-raising with fearless aggression!',
          };
        }
        if (availableCash >= callAmount) {
          return { action: 'CALL', amount: callAmount, reasoning: 'Gambling on the draw. Calling.' };
        }
        return { action: 'FOLD', reasoning: 'Folding.' };
      }

      case 'market_maker':
      default: {
        // Balanced, low variance
        if (callAmount === 0) {
          return { action: 'CHECK', reasoning: 'Checking to manage risk.' };
        }
        if (equity >= potOdds && availableCash >= callAmount) {
          return { action: 'CALL', amount: callAmount, reasoning: 'Calling based on balanced pot odds.' };
        }
        return { action: 'FOLD', reasoning: 'Folding low-EV holding.' };
      }
    }
  }

  // Decide Market Orders based on Bot Strategy & Discrepancy between Market Price and Private Fair Value
  public static decideMarketOrders(obs: BotObservation): BotTradingDecision {
    const { botPlayer, communityCards, markets, activePlayerCount } = obs;
    const availableCash = botPlayer.cashBalance;
    const personality = botPlayer.botPersonality || 'quant';

    if (availableCash < 50) {
      return { orders: [], reasoning: 'Insufficient cash balance for trading.' };
    }

    const equity = estimatePokerEquity(botPlayer.cards, communityCards, activePlayerCount - 1, 100);
    const orders: BotTradingDecision['orders'] = [];
    let reasoning = '';

    // Find the market for this bot winning
    const ownWinMarket = markets.find((m) => m.targetPlayerId === botPlayer.id);

    switch (personality) {
      case 'quant': {
        // Exploit mispricings where Market Price differs significantly from true equity
        if (ownWinMarket && !ownWinMarket.resolved) {
          const edge = equity - ownWinMarket.currentYesPrice;
          if (edge > 0.08) {
            // Market is underpricing Quant's win probability -> BUY YES
            const size = Math.min(availableCash * 0.15, 300);
            const qty = Math.max(10, Math.floor(size / ownWinMarket.currentYesPrice));
            orders.push({
              marketId: ownWinMarket.id,
              side: 'BUY_YES',
              type: 'MARKET',
              price: ownWinMarket.currentYesPrice,
              quantity: qty,
            });
            reasoning = `Found +${(edge * 100).toFixed(1)}% edge on own win contract (Fair ${(equity * 100).toFixed(0)}% vs Mkt ${(ownWinMarket.currentYesPrice * 100).toFixed(0)}%). Buying ${qty} YES.`;
          } else if (edge < -0.10) {
            // Market is overpricing Quant -> BUY NO
            const size = Math.min(availableCash * 0.12, 250);
            const qty = Math.max(10, Math.floor(size / ownWinMarket.currentNoPrice));
            orders.push({
              marketId: ownWinMarket.id,
              side: 'BUY_NO',
              type: 'MARKET',
              price: ownWinMarket.currentYesPrice,
              quantity: qty,
            });
            reasoning = `Market overestimating win rate by ${Math.abs(edge * 100).toFixed(1)}%. Buying ${qty} NO.`;
          }
        }
        break;
      }

      case 'trader': {
        // Fast trend follower
        for (const m of markets) {
          if (m.volume > 200 && Math.random() < 0.35) {
            const side: OrderSide = Math.random() > 0.5 ? 'BUY_YES' : 'BUY_NO';
            const price = side === 'BUY_YES' ? m.currentYesPrice : m.currentNoPrice;
            const qty = Math.floor(Math.min(availableCash * 0.1, 150) / price);
            if (qty > 5) {
              orders.push({
                marketId: m.id,
                side,
                type: 'MARKET',
                price: m.currentYesPrice,
                quantity: qty,
              });
              reasoning = `Following momentum breakout on ${m.ticker}. Placing ${qty} ${side}.`;
              break;
            }
          }
        }
        break;
      }

      case 'poker_shark': {
        // Strategic bluffing / market manipulation
        if (ownWinMarket && !ownWinMarket.resolved) {
          // If holding weak cards (equity < 0.20), Shark bluffs by aggressively buying YES to spoof confidence!
          if (equity < 0.20 && Math.random() < 0.35) {
            const qty = 80;
            orders.push({
              marketId: ownWinMarket.id,
              side: 'BUY_YES',
              type: 'MARKET',
              price: ownWinMarket.currentYesPrice,
              quantity: qty,
            });
            reasoning = `Spoofing hand strength on prediction market to induce opponent folds!`;
          } else if (equity > 0.60) {
            // Trapping: buy quietly
            const qty = 120;
            orders.push({
              marketId: ownWinMarket.id,
              side: 'BUY_YES',
              type: 'LIMIT',
              price: Math.max(0.01, ownWinMarket.currentYesPrice - 0.02),
              quantity: qty,
            });
            reasoning = `Building quiet long position at discount before value betting.`;
          }
        }
        break;
      }

      case 'degen': {
        // Chases board longshot events (Straight+ or Board pairs)
        const boardMarket = markets.find((m) => m.category === 'BOARD_EVENT' || m.category === 'HAND_TYPE');
        if (boardMarket && !boardMarket.resolved && Math.random() < 0.5) {
          const qty = 150;
          orders.push({
            marketId: boardMarket.id,
            side: 'BUY_YES',
            type: 'MARKET',
            price: boardMarket.currentYesPrice,
            quantity: qty,
          });
          reasoning = `Aping into ${boardMarket.name} for high payout!`;
        }
        break;
      }

      case 'market_maker': {
        // Post bid and ask orders across markets to provide liquidity
        for (const m of markets.slice(0, 3)) {
          if (!m.resolved) {
            const bidPrice = Math.max(0.01, Number((m.currentYesPrice - 0.02).toFixed(2)));
            const askPrice = Math.min(0.99, Number((m.currentYesPrice + 0.02).toFixed(2)));
            orders.push(
              {
                marketId: m.id,
                side: 'BUY_YES',
                type: 'LIMIT',
                price: bidPrice,
                quantity: 100,
              },
              {
                marketId: m.id,
                side: 'SELL_YES',
                type: 'LIMIT',
                price: askPrice,
                quantity: 100,
              }
            );
          }
        }
        reasoning = `Providing continuous two-sided liquidity and capturing bid-ask spread.`;
        break;
      }
    }

    return { orders, reasoning: reasoning || 'Observing market flow.' };
  }
}
