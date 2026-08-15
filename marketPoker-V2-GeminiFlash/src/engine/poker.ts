import { Card, HandCategory, HandEvaluation, Rank, Suit } from '../types/game';

export const SUITS: Suit[] = ['spades', 'hearts', 'diamonds', 'clubs'];
export const RANKS: Rank[] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

export const RANK_VALUES: Record<Rank, number> = {
  '2': 2,
  '3': 3,
  '4': 4,
  '5': 5,
  '6': 6,
  '7': 7,
  '8': 8,
  '9': 9,
  'T': 10,
  'J': 11,
  'Q': 12,
  'K': 13,
  'A': 14,
};

export const RANK_NAMES: Record<Rank, string> = {
  '2': 'Deuce',
  '3': 'Three',
  '4': 'Four',
  '5': 'Five',
  '6': 'Six',
  '7': 'Seven',
  '8': 'Eight',
  '9': 'Nine',
  'T': 'Ten',
  'J': 'Jack',
  'Q': 'Queen',
  'K': 'King',
  'A': 'Ace',
};

export function createDeck(): Card[] {
  const deck: Card[] = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({
        suit,
        rank,
        id: `${rank}-${suit}`,
      });
    }
  }
  return deck;
}

export function shuffleDeck(deck: Card[]): Card[] {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }
  return shuffled;
}

// 5-Card hand evaluator helper
function evaluate5CardHand(cards: Card[]): HandEvaluation {
  if (cards.length !== 5) {
    throw new Error('evaluate5CardHand requires exactly 5 cards');
  }

  // Sort descending by rank value
  const sorted = [...cards].sort((a, b) => RANK_VALUES[b.rank] - RANK_VALUES[a.rank]);
  const values = sorted.map((c) => RANK_VALUES[c.rank]);
  const suits = sorted.map((c) => c.suit);

  const isFlush = suits.every((s) => s === suits[0]);

  // Check straight
  let isStraight = false;
  let straightHigh = 0;

  // Normal straight check (5 consecutive values)
  if (
    values[0] - values[1] === 1 &&
    values[1] - values[2] === 1 &&
    values[2] - values[3] === 1 &&
    values[3] - values[4] === 1
  ) {
    isStraight = true;
    straightHigh = values[0];
  } else if (
    values[0] === 14 &&
    values[1] === 5 &&
    values[2] === 4 &&
    values[3] === 3 &&
    values[4] === 2
  ) {
    // Ace-low straight (A-2-3-4-5)
    isStraight = true;
    straightHigh = 5;
  }

  // Count frequencies
  const counts: Record<number, number> = {};
  for (const val of values) {
    counts[val] = (counts[val] || 0) + 1;
  }

  const grouped = Object.entries(counts)
    .map(([valStr, count]) => ({ val: Number(valStr), count }))
    .sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return b.val - a.val;
    });

  // Base multiplier per tier: 10^10 to ensure category superiority
  const TIER = 100000000;

  // 1. Royal Flush / Straight Flush
  if (isFlush && isStraight) {
    if (straightHigh === 14) {
      return {
        category: 'Royal Flush',
        rankScore: 9 * TIER + straightHigh,
        description: 'Royal Flush',
        bestFiveCards: sorted,
      };
    }
    return {
      category: 'Straight Flush',
      rankScore: 8 * TIER + straightHigh,
      description: `Straight Flush, ${RANK_NAMES[RANKS[straightHigh - 2]] || straightHigh} High`,
      bestFiveCards: sorted,
    };
  }

  // 2. Four of a Kind
  if (grouped[0].count === 4) {
    const quadVal = grouped[0].val;
    const kickerVal = grouped[1].val;
    return {
      category: 'Four of a Kind',
      rankScore: 7 * TIER + quadVal * 100 + kickerVal,
      description: `Four of a Kind, ${quadVal}s`,
      bestFiveCards: sorted,
    };
  }

  // 3. Full House
  if (grouped[0].count === 3 && grouped[1].count === 2) {
    const tripVal = grouped[0].val;
    const pairVal = grouped[1].val;
    return {
      category: 'Full House',
      rankScore: 6 * TIER + tripVal * 100 + pairVal,
      description: `Full House, ${tripVal}s full of ${pairVal}s`,
      bestFiveCards: sorted,
    };
  }

  // 4. Flush
  if (isFlush) {
    const kickerScore = values[0] * 1000000 + values[1] * 10000 + values[2] * 100 + values[3] * 1 + values[4] * 0.01;
    return {
      category: 'Flush',
      rankScore: 5 * TIER + kickerScore,
      description: `Flush, ${values[0]} High`,
      bestFiveCards: sorted,
    };
  }

  // 5. Straight
  if (isStraight) {
    return {
      category: 'Straight',
      rankScore: 4 * TIER + straightHigh,
      description: `Straight, ${straightHigh} High`,
      bestFiveCards: sorted,
    };
  }

  // 6. Three of a Kind
  if (grouped[0].count === 3) {
    const tripVal = grouped[0].val;
    const kickers = [grouped[1].val, grouped[2].val].sort((a, b) => b - a);
    return {
      category: 'Three of a Kind',
      rankScore: 3 * TIER + tripVal * 10000 + kickers[0] * 100 + kickers[1],
      description: `Three of a Kind, ${tripVal}s`,
      bestFiveCards: sorted,
    };
  }

  // 7. Two Pair
  if (grouped[0].count === 2 && grouped[1].count === 2) {
    const highPair = Math.max(grouped[0].val, grouped[1].val);
    const lowPair = Math.min(grouped[0].val, grouped[1].val);
    const kicker = grouped[2].val;
    return {
      category: 'Two Pair',
      rankScore: 2 * TIER + highPair * 10000 + lowPair * 100 + kicker,
      description: `Two Pair, ${highPair}s and ${lowPair}s`,
      bestFiveCards: sorted,
    };
  }

  // 8. Pair
  if (grouped[0].count === 2) {
    const pairVal = grouped[0].val;
    const kickers = [grouped[1].val, grouped[2].val, grouped[3].val].sort((a, b) => b - a);
    return {
      category: 'Pair',
      rankScore: 1 * TIER + pairVal * 100000 + kickers[0] * 1000 + kickers[1] * 10 + kickers[2],
      description: `Pair of ${pairVal}s`,
      bestFiveCards: sorted,
    };
  }

  // 9. High Card
  const kickerScore = values[0] * 1000000 + values[1] * 10000 + values[2] * 100 + values[3] * 1 + values[4] * 0.01;
  return {
    category: 'High Card',
    rankScore: kickerScore,
    description: `High Card, ${values[0]}`,
    bestFiveCards: sorted,
  };
}

// Generate all 5-card combinations from a set of cards (e.g. 7 choose 5 = 21 combinations)
export function getCombinations(cards: Card[], k: number = 5): Card[][] {
  const result: Card[][] = [];
  function backtrack(start: number, current: Card[]) {
    if (current.length === k) {
      result.push([...current]);
      return;
    }
    for (let i = start; i < cards.length; i++) {
      current.push(cards[i]);
      backtrack(i + 1, current);
      current.pop();
    }
  }
  backtrack(0, []);
  return result;
}

// Main 7-card / 5-to-7 card evaluator
export function evaluateHand(holeCards: Card[], communityCards: Card[]): HandEvaluation {
  const allCards = [...holeCards, ...communityCards];
  if (allCards.length < 5) {
    // If fewer than 5 cards available, return temporary high card or pair representation
    if (holeCards.length === 2) {
      if (holeCards[0].rank === holeCards[1].rank) {
        return {
          category: 'Pair',
          rankScore: 100000000 + RANK_VALUES[holeCards[0].rank] * 100,
          description: `Pocket Pair of ${holeCards[0].rank}s`,
          bestFiveCards: holeCards,
        };
      }
      return {
        category: 'High Card',
        rankScore: Math.max(RANK_VALUES[holeCards[0].rank], RANK_VALUES[holeCards[1].rank]),
        description: `High Card ${holeCards[0].rank}/${holeCards[1].rank}`,
        bestFiveCards: holeCards,
      };
    }
    return {
      category: 'High Card',
      rankScore: 0,
      description: 'Incomplete Hand',
      bestFiveCards: allCards,
    };
  }

  const combos = getCombinations(allCards, 5);
  let best: HandEvaluation | null = null;

  for (const combo of combos) {
    const evalResult = evaluate5CardHand(combo);
    if (!best || evalResult.rankScore > best.rankScore) {
      best = evalResult;
    }
  }

  return best!;
}

// Fast Monte Carlo Equity Estimator (computes true mathematical probability for Quant/bots/inspector)
export function estimatePokerEquity(
  holeCards: Card[],
  communityCards: Card[],
  numOpponents: number = 4,
  simulations: number = 300
): number {
  if (holeCards.length < 2) return 1 / (numOpponents + 1);

  // Deck without known cards
  const knownCardIds = new Set([...holeCards, ...communityCards].map((c) => c.id));
  const fullDeck = createDeck().filter((c) => !knownCardIds.has(c.id));

  let wins = 0;
  let ties = 0;

  for (let sim = 0; sim < simulations; sim++) {
    // Shuffle remaining deck
    const deck = shuffleDeck(fullDeck);
    let cardPtr = 0;

    // Complete community cards up to 5
    const simBoard = [...communityCards];
    while (simBoard.length < 5) {
      simBoard.push(deck[cardPtr++]);
    }

    // Deal opponents
    const myEval = evaluateHand(holeCards, simBoard);
    let isBest = true;
    let isTie = false;

    for (let opp = 0; opp < numOpponents; opp++) {
      const oppCards = [deck[cardPtr++], deck[cardPtr++]];
      const oppEval = evaluateHand(oppCards, simBoard);

      if (oppEval.rankScore > myEval.rankScore) {
        isBest = false;
        break;
      } else if (oppEval.rankScore === myEval.rankScore) {
        isTie = true;
      }
    }

    if (isBest) {
      if (isTie) ties += 1;
      else wins += 1;
    }
  }

  const equity = (wins + ties * 0.5) / simulations;
  return Math.min(0.99, Math.max(0.01, Number(equity.toFixed(3))));
}

// True probability for Board / Hand Type contracts
export function estimateContractFairValue(
  contractType: 'STRAIGHT_OR_BETTER' | 'BOARD_PAIRS',
  communityCards: Card[],
  allActiveHoleCards: Card[][] = [],
  simulations: number = 250
): number {
  const knownCards = [...communityCards, ...allActiveHoleCards.flat()];
  const knownSet = new Set(knownCards.map((c) => c.id));
  const fullDeck = createDeck().filter((c) => !knownSet.has(c.id));

  let positiveCount = 0;

  for (let s = 0; s < simulations; s++) {
    const deck = shuffleDeck(fullDeck);
    let ptr = 0;
    const simBoard = [...communityCards];
    while (simBoard.length < 5) {
      simBoard.push(deck[ptr++]);
    }

    if (contractType === 'BOARD_PAIRS') {
      const ranks = simBoard.map((c) => c.rank);
      const uniqueRanks = new Set(ranks);
      if (uniqueRanks.size < 5) {
        positiveCount++;
      }
    } else if (contractType === 'STRAIGHT_OR_BETTER') {
      // Check if winning hand among simulated players is straight or better
      let bestRankScore = 0;
      let winningCategory: HandCategory = 'High Card';

      // Use active hole cards if available, else simulate 5 random players
      const testHands =
        allActiveHoleCards.length > 0
          ? allActiveHoleCards
          : [
              [deck[ptr++], deck[ptr++]],
              [deck[ptr++], deck[ptr++]],
              [deck[ptr++], deck[ptr++]],
              [deck[ptr++], deck[ptr++]],
              [deck[ptr++], deck[ptr++]],
            ];

      for (const h of testHands) {
        const ev = evaluateHand(h, simBoard);
        if (ev.rankScore > bestRankScore) {
          bestRankScore = ev.rankScore;
          winningCategory = ev.category;
        }
      }

      const straightOrBetter: HandCategory[] = [
        'Straight',
        'Flush',
        'Full House',
        'Four of a Kind',
        'Straight Flush',
        'Royal Flush',
      ];
      if (straightOrBetter.includes(winningCategory)) {
        positiveCount++;
      }
    }
  }

  return Math.min(0.99, Math.max(0.01, Number((positiveCount / simulations).toFixed(3))));
}
