import {
  Card,
  GamePhase,
  Market,
  MarketCategory,
  MarketExpiryStreet,
  MarketLifespan,
  Order,
  OrderBookLevel,
  OrderSide,
  OrderType,
  Position,
  Trade,
} from '../types/game';

// Helper to create seeded bids & asks for a market
function createSeededBook(
  marketId: string,
  yesPrice: number,
  spread: number = 0.02,
  depth: number = 300
): { bids: Order[]; asks: Order[] } {
  const bidPrice = Math.max(0.01, Number((yesPrice - spread / 2).toFixed(2)));
  const askPrice = Math.min(0.99, Number((yesPrice + spread / 2).toFixed(2)));

  const bids: Order[] = [
    {
      id: `seed-bid-${marketId}-1`,
      marketId,
      playerId: 'system-mm',
      playerName: 'Market Maker',
      side: 'BUY_YES',
      type: 'LIMIT',
      price: bidPrice,
      quantity: depth,
      filledQuantity: 0,
      status: 'OPEN',
      createdAt: Date.now(),
    },
    {
      id: `seed-bid-${marketId}-2`,
      marketId,
      playerId: 'system-mm',
      playerName: 'Market Maker',
      side: 'BUY_YES',
      type: 'LIMIT',
      price: Math.max(0.01, Number((bidPrice - 0.03).toFixed(2))),
      quantity: depth * 2,
      filledQuantity: 0,
      status: 'OPEN',
      createdAt: Date.now(),
    },
  ];

  const asks: Order[] = [
    {
      id: `seed-ask-${marketId}-1`,
      marketId,
      playerId: 'system-mm',
      playerName: 'Market Maker',
      side: 'SELL_YES',
      type: 'LIMIT',
      price: askPrice,
      quantity: depth,
      filledQuantity: 0,
      status: 'OPEN',
      createdAt: Date.now(),
    },
    {
      id: `seed-ask-${marketId}-2`,
      marketId,
      playerId: 'system-mm',
      playerName: 'Market Maker',
      side: 'SELL_YES',
      type: 'LIMIT',
      price: Math.min(0.99, Number((askPrice + 0.03).toFixed(2))),
      quantity: depth * 2,
      filledQuantity: 0,
      status: 'OPEN',
      createdAt: Date.now(),
    },
  ];

  return { bids, asks };
}

// Initial Polymarket & Kalshi style event markets (Core, Short-Lived, Long-Lived)
export function createInitialMarkets(
  playerNames: string[] = ['You', 'The Quant', 'The Trader', 'The Poker Shark', 'The Degen']
): Market[] {
  const markets: Market[] = [];

  // 1. Core Player Win Markets (5 markets)
  playerNames.forEach((name, idx) => {
    const isUser = idx === 0;
    const ticker = isUser ? 'YOU_WIN' : `P${idx + 1}_WIN`;
    const initialPrice = 0.20; // 5 players = 20% prior
    const marketId = `market-player-${idx + 1}`;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 350);

    markets.push({
      id: marketId,
      name: `${name} Wins Hand`,
      ticker,
      category: 'PLAYER_WIN',
      lifespan: 'CORE_HAND',
      expiryStreet: 'SHOWDOWN',
      iconTag: '♠',
      targetPlayerId: `player-${idx + 1}`,
      description: `Resolves to YES ($1.00) if ${name} wins the poker pot at showdown or by forcing all folds.`,
      currentYesPrice: initialPrice,
      currentNoPrice: 0.80,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: {
        yesShares: 1000,
        noShares: 4000,
        kConstant: 4000000,
      },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [
        {
          timestamp: Date.now(),
          phase: 'PREFLOP_TRADING',
          price: initialPrice,
          volume: 0,
        },
      ],
      resolved: false,
    });
  });

  // 2. Core Hand: Straight or Better Wins
  {
    const marketId = 'market-hand-straight-plus';
    const initialPrice = 0.25;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 300);
    markets.push({
      id: marketId,
      name: 'Straight or Better Wins',
      ticker: 'STRAIGHT_PLUS',
      category: 'HAND_TYPE',
      lifespan: 'CORE_HAND',
      expiryStreet: 'SHOWDOWN',
      iconTag: '🏆',
      description: 'Resolves YES if winning hand is Straight, Flush, Full House, Quads, or Straight Flush.',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.75,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 1500, noShares: 4500, kConstant: 6750000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  // 3. Core Board: Board Pairs Contract
  {
    const marketId = 'market-board-pairs';
    const initialPrice = 0.42;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 300);
    markets.push({
      id: marketId,
      name: 'Board Pairs (Community Cards)',
      ticker: 'BOARD_PAIRS',
      category: 'BOARD_EVENT',
      lifespan: 'CORE_HAND',
      expiryStreet: 'SHOWDOWN',
      iconTag: '🃏',
      description: 'Resolves YES if the final 5 community cards contain at least one paired rank.',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.58,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 2000, noShares: 2750, kConstant: 5500000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  // 4. ⚡ Short-Lived Live Question: "Flop Has Ace or King"
  {
    const marketId = 'market-short-flop-ak';
    const initialPrice = 0.44;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 400);
    markets.push({
      id: marketId,
      name: 'Flop Contains Ace or King (A / K)',
      ticker: 'FLOP_ACE_KING',
      category: 'STREET_PROP',
      lifespan: 'SHORT_LIVED',
      expiryStreet: 'FLOP',
      iconTag: '⚡',
      description: 'Resolves YES when Flop is revealed if any of the 3 flop cards is an Ace or King. Instant payout on flop!',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.56,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 2200, noShares: 2800, kConstant: 6160000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  // 5. ⚡ Short-Lived Live Question: "Next Board Card is Red (♥ / ♦)"
  {
    const marketId = 'market-short-next-red';
    const initialPrice = 0.50;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 500);
    markets.push({
      id: marketId,
      name: 'Next Board Card is Red (♥ / ♦)',
      ticker: 'NEXT_CARD_RED',
      category: 'STREET_PROP',
      lifespan: 'SHORT_LIVED',
      expiryStreet: 'FLOP',
      iconTag: '🔴',
      description: 'Resolves YES if the first card of the next street is Hearts or Diamonds.',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.50,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 3000, noShares: 3000, kConstant: 9000000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  // 6. 🎯 Long-Lived Live Question: "Total Pot Crosses $2,000"
  {
    const marketId = 'market-long-pot-2k';
    const initialPrice = 0.38;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 400);
    markets.push({
      id: marketId,
      name: 'Total Poker Pot Crosses $2,000',
      ticker: 'POT_OVER_2K',
      category: 'TOURNAMENT_PROP',
      lifespan: 'LONG_LIVED',
      expiryStreet: 'SHOWDOWN',
      iconTag: '💰',
      description: 'Resolves YES if active betting pushes the final poker pot to $2,000 or more this hand.',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.62,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 1900, noShares: 3100, kConstant: 5890000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  // 7. 🎯 Long-Lived Live Question: "Rex (The Degen) Goes All-In"
  {
    const marketId = 'market-long-degen-allin';
    const initialPrice = 0.52;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 400);
    markets.push({
      id: marketId,
      name: 'Rex (The Degen) Goes All-In',
      ticker: 'REX_ALL_IN',
      category: 'TOURNAMENT_PROP',
      lifespan: 'LONG_LIVED',
      expiryStreet: 'SHOWDOWN',
      iconTag: '🔥',
      description: 'Resolves YES if Rex shoves all his chips into the pot at any point during this hand.',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.48,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 2600, noShares: 2400, kConstant: 6240000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  // 8. 🏆 Long-Lived Tournament Prop: "Player 1 (You) Finishes Top 2"
  {
    const marketId = 'market-long-you-top2';
    const initialPrice = 0.40;
    const { bids, asks } = createSeededBook(marketId, initialPrice, 0.02, 500);
    markets.push({
      id: marketId,
      name: 'You (P1) Finish Top 2 in Tournament',
      ticker: 'YOU_TOP_2',
      category: 'TOURNAMENT_PROP',
      lifespan: 'LONG_LIVED',
      expiryStreet: 'TOURNAMENT',
      iconTag: '🥇',
      description: 'Long-range tournament prediction: Resolves YES if You survive until heads-up (Top 2 seats).',
      currentYesPrice: initialPrice,
      currentNoPrice: 0.60,
      bestBid: bids[0].price,
      bestAsk: asks[0].price,
      volume: 0,
      liquidityPool: { yesShares: 2000, noShares: 3000, kConstant: 6000000 },
      bids,
      asks,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'PREFLOP_TRADING', price: initialPrice, volume: 0 }],
      resolved: false,
    });
  }

  return markets;
}

// Generate new live street-level questions dynamically as cards hit the felt
export function generateStreetMarkets(
  street: 'TURN' | 'RIVER',
  communityCards: Card[],
  handNumber: number
): Market[] {
  const newMarkets: Market[] = [];

  if (street === 'TURN') {
    // 1. Turn is a Spade ♠
    const marketId1 = `market-turn-spade-h${handNumber}`;
    const { bids: b1, asks: a1 } = createSeededBook(marketId1, 0.25, 0.02, 300);
    newMarkets.push({
      id: marketId1,
      name: 'Turn Card is a Spade (♠)',
      ticker: 'TURN_SPADE',
      category: 'STREET_PROP',
      lifespan: 'SHORT_LIVED',
      expiryStreet: 'TURN',
      iconTag: '♠',
      description: 'Resolves YES when the single Turn card is dealt if its suit is Spades.',
      currentYesPrice: 0.25,
      currentNoPrice: 0.75,
      bestBid: b1[0].price,
      bestAsk: a1[0].price,
      volume: 0,
      liquidityPool: { yesShares: 1500, noShares: 4500, kConstant: 6750000 },
      bids: b1,
      asks: a1,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'FLOP_TRADING', price: 0.25, volume: 0 }],
      resolved: false,
    });

    // 2. Turn Card is High Card (10, J, Q, K, A)
    const marketId2 = `market-turn-broadway-h${handNumber}`;
    const { bids: b2, asks: a2 } = createSeededBook(marketId2, 0.38, 0.02, 300);
    newMarkets.push({
      id: marketId2,
      name: 'Turn is High Card (T, J, Q, K, A)',
      ticker: 'TURN_HIGH_CARD',
      category: 'STREET_PROP',
      lifespan: 'SHORT_LIVED',
      expiryStreet: 'TURN',
      iconTag: '👑',
      description: 'Resolves YES if Turn card rank is 10 or higher.',
      currentYesPrice: 0.38,
      currentNoPrice: 0.62,
      bestBid: b2[0].price,
      bestAsk: a2[0].price,
      volume: 0,
      liquidityPool: { yesShares: 1900, noShares: 3100, kConstant: 5890000 },
      bids: b2,
      asks: a2,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'FLOP_TRADING', price: 0.38, volume: 0 }],
      resolved: false,
    });
  } else if (street === 'RIVER') {
    // 1. River Card is Black (♠ / ♣)
    const marketId1 = `market-river-black-h${handNumber}`;
    const { bids: b1, asks: a1 } = createSeededBook(marketId1, 0.50, 0.02, 350);
    newMarkets.push({
      id: marketId1,
      name: 'River Card is Black (♠ / ♣)',
      ticker: 'RIVER_BLACK',
      category: 'STREET_PROP',
      lifespan: 'SHORT_LIVED',
      expiryStreet: 'RIVER',
      iconTag: '♣',
      description: 'Resolves YES if the final 5th river card is Spades or Clubs.',
      currentYesPrice: 0.50,
      currentNoPrice: 0.50,
      bestBid: b1[0].price,
      bestAsk: a1[0].price,
      volume: 0,
      liquidityPool: { yesShares: 2500, noShares: 2500, kConstant: 6250000 },
      bids: b1,
      asks: a1,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'TURN_TRADING', price: 0.50, volume: 0 }],
      resolved: false,
    });

    // 2. River Pairs the Board
    const marketId2 = `market-river-pairs-h${handNumber}`;
    const { bids: b2, asks: a2 } = createSeededBook(marketId2, 0.28, 0.02, 300);
    newMarkets.push({
      id: marketId2,
      name: 'River Card Pairs Existing Board',
      ticker: 'RIVER_PAIRS_BOARD',
      category: 'STREET_PROP',
      lifespan: 'SHORT_LIVED',
      expiryStreet: 'RIVER',
      iconTag: '⚡',
      description: 'Resolves YES if the 5th card matches the rank of any of the first 4 community cards.',
      currentYesPrice: 0.28,
      currentNoPrice: 0.72,
      bestBid: b2[0].price,
      bestAsk: a2[0].price,
      volume: 0,
      liquidityPool: { yesShares: 1400, noShares: 3600, kConstant: 5040000 },
      bids: b2,
      asks: a2,
      noBids: [],
      noAsks: [],
      priceHistory: [{ timestamp: Date.now(), phase: 'TURN_TRADING', price: 0.28, volume: 0 }],
      resolved: false,
    });
  }

  return newMarkets;
}

// Order Matching Engine
export function executeOrder(
  market: Market,
  order: Order,
  phase: GamePhase
): {
  updatedMarket: Market;
  filledOrder: Order;
  trades: Trade[];
  costDebit: number;
} {
  const m = { ...market, bids: [...market.bids], asks: [...market.asks] };
  const trades: Trade[] = [];
  let remainingQty = order.quantity - order.filledQuantity;
  let totalCost = 0;

  if (order.side === 'BUY_YES') {
    // Match against asks (sell YES orders) sorted lowest price first
    m.asks.sort((a, b) => a.price - b.price || a.createdAt - b.createdAt);

    const activeAsks: Order[] = [];
    for (const ask of m.asks) {
      if (remainingQty <= 0) {
        activeAsks.push(ask);
        continue;
      }

      // Check if price matches (market order matches any price, limit order matches if ask.price <= order.price)
      if (order.type === 'MARKET' || ask.price <= order.price) {
        const askRemaining = ask.quantity - ask.filledQuantity;
        const matchQty = Math.min(remainingQty, askRemaining);
        const matchPrice = ask.price;

        ask.filledQuantity += matchQty;
        if (ask.filledQuantity >= ask.quantity) {
          ask.status = 'FILLED';
        } else {
          ask.status = 'PARTIALLY_FILLED';
          activeAsks.push(ask);
        }

        remainingQty -= matchQty;
        totalCost += matchQty * matchPrice;

        trades.push({
          id: `trade-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          marketId: m.id,
          buyerId: order.playerId,
          sellerId: ask.playerId,
          price: matchPrice,
          quantity: matchQty,
          timestamp: Date.now(),
          phase,
          side: 'BUY_YES',
        });
      } else {
        activeAsks.push(ask);
      }
    }
    m.asks = activeAsks;

    // If market order still has remaining qty, execute against AMM pool
    if (remainingQty > 0 && order.type === 'MARKET') {
      const ammPrice = Math.min(0.99, Number((m.currentYesPrice + 0.01).toFixed(2)));
      totalCost += remainingQty * ammPrice;
      trades.push({
        id: `trade-amm-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        marketId: m.id,
        buyerId: order.playerId,
        sellerId: 'system-amm',
        price: ammPrice,
        quantity: remainingQty,
        timestamp: Date.now(),
        phase,
        side: 'BUY_YES',
      });
      remainingQty = 0;
    } else if (remainingQty > 0 && order.type === 'LIMIT') {
      // Add remainder to bids book
      const openOrder: Order = {
        ...order,
        filledQuantity: order.quantity - remainingQty,
        status: remainingQty === order.quantity ? 'OPEN' : 'PARTIALLY_FILLED',
      };
      m.bids.push(openOrder);
      m.bids.sort((a, b) => b.price - a.price || a.createdAt - b.createdAt);
    }
  } else if (order.side === 'SELL_YES' || order.side === 'BUY_NO') {
    // Match against bids (buy YES orders) sorted highest price first
    m.bids.sort((a, b) => b.price - a.price || a.createdAt - b.createdAt);

    const activeBids: Order[] = [];
    for (const bid of m.bids) {
      if (remainingQty <= 0) {
        activeBids.push(bid);
        continue;
      }

      if (order.type === 'MARKET' || bid.price >= order.price) {
        const bidRemaining = bid.quantity - bid.filledQuantity;
        const matchQty = Math.min(remainingQty, bidRemaining);
        const matchPrice = bid.price;

        bid.filledQuantity += matchQty;
        if (bid.filledQuantity >= bid.quantity) {
          bid.status = 'FILLED';
        } else {
          bid.status = 'PARTIALLY_FILLED';
          activeBids.push(bid);
        }

        remainingQty -= matchQty;
        // Selling YES gives revenue (matchQty * matchPrice), Buying NO costs matchQty * (1 - matchPrice)
        totalCost += order.side === 'BUY_NO' ? matchQty * (1 - matchPrice) : -matchQty * matchPrice;

        trades.push({
          id: `trade-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          marketId: m.id,
          buyerId: bid.playerId,
          sellerId: order.playerId,
          price: matchPrice,
          quantity: matchQty,
          timestamp: Date.now(),
          phase,
          side: order.side,
        });
      } else {
        activeBids.push(bid);
      }
    }
    m.bids = activeBids;

    if (remainingQty > 0 && order.type === 'MARKET') {
      const ammPrice = Math.max(0.01, Number((m.currentYesPrice - 0.01).toFixed(2)));
      totalCost += order.side === 'BUY_NO' ? remainingQty * (1 - ammPrice) : -remainingQty * ammPrice;
      trades.push({
        id: `trade-amm-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        marketId: m.id,
        buyerId: 'system-amm',
        sellerId: order.playerId,
        price: ammPrice,
        quantity: remainingQty,
        timestamp: Date.now(),
        phase,
        side: order.side,
      });
      remainingQty = 0;
    } else if (remainingQty > 0 && order.type === 'LIMIT') {
      const openOrder: Order = {
        ...order,
        filledQuantity: order.quantity - remainingQty,
        status: remainingQty === order.quantity ? 'OPEN' : 'PARTIALLY_FILLED',
      };
      m.asks.push(openOrder);
      m.asks.sort((a, b) => a.price - b.price || a.createdAt - b.createdAt);
    }
  }

  // Update market stats & price
  const lastTrade = trades[trades.length - 1];
  if (lastTrade) {
    m.currentYesPrice = Math.min(0.99, Math.max(0.01, Number(lastTrade.price.toFixed(2))));
    m.currentNoPrice = Number((1 - m.currentYesPrice).toFixed(2));
    m.volume += trades.reduce((acc, t) => acc + t.quantity, 0);

    m.priceHistory.push({
      timestamp: Date.now(),
      phase,
      price: m.currentYesPrice,
      volume: m.volume,
    });
  }

  m.bestBid = m.bids.length > 0 ? m.bids[0].price : Math.max(0.01, Number((m.currentYesPrice - 0.02).toFixed(2)));
  m.bestAsk = m.asks.length > 0 ? m.asks[0].price : Math.min(0.99, Number((m.currentYesPrice + 0.02).toFixed(2)));

  const updatedOrder: Order = {
    ...order,
    filledQuantity: order.quantity - remainingQty,
    status: remainingQty === 0 ? 'FILLED' : remainingQty === order.quantity ? 'OPEN' : 'PARTIALLY_FILLED',
  };

  return {
    updatedMarket: m,
    filledOrder: updatedOrder,
    trades,
    costDebit: totalCost,
  };
}

// Compute market depth ladder (order book visual representation)
export function getOrderBookDepth(market: Market): { bids: OrderBookLevel[]; asks: OrderBookLevel[] } {
  const bidMap = new Map<number, { quantity: number; orderCount: number }>();
  const askMap = new Map<number, { quantity: number; orderCount: number }>();

  // If no explicit orders, synthesize realistic Market Maker depth levels around spot
  if (market.bids.length === 0) {
    const p1 = Math.max(0.01, Number((market.currentYesPrice - 0.02).toFixed(2)));
    const p2 = Math.max(0.01, Number((market.currentYesPrice - 0.04).toFixed(2)));
    bidMap.set(p1, { quantity: 180, orderCount: 2 });
    bidMap.set(p2, { quantity: 450, orderCount: 3 });
  } else {
    market.bids.forEach((b) => {
      const remaining = b.quantity - b.filledQuantity;
      if (remaining > 0) {
        const existing = bidMap.get(b.price) || { quantity: 0, orderCount: 0 };
        bidMap.set(b.price, {
          quantity: existing.quantity + remaining,
          orderCount: existing.orderCount + 1,
        });
      }
    });
  }

  if (market.asks.length === 0) {
    const p1 = Math.min(0.99, Number((market.currentYesPrice + 0.02).toFixed(2)));
    const p2 = Math.min(0.99, Number((market.currentYesPrice + 0.04).toFixed(2)));
    askMap.set(p1, { quantity: 180, orderCount: 2 });
    askMap.set(p2, { quantity: 450, orderCount: 3 });
  } else {
    market.asks.forEach((a) => {
      const remaining = a.quantity - a.filledQuantity;
      if (remaining > 0) {
        const existing = askMap.get(a.price) || { quantity: 0, orderCount: 0 };
        askMap.set(a.price, {
          quantity: existing.quantity + remaining,
          orderCount: existing.orderCount + 1,
        });
      }
    });
  }

  const bids: OrderBookLevel[] = Array.from(bidMap.entries())
    .map(([price, d]) => ({ price, quantity: d.quantity, orderCount: d.orderCount }))
    .sort((a, b) => b.price - a.price);

  const asks: OrderBookLevel[] = Array.from(askMap.entries())
    .map(([price, d]) => ({ price, quantity: d.quantity, orderCount: d.orderCount }))
    .sort((a, b) => a.price - b.price);

  return { bids, asks };
}

// Update Position with new trade
export function updatePosition(
  currentPosition: Position | undefined,
  marketId: string,
  playerId: string,
  side: OrderSide,
  price: number,
  quantity: number
): Position {
  const pos: Position = currentPosition
    ? { ...currentPosition }
    : {
        marketId,
        playerId,
        yesContracts: 0,
        avgYesPrice: 0,
        noContracts: 0,
        avgNoPrice: 0,
        realizedPnl: 0,
      };

  if (side === 'BUY_YES') {
    const totalExistingCost = pos.yesContracts * pos.avgYesPrice;
    const newCost = quantity * price;
    pos.yesContracts += quantity;
    pos.avgYesPrice = pos.yesContracts > 0 ? Number(((totalExistingCost + newCost) / pos.yesContracts).toFixed(3)) : 0;
  } else if (side === 'SELL_YES') {
    if (pos.yesContracts > 0) {
      const closeQty = Math.min(pos.yesContracts, quantity);
      const pnl = closeQty * (price - pos.avgYesPrice);
      pos.realizedPnl += Number(pnl.toFixed(2));
      pos.yesContracts -= closeQty;
      if (pos.yesContracts === 0) pos.avgYesPrice = 0;
    }
  } else if (side === 'BUY_NO') {
    const noPrice = Number((1 - price).toFixed(3));
    const totalExistingCost = pos.noContracts * pos.avgNoPrice;
    const newCost = quantity * noPrice;
    pos.noContracts += quantity;
    pos.avgNoPrice = pos.noContracts > 0 ? Number(((totalExistingCost + newCost) / pos.noContracts).toFixed(3)) : 0;
  } else if (side === 'SELL_NO') {
    const noPrice = Number((1 - price).toFixed(3));
    if (pos.noContracts > 0) {
      const closeQty = Math.min(pos.noContracts, quantity);
      const pnl = closeQty * (noPrice - pos.avgNoPrice);
      pos.realizedPnl += Number(pnl.toFixed(2));
      pos.noContracts -= closeQty;
      if (pos.noContracts === 0) pos.avgNoPrice = 0;
    }
  }

  return pos;
}
