export type Suit = 'hearts' | 'diamonds' | 'clubs' | 'spades';
export type Rank = '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | 'T' | 'J' | 'Q' | 'K' | 'A';

export interface Card {
  suit: Suit;
  rank: Rank;
  id: string; // e.g. 'A-spades'
}

export type HandCategory =
  | 'High Card'
  | 'Pair'
  | 'Two Pair'
  | 'Three of a Kind'
  | 'Straight'
  | 'Flush'
  | 'Full House'
  | 'Four of a Kind'
  | 'Straight Flush'
  | 'Royal Flush';

export interface HandEvaluation {
  category: HandCategory;
  rankScore: number; // For exact tie-breaking: category rank + kickers
  description: string;
  bestFiveCards: Card[];
}

export type GamePhase =
  | 'WAITING'
  | 'DEALING'
  | 'PREFLOP_TRADING'
  | 'PREFLOP_BETTING'
  | 'FLOP_REVEAL'
  | 'FLOP_TRADING'
  | 'FLOP_BETTING'
  | 'TURN_REVEAL'
  | 'TURN_TRADING'
  | 'TURN_BETTING'
  | 'RIVER_REVEAL'
  | 'RIVER_TRADING'
  | 'RIVER_BETTING'
  | 'SHOWDOWN'
  | 'MARKET_SETTLEMENT'
  | 'HAND_RESULTS';

export type PlayerStatus = 'active' | 'folded' | 'all-in' | 'sitting-out';

export type BotPersonalityType = 'quant' | 'trader' | 'poker_shark' | 'degen' | 'market_maker';

export interface Player {
  id: string;
  name: string;
  avatar: string;
  isBot: boolean;
  botPersonality?: BotPersonalityType;
  botDescription?: string;
  seatNumber: number;
  cashBalance: number; // Available liquid cash
  reservedCash: number; // Cash locked in open limit buy orders
  currentBet: number; // Current round poker bet
  totalHandBet: number; // Total poker contribution this hand
  loanBalance?: number; // Total borrowed market liquidity debt
  totalInterestPaid?: number; // Cumulative 4% interest auto-paid on winnings
  status: PlayerStatus;
  cards: Card[]; // Hidden from other players in real multiplayer
  showCards?: boolean;
  pokerEquityEstimate?: number; // True mathematical probability (for debug/analytics)
  realizedHandPnl: number;
  lastAction?: string;
}

export type OrderSide = 'BUY_YES' | 'SELL_YES' | 'BUY_NO' | 'SELL_NO';
export type OrderType = 'LIMIT' | 'MARKET';
export type OrderStatus = 'OPEN' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED';

export interface Order {
  id: string;
  marketId: string;
  playerId: string;
  playerName: string;
  side: OrderSide;
  type: OrderType;
  price: number; // 0.01 to 0.99
  quantity: number;
  filledQuantity: number;
  status: OrderStatus;
  createdAt: number;
}

export interface Trade {
  id: string;
  marketId: string;
  buyerId: string;
  sellerId: string;
  price: number; // Execution price 0.01 to 0.99
  quantity: number;
  timestamp: number;
  phase: GamePhase;
  side: OrderSide;
}

export interface Position {
  marketId: string;
  playerId: string;
  yesContracts: number;
  avgYesPrice: number;
  noContracts: number;
  avgNoPrice: number;
  realizedPnl: number;
}

export interface OrderBookLevel {
  price: number;
  quantity: number;
  orderCount: number;
}

export interface MarketPricePoint {
  timestamp: number;
  phase: GamePhase;
  price: number;
  volume: number;
  trueFairValue?: number;
}

export type MarketCategory = 'PLAYER_WIN' | 'HAND_TYPE' | 'BOARD_EVENT' | 'STREET_PROP' | 'TOURNAMENT_PROP';
export type MarketLifespan = 'SHORT_LIVED' | 'LONG_LIVED' | 'CORE_HAND';
export type MarketExpiryStreet = 'PREFLOP' | 'FLOP' | 'TURN' | 'RIVER' | 'SHOWDOWN' | 'TOURNAMENT';

export interface Market {
  id: string;
  name: string;
  ticker: string;
  category: MarketCategory;
  lifespan: MarketLifespan;
  expiryStreet: MarketExpiryStreet;
  iconTag?: string; // e.g. '⚡', '🎯', '♠', '🔥'
  targetPlayerId?: string;
  description: string;
  currentYesPrice: number; // 0.01 - 0.99
  currentNoPrice: number; // 1 - currentYesPrice
  bestBid: number; // Highest buy YES price
  bestAsk: number; // Lowest sell YES price (or 1 - best bid NO)
  volume: number;
  liquidityPool: {
    yesShares: number;
    noShares: number;
    kConstant: number;
  };
  bids: Order[]; // Buy YES limit orders sorted desc by price
  asks: Order[]; // Sell YES limit orders sorted asc by price
  noBids: Order[]; // Buy NO limit orders
  noAsks: Order[]; // Sell NO limit orders
  priceHistory: MarketPricePoint[];
  resolved: boolean;
  outcome?: boolean; // true = YES won, false = NO won
  resolutionText?: string;
}

export interface LedgerEntry {
  id: string;
  timestamp: number;
  handNumber: number;
  playerId: string;
  category:
    | 'INITIAL_CAPITAL'
    | 'POKER_BET'
    | 'POKER_WIN'
    | 'MARKET_BUY'
    | 'MARKET_SELL'
    | 'MARKET_SETTLE'
    | 'ORDER_LOCK'
    | 'ORDER_UNLOCK'
    | 'SYNERGY_BOUNTY'
    | 'SYNERGY_PENALTY'
    | 'MARKET_LOAN_BORROW'
    | 'MARKET_LOAN_INTEREST'
    | 'MARKET_LOAN_REPAY';
  amount: number; // Positive = credit, negative = debit
  balanceAfter: number;
  description: string;
  metadata?: Record<string, any>;
}

export interface SkillRadar {
  cardSkill: number; // 0 - 100
  marketSkill: number; // 0 - 100
  forecastingSkill: number; // 0 - 100
  riskSkill: number; // 0 - 100
  psychologySkill: number; // 0 - 100
  overallRating: number; // e.g. 1500 - 2500
}

export interface DualSynergyBonus {
  triggered: boolean;
  multiplierPct: number;
  bonusAmount: number;
  winningPlayerId: string;
  winningPlayerName: string;
  winningContractsCount: number;
  fundedBy: { playerId: string; name: string; amount: number }[];
}

export interface PostHandAnalytics {
  handNumber: number;
  tournamentLevel: number;
  synergyMultiplierPct: number;
  winners: { playerId: string; name: string; amount: number; handDescription: string }[];
  pokerPot: number;
  userPokerPnl: number;
  userMarketPnl: number;
  userTotalPnl: number;
  userEndingEquity: number;
  loanInterestDeducted?: number; // 4% market loan interest auto-paid from winnings
  loanBalance?: number; // Outstanding loan debt
  dualSynergyBonus?: DualSynergyBonus;
  bestTrade?: {
    marketName: string;
    action: string;
    entryPrice: number;
    estimatedFairValue: number;
    edgePercentage: number;
    pnl: number;
  };
  marketOutcomes: {
    marketId: string;
    marketName: string;
    outcome: boolean;
    settlementPrice: number;
    lifespan?: MarketLifespan;
  }[];
  skillDeltas: {
    category: keyof SkillRadar;
    delta: number;
  }[];
}

export interface GameEvent {
  id: string;
  timestamp: number;
  phase: GamePhase;
  type: 'POKER' | 'MARKET' | 'SYSTEM' | 'BOT_SIGNAL' | 'TOURNAMENT' | 'SYNERGY';
  text: string;
  highlight?: boolean;
}

export interface GameTableState {
  tableId?: string;
  tableName?: string;
  tableTheme?: 'emerald' | 'sapphire' | 'amber' | 'ruby';
  raiseCount?: number;
  handNumber: number;
  tournamentLevel: number;
  handsInCurrentLevel: number;
  handsPerLevel: number;
  synergyMultiplierPct: number; // e.g. 15%, 30%, 50%, 75%, 100%
  dealerIndex: number;
  currentTurnIndex: number;
  phase: GamePhase;
  phaseTimeRemaining: number;
  turnTimeRemaining?: number; // Live seconds remaining for current player action
  totalTurnTime?: number; // Total seconds allocated for this turn (e.g. 15s)
  smallBlind: number;
  bigBlind: number;
  currentHighestBet: number;
  minRaise: number;
  mainPot: number;
  sidePots: { amount: number; eligiblePlayers: string[] }[];
  deck: Card[];
  communityCards: Card[];
  players: Player[];
  markets: Market[];
  positions: Position[];
  openOrders: Order[];
  recentTrades: Trade[];
  events: GameEvent[];
  ledger: LedgerEntry[];
  lastHandAnalytics?: PostHandAnalytics;
  isPaused: boolean;
  autoPlayNextHand: boolean;
  eliminatedPlayerIds: string[];
}
