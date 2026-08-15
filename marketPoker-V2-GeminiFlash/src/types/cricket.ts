export type CricketMatchType = 'T20I' | 'ODI' | 'TEST' | 'T20_LEAGUE';

export type CricketMatchStatus = 'LIVE' | 'INNINGS_BREAK' | 'STUMPS' | 'MATCH_ENDED' | 'UPCOMING';

export interface CricketBatsman {
  name: string;
  runs: number;
  balls: number;
  fours: number;
  sixes: number;
  strikeRate: number;
  onStrike: boolean;
}

export interface CricketBowler {
  name: string;
  overs: string;
  maidens: number;
  runs: number;
  wickets: number;
  economy: number;
}

export interface CricketTeam {
  name: string;
  shortName: string;
  flag: string;
  score: string; // e.g. "248/4"
  overs: string; // e.g. "38.2"
  runs: number;
  wickets: number;
  isBatting: boolean;
  isBowling: boolean;
}

export type CricketMarketCategory =
  | 'MATCH_WINNER'
  | 'OVER_RUNS'
  | 'WICKET_FALL'
  | 'PLAYER_MILESTONE'
  | 'SIX_IN_OVER'
  | 'TOTAL_INNINGS';

export interface CricketSportsMarket {
  id: string;
  matchId: string;
  title: string;
  category: CricketMarketCategory;
  description: string;
  yesPrice: number; // 0.01 to 0.99 (implied prob)
  noPrice: number; // 1 - yesPrice
  yesOddsMultiplier: number; // e.g. 1.55x
  noOddsMultiplier: number; // e.g. 2.80x
  volume: number;
  resolved: boolean;
  outcome?: boolean; // true = YES won, false = NO won
  resolutionText?: string;
  targetMetric?: string;
  targetValue?: number;
  closesInOver?: string;
}

export interface LiveCricketMatch {
  id: string;
  matchTitle: string;
  seriesName: string;
  matchType: CricketMatchType;
  status: CricketMatchStatus;
  statusText: string;
  venue: string;
  team1: CricketTeam;
  team2: CricketTeam;
  currentBatsmen: CricketBatsman[];
  currentBowler: CricketBowler;
  recentBalls: string[]; // e.g. ['1', '4', '0', 'W', '6', '1']
  currentOverNumber: number;
  currentBallInOver: number;
  crr: number; // Current run rate
  rrr?: number; // Required run rate (if chasing)
  target?: number;
  winProbability: {
    team1Pct: number;
    team2Pct: number;
  };
  markets: CricketSportsMarket[];
  lastUpdated: number;
  isLiveScannedWithGemini?: boolean;
  groundingSources?: { title: string; url: string }[];
}

export interface UserSportsBet {
  id: string;
  matchId: string;
  matchTitle: string;
  marketId: string;
  marketTitle: string;
  side: 'BUY_YES' | 'BUY_NO';
  stake: number; // In game credits/dollars
  price: number; // 0.01 - 0.99
  shares: number;
  potentialPayout: number;
  oddsMultiplier: number;
  status: 'OPEN' | 'WON' | 'LOST' | 'CASHED_OUT';
  pnl: number;
  timestamp: number;
  resolvedOver?: string;
}
