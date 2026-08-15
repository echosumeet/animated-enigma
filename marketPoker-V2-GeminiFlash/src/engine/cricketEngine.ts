import {
  LiveCricketMatch,
  CricketSportsMarket,
  UserSportsBet,
  CricketBatsman,
  CricketBowler,
} from '../types/cricket';

export class CricketEngine {
  // Default International Match Data with High Realism
  public static getInitialMatches(): LiveCricketMatch[] {
    return [
      {
        id: 'cric-ind-aus-2026',
        matchTitle: 'India vs Australia - 3rd ODI',
        seriesName: 'Border-Gavaskar ODI Series 2026',
        matchType: 'ODI',
        status: 'LIVE',
        statusText: 'IND need 42 runs in 36 balls | RRR: 7.00 | CRR: 5.68',
        venue: 'Melbourne Cricket Ground, Melbourne',
        team1: {
          name: 'Australia',
          shortName: 'AUS',
          flag: '🇦🇺',
          score: '288/8 (50.0 ov)',
          overs: '50.0',
          runs: 288,
          wickets: 8,
          isBatting: false,
          isBowling: true,
        },
        team2: {
          name: 'India',
          shortName: 'IND',
          flag: '🇮🇳',
          score: '247/4 (44.0 ov)',
          overs: '44.0',
          runs: 247,
          wickets: 4,
          isBatting: true,
          isBowling: false,
        },
        currentBatsmen: [
          {
            name: 'Virat Kohli',
            runs: 76,
            balls: 68,
            fours: 7,
            sixes: 2,
            strikeRate: 111.76,
            onStrike: true,
          },
          {
            name: 'Hardik Pandya',
            runs: 34,
            balls: 24,
            fours: 3,
            sixes: 1,
            strikeRate: 141.67,
            onStrike: false,
          },
        ],
        currentBowler: {
          name: 'Pat Cummins',
          overs: '8.0',
          maidens: 0,
          runs: 48,
          wickets: 2,
          economy: 6.0,
        },
        recentBalls: ['1', '4', '2', '0', '6', '1'],
        currentOverNumber: 44,
        currentBallInOver: 0,
        crr: 5.61,
        rrr: 7.0,
        target: 289,
        winProbability: {
          team1Pct: 34,
          team2Pct: 66,
        },
        lastUpdated: Date.now(),
        markets: [
          {
            id: 'cric-mkt-101',
            matchId: 'cric-ind-aus-2026',
            title: 'Match Winner: India to Win vs Australia',
            category: 'MATCH_WINNER',
            description: 'India needs 42 runs from 36 deliveries with 6 wickets in hand.',
            yesPrice: 0.66,
            noPrice: 0.34,
            yesOddsMultiplier: 1.51,
            noOddsMultiplier: 2.94,
            volume: 14200,
            resolved: false,
          },
          {
            id: 'cric-mkt-102',
            matchId: 'cric-ind-aus-2026',
            title: 'Over 45: Over 7.5 Total Runs Scored',
            category: 'OVER_RUNS',
            description: 'Will 8 or more runs be scored off Over 45 by Pat Cummins?',
            yesPrice: 0.55,
            noPrice: 0.45,
            yesOddsMultiplier: 1.82,
            noOddsMultiplier: 2.22,
            volume: 8750,
            resolved: false,
            targetMetric: 'over_runs',
            targetValue: 7.5,
            closesInOver: '45.0',
          },
          {
            id: 'cric-mkt-103',
            matchId: 'cric-ind-aus-2026',
            title: 'Virat Kohli to Score Century (100+ Runs)',
            category: 'PLAYER_MILESTONE',
            description: 'Kohli currently batting on 76* (68 balls). Needs 24 more runs.',
            yesPrice: 0.72,
            noPrice: 0.28,
            yesOddsMultiplier: 1.39,
            noOddsMultiplier: 3.57,
            volume: 19800,
            resolved: false,
            targetMetric: 'batsman_score',
            targetValue: 100,
          },
          {
            id: 'cric-mkt-104',
            matchId: 'cric-ind-aus-2026',
            title: 'Wicket to Fall in Next 2 Overs (Overs 45-46)',
            category: 'WICKET_FALL',
            description: 'Will Australia claim a wicket before the end of over 46.0?',
            yesPrice: 0.42,
            noPrice: 0.58,
            yesOddsMultiplier: 2.38,
            noOddsMultiplier: 1.72,
            volume: 6400,
            resolved: false,
            closesInOver: '46.0',
          },
          {
            id: 'cric-mkt-105',
            matchId: 'cric-ind-aus-2026',
            title: 'Six to be Hit in Over 45',
            category: 'SIX_IN_OVER',
            description: 'Will Hardik Pandya or Virat Kohli strike a 6 in over 45?',
            yesPrice: 0.38,
            noPrice: 0.62,
            yesOddsMultiplier: 2.63,
            noOddsMultiplier: 1.61,
            volume: 5300,
            resolved: false,
            closesInOver: '45.0',
          },
        ],
      },
      {
        id: 'cric-eng-rsa-2026',
        matchTitle: 'England vs South Africa - 2nd T20I',
        seriesName: 'International T20 Tri-Series',
        matchType: 'T20I',
        status: 'LIVE',
        statusText: 'RSA need 38 runs in 18 balls | RRR: 12.67',
        venue: "Lord's Cricket Ground, London",
        team1: {
          name: 'England',
          shortName: 'ENG',
          flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
          score: '194/5 (20.0 ov)',
          overs: '20.0',
          runs: 194,
          wickets: 5,
          isBatting: false,
          isBowling: true,
        },
        team2: {
          name: 'South Africa',
          shortName: 'RSA',
          flag: '🇿🇦',
          score: '157/4 (17.0 ov)',
          overs: '17.0',
          runs: 157,
          wickets: 4,
          isBatting: true,
          isBowling: false,
        },
        currentBatsmen: [
          {
            name: 'Heinrich Klaasen',
            runs: 48,
            balls: 22,
            fours: 4,
            sixes: 4,
            strikeRate: 218.18,
            onStrike: true,
          },
          {
            name: 'David Miller',
            runs: 29,
            balls: 16,
            fours: 2,
            sixes: 2,
            strikeRate: 181.25,
            onStrike: false,
          },
        ],
        currentBowler: {
          name: 'Jofra Archer',
          overs: '3.0',
          maidens: 0,
          runs: 24,
          wickets: 2,
          economy: 8.0,
        },
        recentBalls: ['6', '1', '4', '1', '2', '6'],
        currentOverNumber: 17,
        currentBallInOver: 0,
        crr: 9.23,
        rrr: 12.67,
        target: 195,
        winProbability: {
          team1Pct: 52,
          team2Pct: 48,
        },
        lastUpdated: Date.now(),
        markets: [
          {
            id: 'cric-mkt-201',
            matchId: 'cric-eng-rsa-2026',
            title: 'Match Winner: South Africa to Win vs England',
            category: 'MATCH_WINNER',
            description: 'South Africa need 38 off 18 balls with Klaasen & Miller at crease.',
            yesPrice: 0.48,
            noPrice: 0.52,
            yesOddsMultiplier: 2.08,
            noOddsMultiplier: 1.92,
            volume: 24500,
            resolved: false,
          },
          {
            id: 'cric-mkt-202',
            matchId: 'cric-eng-rsa-2026',
            title: 'Over 18 (Jofra Archer): 12+ Runs Conceded',
            category: 'OVER_RUNS',
            description: 'Will South Africa score 12 or more runs in death over 18?',
            yesPrice: 0.58,
            noPrice: 0.42,
            yesOddsMultiplier: 1.72,
            noOddsMultiplier: 2.38,
            volume: 11200,
            resolved: false,
            closesInOver: '18.0',
          },
          {
            id: 'cric-mkt-203',
            matchId: 'cric-eng-rsa-2026',
            title: 'Heinrich Klaasen to Score 50+ Runs (Currently 48*)',
            category: 'PLAYER_MILESTONE',
            description: 'Needs just 2 runs to reach half-century in 23 balls.',
            yesPrice: 0.88,
            noPrice: 0.12,
            yesOddsMultiplier: 1.14,
            noOddsMultiplier: 8.33,
            volume: 16800,
            resolved: false,
          },
        ],
      },
      {
        id: 'cric-pak-nz-2026',
        matchTitle: 'Pakistan vs New Zealand - 1st T20I',
        seriesName: 'T20 International Championship',
        matchType: 'T20I',
        status: 'LIVE',
        statusText: 'PAK 112/2 (12.4 ov) | CRR: 8.84 | Projected: 182',
        venue: 'Dubai International Cricket Stadium, Dubai',
        team1: {
          name: 'Pakistan',
          shortName: 'PAK',
          flag: '🇵🇰',
          score: '112/2 (12.4 ov)',
          overs: '12.4',
          runs: 112,
          wickets: 2,
          isBatting: true,
          isBowling: false,
        },
        team2: {
          name: 'New Zealand',
          shortName: 'NZ',
          flag: '🇳🇿',
          score: 'Yet to bat',
          overs: '0.0',
          runs: 0,
          wickets: 0,
          isBatting: false,
          isBowling: true,
        },
        currentBatsmen: [
          {
            name: 'Babar Azam',
            runs: 54,
            balls: 39,
            fours: 6,
            sixes: 1,
            strikeRate: 138.46,
            onStrike: true,
          },
          {
            name: 'Mohammad Rizwan',
            runs: 41,
            balls: 30,
            fours: 4,
            sixes: 1,
            strikeRate: 136.67,
            onStrike: false,
          },
        ],
        currentBowler: {
          name: 'Mitchell Santner',
          overs: '2.4',
          maidens: 0,
          runs: 19,
          wickets: 1,
          economy: 7.12,
        },
        recentBalls: ['1', '0', '4', '1', '1'],
        currentOverNumber: 12,
        currentBallInOver: 4,
        crr: 8.84,
        winProbability: {
          team1Pct: 61,
          team2Pct: 39,
        },
        lastUpdated: Date.now(),
        markets: [
          {
            id: 'cric-mkt-301',
            matchId: 'cric-pak-nz-2026',
            title: 'Pakistan 1st Innings Total Over 175.5 Runs',
            category: 'TOTAL_INNINGS',
            description: 'Pakistan projected 182 at current run rate of 8.84.',
            yesPrice: 0.65,
            noPrice: 0.35,
            yesOddsMultiplier: 1.54,
            noOddsMultiplier: 2.86,
            volume: 9800,
            resolved: false,
          },
          {
            id: 'cric-mkt-302',
            matchId: 'cric-pak-nz-2026',
            title: 'Babar Azam to score 75+ runs (Currently 54*)',
            category: 'PLAYER_MILESTONE',
            description: 'Needs 21 runs off remaining 44 balls.',
            yesPrice: 0.58,
            noPrice: 0.42,
            yesOddsMultiplier: 1.72,
            noOddsMultiplier: 2.38,
            volume: 7200,
            resolved: false,
          },
        ],
      },
    ];
  }

  // Simulate Next Ball & Settle Relevant Short-Lived Markets
  public static simulateNextBall(
    match: LiveCricketMatch,
    bets: UserSportsBet[]
  ): {
    updatedMatch: LiveCricketMatch;
    updatedBets: UserSportsBet[];
    settledPnl: number;
    ballEventText: string;
  } {
    const outcomes = ['0', '1', '1', '2', '4', '6', '1', 'W', '2', '4'];
    const randomBall = outcomes[Math.floor(Math.random() * outcomes.length)];

    let ballRuns = 0;
    let isWicket = false;
    let isSix = false;

    if (randomBall === 'W') {
      isWicket = true;
    } else if (randomBall === '6') {
      ballRuns = 6;
      isSix = true;
    } else if (randomBall === '4') {
      ballRuns = 4;
    } else {
      ballRuns = parseInt(randomBall, 10) || 0;
    }

    const battingTeam = match.team1.isBatting ? { ...match.team1 } : { ...match.team2 };
    const newTotalRuns = battingTeam.runs + ballRuns;
    const newWickets = isWicket ? Math.min(10, battingTeam.wickets + 1) : battingTeam.wickets;

    let newBallInOver = match.currentBallInOver + 1;
    let newOverNumber = match.currentOverNumber;
    let overCompleted = false;

    if (newBallInOver >= 6) {
      newBallInOver = 0;
      newOverNumber += 1;
      overCompleted = true;
    }

    const currentOverStr = `${newOverNumber}.${newBallInOver}`;
    battingTeam.runs = newTotalRuns;
    battingTeam.wickets = newWickets;
    battingTeam.score = `${newTotalRuns}/${newWickets} (${currentOverStr} ov)`;
    battingTeam.overs = currentOverStr;

    // Update batsmen
    const updatedBatsmen = match.currentBatsmen.map((b, idx) => {
      if (idx === 0) {
        // Striker
        const runs = b.runs + ballRuns;
        const balls = b.balls + 1;
        const fours = ballRuns === 4 ? b.fours + 1 : b.fours;
        const sixes = ballRuns === 6 ? b.sixes + 1 : b.sixes;
        const sr = Number(((runs / Math.max(1, balls)) * 100).toFixed(1));
        return {
          ...b,
          runs,
          balls,
          fours,
          sixes,
          strikeRate: sr,
        };
      }
      return b;
    });

    const recentBalls = [...match.recentBalls.slice(1), randomBall];

    // Win probability shift
    let team1Prob = match.winProbability.team1Pct;
    if (match.team2.isBatting) {
      // Team 2 chasing
      if (isWicket) team1Prob = Math.min(95, team1Prob + 7);
      else if (ballRuns >= 4) team1Prob = Math.max(5, team1Prob - 5);
      else team1Prob = Math.min(95, team1Prob + 1);
    } else {
      if (isWicket) team1Prob = Math.max(5, team1Prob - 6);
      else if (ballRuns >= 4) team1Prob = Math.min(95, team1Prob + 4);
    }
    const team2Prob = 100 - team1Prob;

    // Settle markets and bets
    let settledPnl = 0;
    const updatedMarkets = match.markets.map((m) => {
      let resolved = m.resolved;
      let outcome = m.outcome;
      let resolutionText = m.resolutionText;

      // Check Over Runs Market
      if (overCompleted && m.category === 'OVER_RUNS' && !resolved) {
        resolved = true;
        // In realistic simulation, check if over had > 7.5 runs
        const overRunsCount = Math.floor(Math.random() * 8) + ballRuns + 3;
        outcome = overRunsCount >= 8;
        resolutionText = `Over ended with ${overRunsCount} runs scored (${outcome ? 'YES' : 'NO'})`;
      }

      // Check Six In Over
      if (m.category === 'SIX_IN_OVER' && isSix && !resolved) {
        resolved = true;
        outcome = true;
        resolutionText = `SIX struck by batsman! YES won.`;
      } else if (overCompleted && m.category === 'SIX_IN_OVER' && !resolved) {
        resolved = true;
        outcome = false;
        resolutionText = `No six hit in completed over. NO won.`;
      }

      // Check Wicket Fall
      if (m.category === 'WICKET_FALL' && isWicket && !resolved) {
        resolved = true;
        outcome = true;
        resolutionText = `Wicket fell! YES won.`;
      }

      // Check Batsman Century
      if (m.category === 'PLAYER_MILESTONE' && !resolved) {
        const striker = updatedBatsmen[0];
        if (m.targetValue && striker.runs >= m.targetValue) {
          resolved = true;
          outcome = true;
          resolutionText = `${striker.name} reached ${m.targetValue} runs! YES won.`;
        }
      }

      // Dynamically shift odds if still open
      let currentYes = m.yesPrice;
      if (!resolved) {
        if (m.category === 'MATCH_WINNER') {
          currentYes = team2Prob / 100;
        } else if (ballRuns >= 4) {
          currentYes = Math.min(0.95, currentYes + 0.05);
        } else if (isWicket) {
          currentYes = Math.max(0.05, currentYes - 0.08);
        }
      }

      const currentNo = Number((1 - currentYes).toFixed(2));
      const yesOdds = Number((1 / Math.max(0.01, currentYes)).toFixed(2));
      const noOdds = Number((1 / Math.max(0.01, currentNo)).toFixed(2));

      return {
        ...m,
        yesPrice: Number(currentYes.toFixed(2)),
        noPrice: currentNo,
        yesOddsMultiplier: yesOdds,
        noOddsMultiplier: noOdds,
        resolved,
        outcome,
        resolutionText,
      };
    });

    // Settle user bets
    const updatedBets = bets.map((bet) => {
      if (bet.status !== 'OPEN') return bet;
      const targetMkt = updatedMarkets.find((m) => m.id === bet.marketId);
      if (targetMkt && targetMkt.resolved && targetMkt.outcome !== undefined) {
        const isWin = (bet.side === 'BUY_YES' && targetMkt.outcome === true) ||
                      (bet.side === 'BUY_NO' && targetMkt.outcome === false);
        const payout = isWin ? bet.potentialPayout : 0;
        const pnl = isWin ? payout - bet.stake : -bet.stake;
        settledPnl += isWin ? payout : 0;
        return {
          ...bet,
          status: isWin ? ('WON' as const) : ('LOST' as const),
          pnl,
          resolvedOver: currentOverStr,
        };
      }
      return bet;
    });

    const updatedMatch: LiveCricketMatch = {
      ...match,
      team1: match.team1.isBatting ? battingTeam : match.team1,
      team2: match.team2.isBatting ? battingTeam : match.team2,
      currentBatsmen: updatedBatsmen,
      recentBalls,
      currentOverNumber: newOverNumber,
      currentBallInOver: newBallInOver,
      winProbability: {
        team1Pct: team1Prob,
        team2Pct: team2Prob,
      },
      markets: updatedMarkets,
      lastUpdated: Date.now(),
    };

    const ballEventText = isWicket
      ? `🚨 WICKET! Batsman dismissed on ball ${currentOverStr}!`
      : isSix
      ? `🔥 MAXIMUM SIX! Smashed over the boundary ropes!`
      : ballRuns === 4
      ? `🏏 FOUR! Driven crisply through the covers!`
      : `Ball ${currentOverStr}: ${ballRuns} run(s)`;

    return {
      updatedMatch,
      updatedBets,
      settledPnl,
      ballEventText,
    };
  }
}
