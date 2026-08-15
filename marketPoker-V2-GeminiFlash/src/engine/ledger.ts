import { LedgerEntry, Market, Player, Position } from '../types/game';

export class LedgerEngine {
  private entries: LedgerEntry[] = [];

  constructor(initialEntries: LedgerEntry[] = []) {
    this.entries = [...initialEntries];
  }

  public getEntries(): LedgerEntry[] {
    return [...this.entries];
  }

  public recordTransaction(
    handNumber: number,
    playerId: string,
    category: LedgerEntry['category'],
    amount: number,
    currentCashBalance: number,
    description: string,
    metadata?: Record<string, any>
  ): { entry: LedgerEntry; newBalance: number } {
    const newBalance = Number((currentCashBalance + amount).toFixed(2));
    const entry: LedgerEntry = {
      id: `led-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      timestamp: Date.now(),
      handNumber,
      playerId,
      category,
      amount: Number(amount.toFixed(2)),
      balanceAfter: newBalance,
      description,
      metadata,
    };

    this.entries.push(entry);
    return { entry, newBalance };
  }

  // Calculate Player marked-to-market equity
  public static calculatePlayerEquity(
    player: Player,
    positions: Position[],
    markets: Market[]
  ): {
    liquidCash: number;
    reservedCash: number;
    pokerCommitted: number;
    portfolioValue: number;
    unrealizedPnl: number;
    realizedPnl: number;
    totalEquity: number;
  } {
    const playerPositions = positions.filter((p) => p.playerId === player.id);
    const marketMap = new Map<string, Market>(markets.map((m) => [m.id, m]));

    let portfolioValue = 0;
    let unrealizedPnl = 0;
    let totalRealized = 0;

    for (const pos of playerPositions) {
      totalRealized += pos.realizedPnl;
      const m = marketMap.get(pos.marketId);
      if (!m) continue;

      const yesMarkPrice = m.currentYesPrice;
      const noMarkPrice = m.currentNoPrice;

      if (pos.yesContracts > 0) {
        const value = pos.yesContracts * yesMarkPrice;
        const cost = pos.yesContracts * pos.avgYesPrice;
        portfolioValue += value;
        unrealizedPnl += value - cost;
      }

      if (pos.noContracts > 0) {
        const value = pos.noContracts * noMarkPrice;
        const cost = pos.noContracts * pos.avgNoPrice;
        portfolioValue += value;
        unrealizedPnl += value - cost;
      }
    }

    const liquidCash = Number(player.cashBalance.toFixed(2));
    const reservedCash = Number((player.reservedCash || 0).toFixed(2));
    const pokerCommitted = Number(player.currentBet.toFixed(2));
    portfolioValue = Number(portfolioValue.toFixed(2));
    unrealizedPnl = Number(unrealizedPnl.toFixed(2));
    const totalEquity = Number((liquidCash + reservedCash + pokerCommitted + portfolioValue).toFixed(2));

    return {
      liquidCash,
      reservedCash,
      pokerCommitted,
      portfolioValue,
      unrealizedPnl,
      realizedPnl: totalRealized,
      totalEquity,
    };
  }

  // Economic Invariance check
  public static verifyInvariants(
    players: Player[],
    pot: number,
    systemAccountsTotal: number = 50000 // 5 players * 10,000 credits
  ): boolean {
    const totalPlayerLiquid = players.reduce((sum, p) => sum + p.cashBalance + p.reservedCash + p.currentBet, 0);
    // Invariant holds within floating point precision
    return Math.abs(totalPlayerLiquid - systemAccountsTotal) < 1.0;
  }
}
