# MARKET POKER — BOT AI PERSONALITIES & INFORMATION BOUNDARIES

## 1. Five Distinct Bot Engines
1. **The Quant (Alice):** Calculates rigorous Monte Carlo equity from hole cards + board. Exploits market mispricings using Kelly-fraction position sizing and disciplined tight-aggressive poker.
2. **The Trader (Marcus):** Follows momentum breakouts and trading volume. Fast scalper who probes the market with rapid position adjustments.
3. **The Poker Shark (Vance):** High-stakes poker pro. Understands position, range balance, and uses market orders to strategically bluff or trap opponents.
4. **The Degen (Rex):** High-volatility gambler who loves massive pot-sized bets, chases longshot board draws (Straight+ and Board Pairs), and embraces wild variance.
5. **The Market Maker (Nexus MM):** Quotes two-sided tight spreads (bids and asks), captures bid-ask spread, and balances inventory.

## 2. Information Isolation Guardrail
```typescript
interface BotObservation {
  botPlayer: Player; // ONLY their own cards and bankroll
  communityCards: Card[]; // Public cards only
  phase: GamePhase;
  currentHighestBet: number;
  minRaise: number;
  mainPot: number;
  markets: Market[]; // Public prices & order books
  positions: Position[];
  activePlayerCount: number;
}
```
Opponent hole cards are never passed to the bot observation layer.
