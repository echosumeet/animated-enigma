# MARKET POKER — ARCHITECTURAL DECISIONS & TRADEOFFS

1. **Unified Bankroll:** Poker bets and prediction market contracts pull from the exact same cash pool ($10,000 credits). This forces strategic capital allocation tradeoffs between betting in the pot and trading contracts.
2. **Procedural Web Audio Synthesizer:** Built a custom Web Audio API synthesizer for casino cards, chips, order fills, and victory fanfares. This ensures zero external asset dependencies, zero network latency, and 100% reliable sound effects.
3. **Hybrid CLOB + AMM Market Engine:** Implemented continuous double-auction limit order books paired with constant-product AMM liquidity pools to ensure instant market fills even with low liquidity.
4. **Monte Carlo Equity Estimation:** Utilized 150-iteration fast Monte Carlo simulation for real-time Bot evaluations and Dev Inspector true probability estimation.
