# MARKET POKER — GAME DESIGN DOCUMENT

## 1. Overview & Vision
Market Poker fuses **Texas Hold'em Poker** with **Continuous Order-Book Prediction Markets**, **Game Theory**, and **Shared Bankroll Management**.

## 2. The Core Strategic Loop
1. **Dealing & Prior Odds:** 5 players each receive 2 private hole cards. Markets initialize with a 20% baseline prior.
2. **Private Information & Exploitation:** Players evaluate their private card equity. If the public market underprices their win probability, they can buy YES contracts at a discount.
3. **Market Signaling & Bluffing:** Aggressive market buys push contract prices higher, serving as a visible strength signal to induce folds from opponents.
4. **Community Board Evolution:** Flop, Turn, and River arrive as public macroeconomic shocks, triggering rapid price discovery.
5. **Showdown & Dual Settlement:** The poker pot is awarded to the best 5-card hand, while event contracts settle deterministically to $1.00 (YES) or $0.00 (NO).
6. **Quant Post-Hand Analytics:** Tracks P&L, EV edge, calibration score, and a 5-dimensional skill rating (Card, Market, Forecasting, Risk, Psychology).

## 3. The 7 Active Prediction Markets
1. **Player 1 (You) Wins:** Resolves YES if Player 1 wins the poker pot.
2. **Alice (The Quant) Wins:** Resolves YES if Alice wins.
3. **Marcus (The Trader) Wins:** Resolves YES if Marcus wins.
4. **Vance (The Shark) Wins:** Resolves YES if Vance wins.
5. **Rex (The Degen) Wins:** Resolves YES if Rex wins.
6. **Straight or Better Wins:** Resolves YES if the winning hand is a Straight, Flush, Full House, Quads, or Straight Flush.
7. **Board Pairs:** Resolves YES if the 5 community cards contain at least one pair.

## 4. Shared Bankroll Invariant
All players begin with **10,000 play-money credits**. Every poker bet and market order is deducted from the exact same cash pool. Total Equity is continuously computed as:
$$\text{Total Equity} = \text{Liquid Cash} + \text{Reserved Limit Cash} + \text{Poker Commitments} + \text{Marked-to-Market Portfolio Value}$$
