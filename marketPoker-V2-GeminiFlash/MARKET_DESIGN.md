# MARKET POKER — MARKET DESIGN & MECHANISM

## 1. Hybrid Market Architecture
Market Poker implements a **Hybrid Central Limit Order Book (CLOB) + Automated Market Maker (AMM)**.

### Order Book Matching Rules
- **Price-Time Priority:** Orders are filled against the best available price in chronological submission order.
- **Order Types:** Limit Orders (Buy YES, Sell YES, Buy NO, Sell NO) and Market Orders (Instant execution against book and AMM).
- **Price Bounds:** Contract prices are bounded between $\$0.01$ (1%) and $\$0.99$ (99%).
- **Settlement:** Binary event contracts settle to $\$1.00$ for winning outcomes and $\$0.00$ for losing outcomes.

### Core Principle: Price $\neq$ True Probability
The game does NOT force mathematically calculated probabilities onto the market. Instead:
- Initial liquidity is seeded by the system Market Maker.
- Human and bot order flow pushes prices up and down.
- Private information allows players to spot mispricings (e.g. holding Pocket Aces while the market is priced at 20¢).
- Market behavior serves as a strategic bluffing mechanism.
