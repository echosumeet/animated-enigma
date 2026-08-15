# ALGORITHM PROVENANCE REGISTER

### 1. 7-Card Poker Hand Ranking & Combinatorial Subsets
- **Purpose:** Evaluate the best 5-card Texas Hold'em hand from 7 cards.
- **Implementation:** Original implementation (`src/engine/poker.ts`).
- **Reference:** Standard combinatorial evaluation ($7 \choose 5 = 21$ subsets) with mathematical tier multipliers for 100% accurate kicker tie-breaking.
- **License:** MIT / Original.

### 2. Monte Carlo Equity Estimator
- **Purpose:** Compute accurate poker win probability against $N$ opponents with unknown hole cards.
- **Implementation:** Original implementation (`src/engine/poker.ts`).
- **Reference:** Standard Monte Carlo sampling over remaining deck.
- **License:** MIT / Original.

### 3. Continuous Limit Order Book & AMM Matching
- **Purpose:** Price-time priority matching with constant-product AMM fallback.
- **Implementation:** Original implementation (`src/engine/market.ts`).
- **Reference:** Standard continuous double auction (CDA) & $x \cdot y = k$ equations.
- **License:** MIT / Original.

### 4. Double-Entry Immutable Ledger
- **Purpose:** Record and reconcile all virtual credit movements and verify economic invariants.
- **Implementation:** Original implementation (`src/engine/ledger.ts`).
- **Reference:** Standard double-entry accounting principles.
- **License:** MIT / Original.
