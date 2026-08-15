# MARKET POKER — TESTING & VALIDATION SPECIFICATION

## Test Coverage
- **Poker Engine Tests:** Fisher-Yates deck uniqueness, 7-card evaluator hierarchy, kicker comparisons, Monte Carlo equity bounds $[0.01, 0.99]$.
- **Market Engine Tests:** Limit and Market order execution, order book sorting (Bids desc, Asks asc), partial fills, position tracking, realized and unrealized P&L calculations.
- **Settlement Tests:** Deterministic resolution of Player Win, Straight+, and Board Pairs contracts. Idempotency guarantees.
- **Economic Invariant Tests:** Verifies that Total System Credits across all 5 wallets + Pot equals 50,000 credits before and after 100-to-1,000 hand simulations.
- **Simulation Suite:** Automated 100-to-1,000 hand headless Monte Carlo tournament verifying game balance and bot win-rate distributions.
