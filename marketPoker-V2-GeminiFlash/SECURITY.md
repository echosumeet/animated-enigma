# MARKET POKER — SECURITY, INTEGRITY & THREAT MODEL

## 1. Threat Model & Mitigations
- **Hole Card Leakage Prevention:** In multiplayer/bot architectures, opponent cards are redacted from state payloads before rendering or bot observation evaluation.
- **Double Spending & Invariant Violations:** All poker bets and market buys are checked against available liquid cash. The double-entry ledger verifies total circulating credits continuously.
- **Idempotent Settlements:** Settlement calls flag markets as resolved and compute deterministic outcomes, ensuring multiple calls cannot trigger duplicate credit distributions.
- **Input Sanitation:** Order prices are strictly clamped in the range $[0.01, 0.99]$ with positive integer quantities.
- **Play-Money Only:** Virtual play credits only. No cryptocurrency, real-world deposits, or withdrawals are permitted in the MVP.
