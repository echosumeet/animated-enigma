# MARKET POKER — ARCHITECTURE SPECIFICATION

## System Diagram & Modules

```
[ Frontend: React 19 + Tailwind CSS + Framer Motion ]
                     │
         [ State Machine Orchestrator ]
        ┌────────────┼────────────┐
        ▼            ▼            ▼
 [ Poker Engine ] [ Market CLOB ] [ Ledger Engine ]
 (Deck, 7-card    (Order Book,     (Double-Entry,
  Evaluator, MC    AMM Liquidity,   Equity Invariant,
  Equity Model)    Matching Engine) P&L Reconciliation)
        ▲            ▲            ▲
        └────────────┼────────────┘
              [ 5 Bot Engines ]
        (Quant, Trader, Shark, Degen, MM)
        [ Private Observation Boundary ]
```

## Security & Information Boundaries
- **Strict Bot Observation Isolation:** Bots receive only public table state + their own 2 private cards. Under no circumstances can bots peek at opponent hole cards.
- **Authoritative State Transitions:** Phases (Dealing, Trading, Betting, Flop, Turn, River, Showdown, Settlement) are governed strictly by the game engine.
- **Double-Entry Ledger:** Every credit transaction is recorded with an immutable timestamp, hand index, category, and audit metadata.
