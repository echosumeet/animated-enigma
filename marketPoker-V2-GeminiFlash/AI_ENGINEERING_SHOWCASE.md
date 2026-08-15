# Market Poker & Live Sports Arena: AI Engineering Showcase

> **Mapping to Andrew Ng's AI Engineering Skills Map (Published August 2026)**
> *A Production-Grade Full-Stack Tournament Texas Hold'em, Continuous Prediction Market, and Google-Grounded Live Cricket Micro-Betting Platform.*

---

## 1. Title & Overview

**Project Name:** Market Poker & Live Sports Arena
**One-Sentence Purpose:** A real-time competitive gaming platform that synthesizes 5-seat tournament Texas Hold'em, high-frequency prediction markets (Kalshi/Polymarket style), and Google Search Grounded live international cricket score intelligence with dual-victory synergy bounties.

### Executive Summary
Market Poker redefines tactical online gaming by marrying Texas Hold'em decision theory with quantitative continuous order-book prediction markets and real-world live sports event scanning. As poker hands unfold across Preflop, Flop, Turn, and River, micro-markets dynamically price board outcomes, player actions, and hand strengths in real time. Simultaneously, a dedicated **Live Sports Bet Arena** powered by **Gemini 3.7 Flash and Google Search Grounding** scans real-world ongoing international cricket matches (ICC, Border-Gavaskar, T20 Tri-Series), delivering instant ball-by-ball updates, live odds, and instant micro-settlements.

This codebase demonstrates modern **AI Engineering** across all four pillars of Andrew Ng’s AI Engineering Skills framework: combining server-side grounded model inference, deterministic financial ledger engines, rigorous agentic verification loops, and deep product intuition.

---

## 2. System Architecture Diagram

```
+--------------------------------------------------------------------------------------------------+
|                                    CLIENT APPLICATION (React 18 + Vite)                          |
|                                                                                                  |
|  +---------------------------+  +-------------------------------+  +--------------------------+  |
|  |     Poker Table & HUD     |  |    Live Prediction Markets    |  |     Live Sports Arena    |  |
|  |  - 5-Seat AI Tournament   |  |  - Continuous Order Books     |  |  - Google Cricket Scores |  |
|  |  - Blind Escalation Levels|  |  - 1-Click Quick Betting      |  |  - Ball-by-Ball Sim      |  |
|  |  - Dual Synergy Bounties  |  |  - Street Level Settlement    |  |  - Dynamic Over/Win Odds |  |
|  +---------------------------+  +-------------------------------+  +--------------------------+  |
|               |                                 |                                |               |
|               +---------------------------------+--------------------------------+               |
|                                                 |                                                |
|                                 +-------------------------------+                                |
|                                 |  Double-Entry Ledger Engine   |                                |
|                                 |  - Real-time Equity Valuation |                                |
|                                 |  - Cash / Chip Segregation    |                                |
|                                 |  - Zero-Sum Bounty Allocation |                                |
|                                 +-------------------------------+                                |
|                                                 |                                                |
+-------------------------------------------------|------------------------------------------------+
                                                  | HTTP / JSON API
                                                  v
+--------------------------------------------------------------------------------------------------+
|                                BACKEND SERVER (Node.js + Express)                                |
|                                                                                                  |
|  +---------------------------+  +-------------------------------+  +--------------------------+  |
|  |      /api/health          |  |      /api/cricket/live        |  |    /api/cricket/scan     |  |
|  |  Health & Liveness Check  |  |   In-Memory Match Caching     |  |   Gemini 3.7 Search      |  |
|  |                           |  |   & Snapshot Distribution     |  |   Grounding Pipeline     |  |
|  +---------------------------+  +-------------------------------+  +--------------------------+  |
|                                                                                 |                |
+---------------------------------------------------------------------------------|----------------+
                                                                                  |
                                                                                  v
                                                            +-----------------------------------+
                                                            |    GOOGLE GENAI SDK (Server)      |
                                                            |  - Model: gemini-3.7-flash        |
                                                            |  - Tool: googleSearch ({})        |
                                                            |  - Search Grounding Citations     |
                                                            |  - Structured JSON Output         |
                                                            +-----------------------------------+
                                                                                  |
                                                                                  v
                                                            +-----------------------------------+
                                                            |   Google Real-Time Web Search     |
                                                            |   (Live International Scores)     |
                                                            +-----------------------------------+
```

---

## 3. The Four AI Engineering Skills — How They Appeared

### I. Building and Deploying AI Applications
*Mastering unpredictable model behavior with rigorous prompt steering, grounding, and fallback governance.*

- **Google Search Grounding**: Live sports data changes by the second. Instead of relying on static parametric model knowledge, the system binds `gemini-3.7-flash` with the `googleSearch: {}` tool on the backend server (`server.ts`). This fetches verified web snippets from official sports authorities (ICC, ESPNcricinfo, Google Sports).
- **Strict JSON Schema Steering**: Prompts enforce exact structural outputs (teams, scores, overs, run rates, batsmen strike rates, bowling economics, and win probability matrices) while sanitizing markdown code blocks (`text.replace(/```json/g, '').trim()`).
- **Resilient Fallback Governance**: When API keys are absent in development or network constraints throttle live calls, the system gracefully falls back to a deterministic, high-fidelity local match simulator (`CricketEngine.simulateNextBall()`) without breaking UI rendering or crashing financial transactions.
- **Source Citation & Provenance**: Grounding chunks (`groundingMetadata.groundingChunks`) are preserved and passed to the frontend to show direct URLs and titles for verifiable transparency.

### II. Software Engineering Fundamentals
*Robust architecture, zero-leak financial integrity, deterministic state machines, and sound performance.*

- **Dual-Engine Separation**: Clear demarcation between the **Poker Game State Machine** (`gameManager.ts`), **Continuous Order Book Engine** (`market.ts`), **Sports Betting Engine** (`cricketEngine.ts`), and **Double-Entry Financial Ledger** (`ledger.ts`).
- **Financial Rigor & Zero-Sum Invariant**: All transactions (chips, cash, market bids, sports wagers, synergy bounties) are tracked in an immutable double-entry ledger. Bounty bonuses are mathematically extracted from losing opponents' stacks rather than created out of thin air.
- **Client-Side Latency & Responsive State**: Built on Vite + React 18 with modular components, Lucide icons, and zero layout thrashing. Audio feedback is synthesized via Web Audio API oscillators (`sound.ts`), avoiding external asset load dependencies.
- **Server Bundling**: Backed by `esbuild` CommonJS bundling (`dist/server.cjs`) with `--packages=external` to eliminate runtime ESM relative import friction in Cloud Run container environments.

### III. Using Coding Agents
*Maximizing agent velocity with disciplined context management, atomic verification, and tool-driven iteration.*

- **Read-Modify-Write Cycles**: Strict adherence to inspecting actual file manifests (`view_file`) prior to surgical updates (`edit_file`).
- **Automated Verification Loops**: Every code iteration was validated through `lint_applet` (TypeScript `--noEmit` type checking) and `compile_applet` (Vite build & bundling) to guarantee zero runtime syntax errors, broken imports, or type regressions.
- **Context Boundary Control**: Focused agent context on specific modular subsystems (e.g. `cricket.ts` types separated from `game.ts`) to prevent token bloating and file truncation.

### IV. Shaping the Build
*Product sense, game theory synergy, and high-impact UX craftsmanship.*

- **The "Dual Victory Synergy" Innovation**: Recognizing that players love compounding wins, we engineered the **Dual Synergy Bounty** (+15% to +100% bonus pool) which rewards players who dominate both the felt (Poker pot) and the market (Prediction contracts) in the same hand.
- **1-Click Micro-Betting UX**: Fast-paced poker requires ultra-low friction. Rather than forcing complex order modals, 1-click preset tickets ($25, $50, $100, $250) allow players to hedge their poker hands during 15-second trading windows.
- **Live Sports & Poker Convergence**: Integrated real-time cricket scores directly into the side arena, enabling players to switch seamlessly between Texas Hold'em event contracts and international sports wagers without leaving the table.

---

## 4. Step-by-Step Build Narrative

| Step | Milestone | Core AI Skill Exercised | Implementation Highlights |
| :--- | :--- | :--- | :--- |
| **1** | **Core Poker & Order Book Engine** | *Software Fundamentals* | Implemented 5-seat Texas Hold'em engine, hand evaluator, order book matching engine with binary outcome pricing ($0.01 - $0.99). |
| **2** | **Tournament Escalation & Dual Bounty** | *Shaping the Build* | Designed escalating tournament blind structure (Levels 1-5) and zero-sum synergy bounty extraction from losing players. |
| **3** | **Live Street-Level Fast Markets** | *Software Fundamentals* | Built dynamic market generators for Flop, Turn, and River micro-propositions (e.g., Ace/King flop, suit runs) with instant card settlements. |
| **4** | **Google Cricket Score Server Grounding** | *Building & Deploying AI Apps* | Configured Express backend with `gemini-3.7-flash` and Google Search tool to scan real-world ongoing international cricket matches. |
| **5** | **Live Sports Bet Arena & Ball Simulator** | *Building & Deploying AI Apps* + *Shaping the Build* | Built responsive Live Cricket Panel with ball-by-ball score updates, over-by-over micro-bets, win probability meters, and instant payout triggers. |
| **6** | **AI Engineering Showcase & Visuals** | *Using Coding Agents* + *Shaping the Build* | Synthesized codebase architecture, mapped to Andrew Ng's Skills Map, and embedded interactive visual architecture modal in the UI. |

---

## 5. What AI Does in the Live Product

| Dimension | Development Phase (Offline) | Running Product (Online Runtime) |
| :--- | :--- | :--- |
| **Tooling / Models** | Coding Agent (Antigravity & Gemini) | `gemini-3.7-flash` with Google Search Tool |
| **Primary Responsibility**| Writing TypeScript types, state machines, and styling | Real-time web search grounding for live cricket scores & odds |
| **Safety / Guardrails** | Automated linter (`tsc`) & compiler checks | Structured JSON extraction, timeout bounds, and local fallback engine |
| **User Interaction** | Code generation & architectural refactoring | Live match scores, ball ticker, win probability & market generation |

---

## 6. Key Takeaways for AI Engineers

1. **Grounding Over Hallucination for Dynamic Real-World Data**: Using Google Search Grounding with Gemini 3.7 turns an LLM into an up-to-the-minute data parser without maintaining brittle scrapers.
2. **Deterministic Financial Cores**: Always isolate financial, order-matching, and payout calculations into pure, deterministic TypeScript functions; let the AI handle perception, scanning, and natural language synthesis.
3. **Graceful Fallbacks are Mandatory**: Live AI web queries can fail or be throttled; a production AI application must have deterministic local simulation pathways to guarantee uninterrupted user experience.
4. **Product Synergy Drives Engagement**: Combining seemingly disparate domains (Poker + Prediction Markets + Live Sports) with a unifying economic loop (Shared Cash Balance & Synergy Bounties) creates exponential engagement.
