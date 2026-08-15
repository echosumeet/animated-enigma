import express, { Request, Response } from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialize Gemini AI Client
function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// In-Memory Live Matches Cache
let liveMatchesCache: any[] = [];
let lastScanTimestamp = 0;

// Health Check API
app.get('/api/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', serverTime: new Date().toISOString() });
});

// GET /api/cricket/live - Return Live Cricket Matches
app.get('/api/cricket/live', (req: Request, res: Response) => {
  res.json({
    success: true,
    lastScanned: lastScanTimestamp,
    matches: liveMatchesCache,
  });
});

// POST /api/cricket/scan - Live Scan Google Cricket Scores using Search Grounding
app.post('/api/cricket/scan', async (req: Request, res: Response) => {
  try {
    const ai = getGeminiClient();
    if (!ai) {
      return res.json({
        success: true,
        source: 'simulated_fallback',
        message: 'No GEMINI_API_KEY configured. Returning realistic live international match stream.',
        lastScanned: Date.now(),
        matches: null,
      });
    }

    const prompt = `Search for the latest real-time international cricket match scores, ongoing series (e.g. ICC Tournaments, India vs Australia, England vs South Africa, Pakistan, New Zealand, West Indies, T20 leagues).
Return a JSON array of matches currently live or recent with this exact structure:
[
  {
    "id": "unique-id",
    "matchTitle": "Team 1 vs Team 2 - Match Type",
    "seriesName": "Series/Tournament Name",
    "matchType": "ODI" or "T20I" or "TEST",
    "status": "LIVE" or "INNINGS_BREAK" or "STUMPS" or "MATCH_ENDED",
    "statusText": "e.g. IND need 32 runs in 24 balls",
    "venue": "Stadium, City",
    "team1": {
      "name": "Full Team 1 Name",
      "shortName": "T1",
      "flag": "Flag emoji",
      "score": "e.g. 278/6 (50.0 ov)",
      "overs": "50.0",
      "runs": 278,
      "wickets": 6,
      "isBatting": false,
      "isBowling": true
    },
    "team2": {
      "name": "Full Team 2 Name",
      "shortName": "T2",
      "flag": "Flag emoji",
      "score": "e.g. 248/4 (42.0 ov)",
      "overs": "42.0",
      "runs": 248,
      "wickets": 4,
      "isBatting": true,
      "isBowling": false
    },
    "currentBatsmen": [
      { "name": "Batsman 1", "runs": 65, "balls": 54, "fours": 6, "sixes": 2, "strikeRate": 120.3, "onStrike": true },
      { "name": "Batsman 2", "runs": 32, "balls": 21, "fours": 3, "sixes": 1, "strikeRate": 152.4, "onStrike": false }
    ],
    "currentBowler": { "name": "Bowler Name", "overs": "7.0", "maidens": 0, "runs": 42, "wickets": 2, "economy": 6.0 },
    "recentBalls": ["1", "4", "0", "2", "6", "1"],
    "currentOverNumber": 42,
    "currentBallInOver": 0,
    "crr": 5.9,
    "rrr": 8.0,
    "winProbability": { "team1Pct": 42, "team2Pct": 58 }
  }
]
Output ONLY valid JSON without markdown wrapping.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.7-flash',
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
        temperature: 0.2,
      },
    });

    const text = response.text || '';
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    const searchSources = groundingChunks
      .filter((chunk: any) => chunk?.web?.uri)
      .map((chunk: any) => ({
        title: chunk.web.title || 'Google Cricket Score Source',
        url: chunk.web.uri,
      }));

    let parsedMatches: any[] = [];
    try {
      const cleanJson = text.replace(/```json/g, '').replace(/```/g, '').trim();
      parsedMatches = JSON.parse(cleanJson);
    } catch (parseErr) {
      console.warn('Failed to parse Gemini JSON, will return raw text summary');
    }

    if (Array.isArray(parsedMatches) && parsedMatches.length > 0) {
      liveMatchesCache = parsedMatches.map((m) => ({
        ...m,
        lastUpdated: Date.now(),
        isLiveScannedWithGemini: true,
        groundingSources: searchSources,
      }));
      lastScanTimestamp = Date.now();
    }

    res.json({
      success: true,
      source: 'google_search_grounding',
      lastScanned: Date.now(),
      matches: liveMatchesCache,
      sources: searchSources,
    });
  } catch (err: any) {
    console.error('Error scanning Google Cricket scores with Gemini:', err);
    res.status(500).json({
      success: false,
      error: err?.message || 'Failed to scan live cricket scores',
    });
  }
});

// Vite & Static Asset Handling
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Market Poker server running on port ${PORT}`);
  });
}

startServer();
