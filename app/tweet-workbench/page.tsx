"use client";

import { useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_TW_API || "http://localhost:8787";
const TOKEN = process.env.NEXT_PUBLIC_TW_TOKEN || "dev-only-change-me";

type PickRow = {
  player: string;
  line: string;
  market: string;
  side: string;
  edge_bps: number;
  model_prob?: number;
  mkt_prob?: number;
  books?: string;
  fact: string;
};

export default function TweetWorkbench() {
  const [games, setGames] = useState<string[]>([]);
  const [game, setGame] = useState<string>("");
  const [picks, setPicks] = useState<PickRow[]>([]);
  const [selectedFacts, setSelectedFacts] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string>(
    "Keep all numbers & bets exact. Bettor voice, not hype. One clean sentence per tweet."
  );
  const [tweets, setTweets] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const hdrs = useMemo(
    () => ({ "X-Admin-Token": TOKEN, "Content-Type": "application/json" }),
    []
  );

  useEffect(() => {
    fetch(`${API}/games`, { headers: hdrs })
      .then((r) => r.json())
      .then((d) => setGames(d.games || []))
      .catch(() => {});
  }, [hdrs]);

  function loadPicks(g: string) {
    setGame(g);
    setSelectedFacts([]);
    setTweets([]);
    fetch(`${API}/picks?game=${encodeURIComponent(g)}`, { headers: hdrs })
      .then((r) => r.json())
      .then((d) => setPicks(d.picks || []))
      .catch(() => setPicks([]));
  }

  async function doPolish() {
    if (!game || selectedFacts.length === 0) return;
    setBusy(true);
    try {
      const r = await fetch(`${API}/polish`, {
        method: "POST",
        headers: hdrs,
        body: JSON.stringify({
          game,
          picks: selectedFacts,
          instruction: prompt,
          model: "gpt-5",
          temperature: 0.7,
        }),
      });
      const d = await r.json();
      setTweets(d.tweets || []);
    } finally {
      setBusy(false);
    }
  }

  async function queueSelected() {
    if (!game || tweets.length === 0) return;
    await fetch(`${API}/queue`, {
      method: "POST",
      headers: hdrs,
      body: JSON.stringify({ game, tweets }),
    });
    alert("Queued to tweet_queue.json");
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Tweet Workbench (private)</h1>

      <div className="space-y-2">
        <label className="text-sm font-medium">Game</label>
        <select
          className="w-full border rounded-lg p-2"
          value={game}
          onChange={(e) => loadPicks(e.target.value)}
        >
          <option value="">Select a game…</option>
          {games.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
      </div>

      {!!picks.length && (
        <div className="space-y-3">
          <div className="text-sm text-gray-600">
            Choose picks to include (click to toggle). We’ll keep numbers exact.
          </div>
          <ul className="space-y-2">
            {picks.map((p, i) => {
              const checked = selectedFacts.includes(p.fact);
              return (
                <li
                  key={i}
                  className={`border rounded-lg p-3 cursor-pointer ${
                    checked ? "bg-indigo-50 border-indigo-300" : ""
                  }`}
                  onClick={() =>
                    setSelectedFacts((prev) =>
                      checked ? prev.filter((f) => f !== p.fact) : [...prev, p.fact]
                    )
                  }
                >
                  <div className="font-medium">
                    {p.player} — {p.side} {p.line} {p.market}
                  </div>
                  <div className="text-sm text-gray-600">
                    model {Math.round((p.model_prob || 0) * 100)}% vs{" "}
                    {Math.round((p.mkt_prob || 0) * 100)}%; +
                    {Math.round(p.edge_bps || 0)} bps
                    {p.books ? ` • Best: ${p.books}` : ""}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div className="space-y-2">
        <label className="text-sm font-medium">Prompt (optional)</label>
        <textarea
          className="w-full border rounded-lg p-2 h-24"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      <div className="flex gap-3">
        <button
          onClick={doPolish}
          disabled={!game || selectedFacts.length === 0 || busy}
          className="px-4 py-2 rounded-lg bg-black text-white disabled:opacity-50"
        >
          Generate 3 tweets
        </button>
        <button
          onClick={queueSelected}
          disabled={tweets.length === 0}
          className="px-4 py-2 rounded-lg bg-gray-200"
        >
          Queue to X
        </button>
      </div>

      {!!tweets.length && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Tweets</h2>
          {tweets.map((t, i) => (
            <div key={i} className="border rounded-lg p-3">
              <div className="text-sm text-gray-500 mb-2">Variant {i + 1}</div>
              <textarea
                className="w-full border rounded p-2 h-24"
                value={t}
                onChange={(e) =>
                  setTweets((prev) => prev.map((x, j) => (j === i ? e.target.value : x)))
                }
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
