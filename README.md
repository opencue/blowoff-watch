# blowoff-watch

Real-time blow-off top / buying climax detector for futures, stocks and crypto.

It watches for the classic sequence you see on high-volatility instruments (crude oil, GameStop-style momentum stocks): a staircase uptrend where buyers absorb every dip, a parabolic final push above the trend channel on euphoric volume, then a vicious reversal that erases days of gains in minutes — and alerts you at each stage.

The detection logic is a direct implementation of the five tells described in TradingSim's *5 Ways to Identify Blow-Off Tops*, turned into a deterministic state machine over OHLCV data.

```
NEUTRAL ──► UPTREND ──► CLIMAX_WARNING ──► VICIOUS_DROP ──► BOUNCE_WATCH ──► CONFIRMED
   ▲                                                                            │
   └──────────────────────────── reset (failed pattern) ────────────────────────┘
```

## The five criteria and how they map to code

**#1 — Massive uptrend with no real pullbacks** (`uptrend_quality`)
The advance is anchored at the lowest low preceding the highest high in the lookback window. It qualifies when the total gain exceeds a threshold (in ATR units, optionally also in %), a rolling linear regression fits with high R² (the "orderly staircase"), and no walk-forward pullback ever retraces more than 78.6% of the advance. Retracements are only measured once the advance exceeds 3 ATR, so early-trend noise doesn't count as a 100% retrace.

**The climax itself** (`climax_check`)
A regression channel is fitted on the staircase *excluding* the most recent bars, then extrapolated forward. The parabolic push registers when the local slope of the last N bars is a multiple of the channel slope, price closes ≥ 1 ATR above the extrapolated upper channel line, RSI is in climax territory (75+) and volume z-score spikes. This is the "final blow-off" — the moment buyers go vertical.

**#2 + #4 — Vicious pullback on significant volume** (`vicious_drop_check`)
Within a short window, price must fall a configurable multiple of ATR from the peak, with at least one heavy-volume down bar. Speed is the point: a slow drift down is a normal pullback, a multi-ATR drop in a handful of bars is the blow-off signature.

**#3 — Broad market topping** (`benchmark_topping`)
Optionally feed a benchmark (ES1!, SPY, IWM, DXY — whatever drives your instrument). It's flagged as topping when price is below its EMA(50) or printing lower highs. Per the article this is the "magic sauce" that separates real blow-off tops from a 30–40% haircut that grinds back to new highs. It's reported in every alert but doesn't gate the state machine, so you can weight it yourself.

**#5 — Weak counter-rally** (`bounce_check`)
After the drop, the bounce must recover less than 50% of the fall on at-or-below-average volume to confirm. If price instead reclaims 78.6% of the drop, the pattern is invalidated and the machine resets — that's the "traders mistakenly think the top is in" trap the article warns about.

Only when all stages fire in sequence do you get **BLOW-OFF TOP CONFIRMED (5/5)**.

## TradingView — live on the chart

Two Pine Script v6 ports of the same state machine live in [`pine/`](pine/) — pick one (or run both):

| Script | What it draws |
|---|---|
| [`blowoff_watch.pine`](pine/blowoff_watch.pine) | **zones edition** — background state zones (teal uptrend → orange climax → red drop → purple bounce-watch → deep red confirmed), transition labels, extrapolated channel top, full 5-criteria status table |
| [`blowoff_watch_levels.pine`](pine/blowoff_watch_levels.pine) | **levels edition** — forward-projected regression channel + dashed **climax trigger line** *before* the top, then TOP / LOW labels plus the **weak-bounce ceiling** (*Max bounce retrace*, default 50%) and **invalidation** (default 78.6%) retracement levels extended to the right. Levels track live as new extremes print; invalidated patterns are wiped, confirmed blow-offs stay as a frozen record (bounded by TradingView's drawing limit — oldest records are pruned). Optional **scenario fan**: three dashed templates per state — ① the pattern path (gray), ② the bullish alternative (teal), ③ the bearish alternative (maroon). Downside paths **stair-step at the chart's own measured wave rhythm** (swing-pivot leg size/duration), pausing at detected support, and endpoints carry **≈ETA stamps**. Plus a **break-watch layer**: state-aware decision levels annotated with what a close ↑above/↓below each means, swing **supports/resistance**, and **unfilled-gap magnet zones** — each with its own wave-speed ETA. Maps, not forecasts; ETAs assume continuous bars (sessions/weekends push real times later). **Anticipation layer**: climax-proximity counter in UPTREND (alert at 3/4 conditions) and a **top-ripeness score** (0–100: RSI divergence, volume fade, over-stretch, upper wicks, parabola deceleration) in CLIMAX — fires a **PRE-DROP warning** above the threshold and deepens the background as the top ripens. Risk meters, not entry signals |

Usage (both):

1. Open TradingView → **Pine Editor** → paste the script → **Add to chart**.
2. Set the benchmark for criterion #3 in the settings (e.g. `CME_MINI:ES1!` or `AMEX:SPY`); in the levels edition also tick **Use benchmark filter**.
3. **Alerts**: create one alert on the indicator with *"Any alert() function call"* to get the full dynamic message at every stage transition, or use the four `alertcondition()` entries (climax warning / vicious drop / confirmed / reset) for single-stage alerts.

Notes: both state machines advance on **confirmed bars only** — signals never repaint; what fired historically is what would have fired live. (The levels edition's *projected channel line* and *scenario ghost* are live drawings refreshed on the current bar — cosmetic only, they don't affect signals.) TradingView recomputes state from chart history on load, so there is no state file (the Python version persists state across restarts instead).

## Install (Python detector)

```bash
pip install pandas numpy scipy
pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
```

## Usage

```bash
# Crude oil futures, 15m bars, ES as broad-market benchmark
python blowoff_detector.py --symbol "CL1!" --exchange NYMEX --interval 15m \
    --benchmark "ES1!" --benchmark-exchange CME_MINI

# One-shot analysis (useful for cron / n8n / CI)
python blowoff_detector.py --symbol GME --exchange NYSE --interval 1h --once

# Meme-stock tuning example: require a real parabolic run
# (edit Config: min_trend_gain_pct=1.0, accel_ratio=4.0, vicious_drop_atr=5.0)
```

State survives restarts via `.blowoff_state.json`. Every poll prints a full JSON report of all five criteria; alerts fire on state transitions. Wire Telegram/Discord/ntfy into `alert()`.

## Data sources

tvDatafeed is the default because it maps 1:1 to TradingView symbols, but it's an unofficial library (ToS grey zone, can break anytime). The detector only needs an OHLCV DataFrame, so swapping `fetch_data()` gets you: **yfinance** (free, stocks/ETFs), **Polygon.io** or **Databento** (proper futures data incl. CL), broker APIs (**IBKR**, **Alpaca**), or **CCXT** (crypto). A fully ToS-clean TradingView integration is the Pine Script above — TradingView's own alert webhooks can call your server directly.

## Tuning

Defaults are calibrated roughly for CL 15m and **must be backtested per instrument**. The knobs that matter most:

| Parameter | Default | Meaning |
|---|---|---|
| `min_trend_gain_atr` | 8.0 | how big the staircase advance must be |
| `min_trend_gain_pct` | 0.0 | %-based gate for stocks (article: 100–500%+) |
| `max_retracement` | 0.786 | deepest allowed pullback of the advance |
| `accel_ratio` | 2.5 | local slope vs channel slope (memes: 4–5×) |
| `vicious_drop_atr` | 3.0 | drop size that counts as "vicious" |
| `max_bounce_retrace` | 0.50 | weak-bounce ceiling for confirmation |
| `climax_grace` | 40 | bars to keep watching for the drop after a climax |

## AI layer (optional)

The rule engine is cheap and deterministic; AI is best used as a **confirmation filter**, not the primary detector. Two patterns that work: send a chart screenshot to a vision model only when the machine reaches `CLIMAX_WARNING` and ask for a yes/no read, or collect labeled historical windows (PLUG, XOMA, GME, CL episodes) and train a small gradient-boosting classifier on the criterion features this script already computes.

## Disclaimer

This is a pattern detection tool, **not trading advice and not an entry signal**. Blow-off tops are notorious for one more squeeze before the reversal — the article's own advice is to let the structure confirm before acting, which is exactly why `CONFIRMED` requires all five criteria. Trade at your own risk.

## Credits

Pattern definition based on Al Hill's *5 Ways to Identify Blow-Off Tops* (TradingSim, updated June 2026).
