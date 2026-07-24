"""
blowoff-watch — Blow-off top / buying climax detector
=====================================================
Detects the classic "staircase uptrend -> parabolic climax -> vicious reversal"
pattern as a 5-stage state machine, based on the five tells described in
TradingSim's "5 Ways to Identify Blow-Off Tops":

  #1  Massive uptrend with no real pullbacks (no >78.6% retracement, long
      duration, large total gain)
  #2  Significant volume on the down move
  #3  The broad market / benchmark is also putting in a top
  #4  The pullback from the top is vicious
  #5  The counter-rally is weak (low price recovery, low volume)

State machine:

  NEUTRAL -> UPTREND -> CLIMAX_WARNING -> VICIOUS_DROP -> BOUNCE_WATCH -> CONFIRMED
     ^                                                                       |
     +------------------------- reset ---------------------------------------+

Data source: tvDatafeed (unofficial TradingView). Swappable — the detector
only needs an OHLCV DataFrame, so yfinance / Polygon / Databento / CCXT all
work by replacing fetch_data().

Dependencies:
    pip install pandas numpy scipy
    pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

NOT financial advice. A detector is a warning system, not an entry button.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # --- Criterion #1: massive, orderly uptrend -----------------------------
    trend_lookback: int = 300          # bars used to evaluate the uptrend
    min_trend_gain_atr: float = 8.0    # total advance in ATR units (use % for stocks)
    min_trend_gain_pct: float = 0.0    # optional: e.g. 0.5 = +50% (meme stocks: 1.0+)
    max_retracement: float = 0.786     # no pullback deeper than 78.6% of the advance
    min_r2: float = 0.70               # regression fit quality = "orderly" staircase

    # --- Climax (parabolic extension above the channel) ---------------------
    channel_lookback: int = 120        # bars for the regression channel fit
    accel_lookback: int = 12           # local slope window
    accel_ratio: float = 2.5           # local slope / channel slope
    channel_break_atr: float = 1.0     # close above upper channel line, in ATR
    rsi_climax: float = 75.0
    climax_vol_z: float = 2.0          # volume z-score during the final push
    climax_grace: int = 40             # bars to keep watching for the drop after a climax

    # --- Criteria #2 + #4: vicious drop on heavy volume ---------------------
    drop_window: int = 16              # the drop must happen within this many bars
    vicious_drop_atr: float = 3.0      # peak-to-trough distance in ATR units
    down_vol_z: float = 1.5            # volume z-score on at least one down bar

    # --- Criterion #3: benchmark / broad market topping ---------------------
    benchmark_symbol: str | None = None    # e.g. "ES1!" / "SPY" / "IWM"
    benchmark_exchange: str = "CME_MINI"
    benchmark_ema: int = 50

    # --- Criterion #5: weak counter-rally -----------------------------------
    max_bounce_retrace: float = 0.50   # bounce recovers < 50% of the drop
    bounce_vol_z_max: float = 0.0      # bounce volume at/below average
    bounce_min_bars: int = 3           # give the bounce a few bars to form
    bounce_timeout: int = 60           # give up waiting after this many bars
    failed_retrace: float = 0.786      # reclaiming 78.6% of the drop = not a blow-off

    # --- Indicators ---------------------------------------------------------
    atr_period: int = 14
    rsi_period: int = 14
    vol_ma: int = 50


class State(str, Enum):
    NEUTRAL = "NEUTRAL"
    UPTREND = "UPTREND"                 # criterion #1 satisfied
    CLIMAX_WARNING = "CLIMAX_WARNING"   # parabolic extension in progress
    VICIOUS_DROP = "VICIOUS_DROP"       # criteria #2 + #4 fired
    BOUNCE_WATCH = "BOUNCE_WATCH"       # waiting for criterion #5
    CONFIRMED = "CONFIRMED"             # full blow-off top confirmed


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def linreg(y: np.ndarray) -> tuple[float, float, float]:
    """slope, intercept, r2 over bar index."""
    x = np.arange(len(y))
    res = stats.linregress(x, y)
    return res.slope, res.intercept, res.rvalue ** 2


def enrich(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr(df, cfg.atr_period)
    df["rsi"] = rsi(df["close"], cfg.rsi_period)
    df["vol_z"] = (
        (df["volume"] - df["volume"].rolling(cfg.vol_ma).mean())
        / df["volume"].rolling(cfg.vol_ma).std()
    )
    return df


# --------------------------------------------------------------------------- #
# Criterion #1 — massive uptrend without real pullbacks
# --------------------------------------------------------------------------- #

def uptrend_quality(df: pd.DataFrame, cfg: Config) -> dict:
    win = df.iloc[-cfg.trend_lookback:]
    closes = win["close"].to_numpy()
    lows = win["low"].to_numpy()
    cur_atr = float(df["atr"].iloc[-1])

    slope, _, r2 = linreg(closes)

    # Anchor the trend at the lowest low that precedes the highest high
    peak_idx = int(win["high"].to_numpy().argmax())
    start_idx = int(lows[: peak_idx + 1].argmin()) if peak_idx > 0 else 0
    start_low = float(lows[start_idx])
    advance = float(win["high"].iloc[peak_idx]) - start_low

    gain_atr = advance / cur_atr if cur_atr else 0.0
    gain_pct = advance / start_low if start_low else 0.0

    # Deepest retracement of the advance so far (walk-forward running high).
    # Only measured once the advance is established (> 3 ATR), otherwise any
    # early wiggle counts as a 100% retracement.
    max_retr = 0.0
    running_high = start_low
    for i in range(start_idx, len(win)):
        running_high = max(running_high, float(win["high"].iloc[i]))
        rng = running_high - start_low
        if rng > 3 * cur_atr:
            retr = (running_high - float(win["low"].iloc[i])) / rng
            max_retr = max(max_retr, retr)

    ok = (
        slope > 0
        and r2 >= cfg.min_r2
        and gain_atr >= cfg.min_trend_gain_atr
        and gain_pct >= cfg.min_trend_gain_pct
        and max_retr < cfg.max_retracement
    )
    return {
        "ok": ok,
        "r2": round(r2, 3),
        "gain_atr": round(gain_atr, 1),
        "gain_pct": round(gain_pct * 100, 1),
        "max_retracement": round(max_retr, 3),
    }


# --------------------------------------------------------------------------- #
# Climax — parabolic extension above the staircase channel
# --------------------------------------------------------------------------- #

def climax_check(df: pd.DataFrame, cfg: Config) -> dict:
    win = df.iloc[-cfg.channel_lookback:]
    closes = win["close"].to_numpy()
    cur_atr = float(df["atr"].iloc[-1])

    # Fit the channel on the staircase EXCLUDING the most recent (possibly
    # parabolic) bars, then extrapolate the upper line to "now". If the last
    # bars were included, the upper line would hug the blow-off itself and
    # price could never register as "above the channel".
    body = closes[: -cfg.accel_lookback]
    ch_slope, ch_intercept, _ = linreg(body)
    residuals = body - (ch_slope * np.arange(len(body)) + ch_intercept)
    upper_now = ch_slope * (len(closes) - 1) + ch_intercept + residuals.max()

    local_slope, _, _ = linreg(df["close"].iloc[-cfg.accel_lookback:].to_numpy())
    accel = ch_slope > 0 and (local_slope / ch_slope) >= cfg.accel_ratio

    last = df.iloc[-1]
    above = (float(last["close"]) - upper_now) / cur_atr if cur_atr else 0.0

    ok = (
        accel
        and above >= cfg.channel_break_atr
        and float(last["rsi"]) >= cfg.rsi_climax
        and (df["vol_z"].iloc[-3:] >= cfg.climax_vol_z).any()
    )
    return {
        "ok": ok,
        "accel_ratio": round(local_slope / ch_slope, 2) if ch_slope else None,
        "above_channel_atr": round(above, 2),
        "rsi": round(float(last["rsi"]), 1),
        "upper_channel": round(float(upper_now), 2),
    }


# --------------------------------------------------------------------------- #
# Criteria #2 + #4 — vicious drop on significant volume
# --------------------------------------------------------------------------- #

def vicious_drop_check(df: pd.DataFrame, cfg: Config) -> dict:
    win = df.iloc[-cfg.drop_window:]
    cur_atr = float(df["atr"].iloc[-1])

    peak_idx = int(win["high"].to_numpy().argmax())
    peak = float(win["high"].iloc[peak_idx])
    low_after = float(win["low"].iloc[peak_idx:].min())
    drop_atr = (peak - low_after) / cur_atr if cur_atr else 0.0

    down_bars = win.iloc[peak_idx:]
    heavy_down_vol = bool(
        (
            (down_bars["close"] < down_bars["open"])
            & (down_bars["vol_z"] >= cfg.down_vol_z)
        ).any()
    )

    ok = drop_atr >= cfg.vicious_drop_atr and heavy_down_vol
    return {
        "ok": ok,
        "drop_atr": round(drop_atr, 2),
        "heavy_down_volume": heavy_down_vol,
        "drop_peak": peak,
        "drop_low": low_after,
    }


# --------------------------------------------------------------------------- #
# Criterion #3 — benchmark / broad market topping
# --------------------------------------------------------------------------- #

def benchmark_topping(bench: pd.DataFrame | None, cfg: Config) -> dict:
    """Broad market weakness confirmation. Optional — if no benchmark data is
    supplied the criterion is reported but does not block the state machine."""
    if bench is None or len(bench) < cfg.benchmark_ema + 10:
        return {"ok": None, "note": "no benchmark data"}

    close = bench["close"]
    ema = close.ewm(span=cfg.benchmark_ema, adjust=False).mean()
    below_ema = bool(close.iloc[-1] < ema.iloc[-1])

    # Lower high: max of last 20 bars vs max of the 40 bars before that
    recent_high = float(close.iloc[-20:].max())
    prior_high = float(close.iloc[-60:-20].max())
    lower_high = recent_high < prior_high

    return {"ok": below_ema or lower_high, "below_ema": below_ema, "lower_high": lower_high}


# --------------------------------------------------------------------------- #
# Criterion #5 — weak counter-rally
# --------------------------------------------------------------------------- #

def bounce_check(df: pd.DataFrame, drop_peak: float, drop_low: float, cfg: Config) -> dict:
    """Evaluate the counter-rally after the vicious drop."""
    last = df.iloc[-1]
    rng = drop_peak - drop_low
    if rng <= 0:
        return {"ok": False, "failed": True}

    retrace = (float(last["close"]) - drop_low) / rng
    bounce = df.iloc[-cfg.bounce_min_bars:]
    weak_volume = bool(bounce["vol_z"].mean() <= cfg.bounce_vol_z_max)
    bounced = float(last["close"]) > drop_low

    return {
        "ok": bounced and retrace < cfg.max_bounce_retrace and weak_volume,
        "failed": retrace >= cfg.failed_retrace,  # reclaimed -> not a blow-off
        "retrace": round(retrace, 3),
        "weak_volume": weak_volume,
    }


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #

@dataclass
class Snapshot:
    state: str = State.NEUTRAL.value
    drop_peak: float | None = None
    drop_low: float | None = None
    bounce_bars: int = 0
    climax_bars: int = 0
    last_bar: str = ""
    history: list = field(default_factory=list)


class BlowoffMachine:
    def __init__(self, cfg: Config, persist: Path | None = None):
        self.cfg = cfg
        self.persist = persist
        self.snap = self._load()

    def _load(self) -> Snapshot:
        if self.persist and self.persist.exists():
            return Snapshot(**json.loads(self.persist.read_text()))
        return Snapshot()

    def _save(self):
        if self.persist:
            self.persist.write_text(json.dumps(asdict(self.snap)))

    def _transition(self, new: State, report: dict):
        old = self.snap.state
        if old != new.value:
            self.snap.history.append({"from": old, "to": new.value, "at": self.snap.last_bar})
            self.snap.history = self.snap.history[-50:]
            report["transition"] = f"{old} -> {new.value}"
        self.snap.state = new.value

    def step(self, df: pd.DataFrame, bench: pd.DataFrame | None = None) -> dict:
        cfg = self.cfg
        df = enrich(df, cfg)
        self.snap.last_bar = str(df.index[-1])

        trend = uptrend_quality(df, cfg)
        climax = climax_check(df, cfg)
        drop = vicious_drop_check(df, cfg)
        market = benchmark_topping(enrich(bench, cfg) if bench is not None else None, cfg)

        report = {
            "time": self.snap.last_bar,
            "close": float(df["close"].iloc[-1]),
            "state": self.snap.state,
            "c1_uptrend": trend,
            "climax": climax,
            "c2_c4_vicious_drop": drop,
            "c3_market_topping": market,
            "c5_weak_bounce": None,
            "alert": None,
        }

        st = State(self.snap.state)

        if st in (State.NEUTRAL, State.UPTREND):
            if trend["ok"] and climax["ok"]:
                self.snap.climax_bars = 0
                self._transition(State.CLIMAX_WARNING, report)
                report["alert"] = (
                    f"CLIMAX WARNING: parabolic push {climax['above_channel_atr']} ATR above "
                    f"channel, RSI {climax['rsi']}, accel x{climax['accel_ratio']}. "
                    f"Uptrend gain {trend['gain_atr']} ATR, max retrace {trend['max_retracement']}."
                )
            elif trend["ok"]:
                self._transition(State.UPTREND, report)
            else:
                self._transition(State.NEUTRAL, report)

        elif st == State.CLIMAX_WARNING:
            self.snap.climax_bars += 1
            if drop["ok"]:
                self.snap.drop_peak = drop["drop_peak"]
                self.snap.drop_low = drop["drop_low"]
                self.snap.bounce_bars = 0
                self._transition(State.VICIOUS_DROP, report)
                mk = " Benchmark also topping." if market.get("ok") else ""
                report["alert"] = (
                    f"VICIOUS DROP: {drop['drop_atr']} ATR off the peak on heavy volume "
                    f"(criteria #2+#4).{mk} Watching the counter-rally."
                )
            elif climax["ok"]:
                self.snap.climax_bars = 0  # climax still hot, keep the clock fresh
            elif self.snap.climax_bars > cfg.climax_grace:
                # climax cooled off and no vicious drop arrived -> stand down
                self._transition(State.NEUTRAL, report)

        elif st in (State.VICIOUS_DROP, State.BOUNCE_WATCH):
            # Track the extremes while the drop / bounce evolves
            self.snap.drop_peak = max(self.snap.drop_peak or 0, drop["drop_peak"])
            self.snap.drop_low = min(self.snap.drop_low or 1e18, drop["drop_low"])
            self.snap.bounce_bars += 1

            b = bounce_check(df, self.snap.drop_peak, self.snap.drop_low, cfg)
            report["c5_weak_bounce"] = b

            if b["failed"]:
                self._transition(State.NEUTRAL, report)
                report["alert"] = (
                    f"RESET: price reclaimed {int(cfg.failed_retrace * 100)}% of the drop — "
                    "not a blow-off top (the 'grind higher and make new highs' trap)."
                )
            elif self.snap.bounce_bars > cfg.bounce_timeout:
                self._transition(State.NEUTRAL, report)
            elif b["ok"] and self.snap.bounce_bars >= cfg.bounce_min_bars:
                self._transition(State.CONFIRMED, report)
                mk = "benchmark topping: yes" if market.get("ok") else "benchmark topping: no/unknown"
                report["alert"] = (
                    f"BLOW-OFF TOP CONFIRMED (5/5): weak counter-rally "
                    f"(retrace {b['retrace']}, low volume), {mk}. "
                    f"Peak {self.snap.drop_peak} -> low {self.snap.drop_low}."
                )
            else:
                self._transition(State.BOUNCE_WATCH, report)

        elif st == State.CONFIRMED:
            # Stay confirmed until structure invalidates, then re-arm
            b = bounce_check(df, self.snap.drop_peak, self.snap.drop_low, cfg)
            report["c5_weak_bounce"] = b
            if b["failed"]:
                self._transition(State.NEUTRAL, report)

        report["state"] = self.snap.state
        self._save()
        return report


# --------------------------------------------------------------------------- #
# Data + alerting
# --------------------------------------------------------------------------- #

def fetch_data(symbol: str, exchange: str, interval: str, bars: int = 600) -> pd.DataFrame:
    """tvDatafeed fetch — replace with yfinance/Polygon/Databento/CCXT as needed."""
    from tvDatafeed import TvDatafeed, Interval

    iv = {
        "1m": Interval.in_1_minute,
        "5m": Interval.in_5_minute,
        "15m": Interval.in_15_minute,
        "1h": Interval.in_1_hour,
        "4h": Interval.in_4_hour,
        "1d": Interval.in_daily,
    }[interval]
    tv = TvDatafeed()  # anonymous works; login gives more history
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=iv, n_bars=bars)
    return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]


def alert(msg: str):
    """Wire up Telegram / Discord / ntfy here."""
    print(f"\n{'=' * 70}\n[ALERT] {msg}\n{'=' * 70}\n")
    # import requests
    # requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    #               json={"chat_id": CHAT_ID, "text": msg})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Blow-off top detector (5-criteria state machine)")
    p.add_argument("--symbol", default="CL1!")
    p.add_argument("--exchange", default="NYMEX")
    p.add_argument("--interval", default="15m")
    p.add_argument("--benchmark", default=None, help="e.g. ES1! (criterion #3)")
    p.add_argument("--benchmark-exchange", default="CME_MINI")
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--once", action="store_true", help="run one analysis pass and exit")
    p.add_argument("--state-file", default=".blowoff_state.json")
    args = p.parse_args()

    cfg = Config(benchmark_symbol=args.benchmark, benchmark_exchange=args.benchmark_exchange)
    machine = BlowoffMachine(cfg, persist=Path(args.state_file))

    while True:
        try:
            df = fetch_data(args.symbol, args.exchange, args.interval)
            bench = (
                fetch_data(cfg.benchmark_symbol, cfg.benchmark_exchange, args.interval)
                if cfg.benchmark_symbol
                else None
            )
            report = machine.step(df, bench)
            print(json.dumps(report, indent=2, default=str))
            if report.get("alert"):
                alert(f"[{args.symbol}] {report['alert']}")
        except Exception as e:  # noqa: BLE001
            print("error:", e)

        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
