"""Render the human-readable audit report from replay artifacts."""
from __future__ import annotations
import json
import sqlite3
from collections import Counter
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent

def table(headers,rows):
    rows=[["—" if v is None else str(v) for v in row] for row in rows]
    widths=[max(len(str(h)),*(len(row[i]) for row in rows)) for i,h in enumerate(headers)]
    line=lambda cells:"| " + " | ".join(str(v).ljust(widths[i]) for i,v in enumerate(cells)) + " |"
    return "\n".join([line(headers),"| "+" | ".join("-"*w for w in widths)+" |",*(line(r) for r in rows)])

def fmt(v,n=4):
    return "—" if v is None or pd.isna(v) else f"{float(v):.{n}f}"

def find_examples(db_path: Path):
    db=sqlite3.connect(db_path); db.row_factory=sqlite3.Row
    records=db.execute("SELECT * FROM analysis_signals ORDER BY candle_timestamp").fetchall(); db.close()
    decoded=[]
    for r in records:
        p=json.loads(r["payload"]); outcome=json.loads(r["outcome_payload"]) if r["outcome_payload"] else None
        decoded.append({"row":r,"payload":p,"outcome":outcome})
    def example(name,predicate):
        item=next((x for x in decoded if predicate(x)),None)
        if not item: return f"**{name}:** No qualifying case occurred in this replay."
        p=item["payload"]; out=item["outcome"] or {}; ind=p.get("indicators",{}); struct=p.get("structure",{}); regime=p.get("market_regime",{})
        reasons=p.get("reasons",[])
        return (f"**{name}:** {p.get('candle_timestamp')} (candle start; generated {p.get('generated_at')}); "
          f"decision={p.get('classification')}, close={ind.get('close')}, regime={regime.get('classification')}, "
          f"score={fmt(p.get('final_decision_score'),3)}, raw ML UP={fmt(p.get('ml_evidence',{}).get('raw_probability_up'),4)}, "
          f"support={((struct.get('nearest_support') or {}).get('midpoint'))}, resistance={((struct.get('nearest_resistance') or {}).get('midpoint'))}; "
          f"status={item['row']['status']}, gross return={fmt(out.get('gross_return_pct'),4)}%. "
          f"Reasons: {' '.join(str(x) for x in reasons[:3])}")
    return [
      example("Profitable BUY setup",lambda x:x["row"]["classification"]=="BUY" and x["outcome"] and float(x["outcome"].get("gross_return_pct") or 0)>0),
      example("Profitable SELL setup",lambda x:x["row"]["classification"]=="SELL" and x["outcome"] and float(x["outcome"].get("gross_return_pct") or 0)>0),
      example("WAIT due to conflicting evidence",lambda x:x["row"]["classification"]=="WAIT FOR CONFIRMATION" and bool(x["payload"].get("conflicting_signals"))),
      example("Failed setup (stop hit)",lambda x:x["row"]["status"]=="STOP_LOSS_HIT"),
      example("Setup rejected on risk/reward",lambda x:x["row"]["classification"]=="WAIT FOR CONFIRMATION" and any("risk/reward" in str(r).lower() for r in x["payload"].get("reasons",[]))),
    ]

def main():
    meta=json.loads((HERE/"replay_metadata.json").read_text(encoding="utf-8"))
    dist=pd.read_csv(HERE/"signal_distribution.csv",dtype=str,keep_default_na=False)
    trades=pd.read_csv(HERE/"trade_outcomes.csv",dtype=str,keep_default_na=False)
    regimes=pd.read_csv(HERE/"regime_performance.csv",dtype=str,keep_default_na=False)
    db=sqlite3.connect(HERE/"replay.sqlite3")
    regime_counts=Counter()
    for (payload,) in db.execute("SELECT payload FROM analysis_signals"):
        p=json.loads(payload); regime=p.get("market_regime",{}).get("classification")
        if regime: regime_counts[regime]+=1
    db.close()
    combined=meta["strategies"]["combined"]
    entered=trades[(trades.strategy=="combined") & (trades.entry_price!="")]
    resolved=entered[entered.gross_return_pct!=""]
    status_counts=trades[trades.strategy=="combined"].status.value_counts().to_dict()
    target1_count=int((entered.target1_hit.astype(str).str.lower()=="true").sum())
    t2_count=int((trades[(trades.strategy=="combined") & (trades.status=="TARGET_2_HIT")].shape[0]))
    stop_count=int((trades[(trades.strategy=="combined") & (trades.status=="STOP_LOSS_HIT")].shape[0]))
    target1_rate=target1_count/len(entered)*100 if len(entered) else None
    t2_rate=t2_count/len(resolved)*100 if len(resolved) else None
    stop_rate=stop_count/len(resolved)*100 if len(resolved) else None
    report=[]
    report += ["# AMZN Trading Analysis Engine — Independent Correctness Audit", "",
      f"Replay period: **{meta['start_date_et']} through {meta['end_date_et']} ET** ({meta['completed_valid_rth_candles']:,} completed regular-session candles).",
      f"Source: `{meta['source_file']}` SHA-256 `{meta['source_sha256']}`. No training, tuning, live Alpaca request, broker order, or production DB write occurred.",
      f"Production model: `{meta['model_version']}` ({meta['model_class']}), SHA-256 `{meta['model_artifact_sha256']}`, {meta['feature_count']} ordered inputs; class order `{meta['class_order']}`. Raw model scores are not calibrated confidence.",
      f"The replay took **{meta['replay_seconds']:,.0f} seconds** on this environment. Independent SHA-256 checks matched the input snapshot and model artifact to the hashes recorded in replay metadata.",
      "", "## Independent correctness findings", "",
      "- **Causal inputs:** EMA/SMA, rolling returns, MACD and Bollinger features are backward-looking. RSI, ATR and volume averages reset by New York session. The production 40-feature adapter uses the unchanged legacy manifest; replay batch feature values were compared against prefix-only generation at four fixed positions.",
      "- **Pivot timing:** a 2×2 pivot is exposed only at the close of the second right-side candle. RSI divergence is attached to that confirmation candle. Replay prefix checks compared full-series and prefix indicator rows and filtered cached pivots by `confirmed_at <= current candle start` (the candle is completed before the decision).",
      "- **Zones:** only confirmed pivots are clustered; support/resistance prices and strength are computed from data available by each replay timestamp. Zone reaction strength uses up to six candles after pivot confirmation, truncated to the replay prefix. A touch is capped at one pivot per session.",
      "- **Breakout state:** close cross plus ATR penetration and relative volume are required for confirmation. Failed-breakout labels now require a matching upward/downward crossing in the configured 12-bar lookback; this avoids labeling a remote old zone as a recent failed break.",
      "- **Timestamp discipline:** candle timestamps denote starts. Analysis generation time is the completed candle end. Incomplete bars are filtered; duplicate timestamps and invalid OHLC are rejected; non-session/early-close-overrun records are excluded by the existing calendar. The latest live snapshot must be within the configured 15-minute lag.",
      "- **ML:** replay uses the frozen production LightGBM artifact and `legacy_engineering.generate_features(..., reset_daily=False)` with exact 40-column order and class mapping. Prediction for row *t* is made from that row's completed features only. No research challenger is loaded.",
      "- **Risk/outcomes:** signal close is a reference, not a fill. Entry is the next same-session 5-minute open only when inside the planned entry zone; otherwise the setup expires without P&L. A single position is allowed per strategy. Stop takes priority over target 2 when a bar touches both. Missing same-session intervals stay pending; no overnight entry/exit is assumed.",
      "", "### Corrections made during the audit", "",
      "- Replaced same-candle close fill assumption with next-bar open/entry-zone evaluation.",
      "- HOLD is now non-actionable and cannot open another stored position; opposite positions do not overlap.",
      "- Prevented intraday open positions from being resolved using next-session prices when the session-end bar is missing.",
      "- Made target-1 lifecycle events idempotent and prevented target 1 being credited when the same candle also hits the stop.",
      "- Restricted failed-breakout labeling to recent initiating crossings.",
      "", "### Prefix parity checks", "",
      table(["Index","Candle timestamp","Legacy feature prefix","Indicator prefix","Structure prefix","Future pivots excluded"],
        [[p["index"],p["timestamp"],p["legacy_feature_prefix_equal"],p["indicator_prefix_equal"],p["structure_prefix_equal"],p["future_pivots_excluded"]] for p in meta["lookahead_prefix_parity"]]),
      "", "## Signal distribution", "",
      f"Valid analyses: **{meta['valid_analysis_results']:,}**; rows without required warm-up: **{meta['insufficient_history_rows']:,}**; completed valid candles: **{meta['completed_valid_rth_candles']:,}**; calendar-filtered rows: **{meta['calendar_rows_excluded']:,}**.",
      f"Actionable BUY/SELL signals: **{meta['actionable_signal_count']:,} ({fmt(meta['actionable_pct_valid'],2)}% of valid analyses)**. HOLD is not counted as a new trade.",
      "", table(["Decision","Count","Share of valid analyses (%)"],[[r.classification,r['count'],fmt(r.share_of_valid_pct,2) if r.share_of_valid_pct!="" else "—"] for _,r in dist.iterrows()]),
      "", "## Hypothetical execution results", "",
      "Each strategy is an independent, one-position-at-a-time ledger. Entries use one share and fixed size; return percentages are summed additively, not compounded. Drawdown is measured in percentage points on cumulative net returns. Baselines use the same entry-zone, stop/target, close/time limit and 10 bps assumed round-trip cost. Different strategy ledgers are not combined into a portfolio.",
      "", table(["Strategy","Setups","Resolved trades","Gross return sum (%)","Net return sum (%)","Max DD (pp)","Gross PF","Net PF","Avg net/trade (%)","Expired","Unresolved"],
        [[name,s["setups"],s["trades"],fmt(s["gross_return_pct_sum"]),fmt(s["net_return_pct_sum"]),fmt(s["max_drawdown_pct_points"]),fmt(s["profit_factor_gross"]),fmt(s["profit_factor_net"]),fmt(s["average_net_return_pct"]),s["expired"],s["unresolved"]] for name,s in meta["strategies"].items()]),
      "", f"Combined setup hit counts: target 1 **{target1_count}** ({fmt(target1_rate,2)}% of entered setups), target 2 **{t2_count}** ({fmt(t2_rate,2)}% of resolved trades), stop loss **{stop_count}** ({fmt(stop_rate,2)}% of resolved trades), expired **{combined['expired']}**, stop-invalidated **{combined['invalidated']}**, unresolved **{combined['unresolved']}**. Target 1 is a milestone only; no partial exit/P&L is booked.",
      "", "### Regime breakdown", "",
      "`valid_candle_count` is the full valid decision sample in each regime; `entered_or_closed_setups` is the strategy-specific setup sample. These are small-sample historical observations, not evidence of profitability.",
      "", table(["Strategy","Regime","Valid candles","Setups","Resolved","Net return (%)","Max DD (pp)","Target 1","Target 2","Stops"],
        [[r.strategy,r.regime,regime_counts[r.regime],r.entered_or_closed_setups,r.trades,fmt(r.net_return_pct_sum),fmt(r.max_drawdown_pct_points),r.target1_hits,r.target2_hits,r.stop_hits] for _,r in regimes.iterrows()]),
      "", "## Historical signal examples", "", *find_examples(HERE/"replay.sqlite3"),
      "", "## API and operational verification", "",
      "Actual HTTP calls were made to an isolated Uvicorn process whose SQLite path was redirected to a temporary database. `GET latest`, `GET signals`, `GET active`, and `GET outcomes` returned 200; unknown signal detail returned 404; outcome resolution returned 200. `POST analysis/run` returned 422 because the latest local snapshot is older than the allowed current-session staleness window; this verifies stale-data rejection. No production database was used for these HTTP checks.",
      "No live Alpaca request was made. The scheduler hook was code-inspected and exercised through outcome service unit tests, not by leaving a scheduler process running.",
      "", "## Correctness status and limitations", "",
      "**Verified:** timestamp completion and calendar filtering; duplicate prevention; frozen-model manifest/class order; selected prefix parity; pivot confirmation timing; single-position entry/exit assumptions; target/stop order; no P&L for expired/unresolved setups; repeat outcome idempotence; API response routing on isolated storage.",
      "**Heuristic / not validated as predictive:** the combined score and its weights, zone strength, trend/regime thresholds, risk distances and assumed 10 bps cost. The model raw probability is uncalibrated. Stop-first handling is conservative but intrabar event order cannot be known from 5-minute OHLC.",
      "**Results not trustworthy as investment evidence:** this is one historical period with fixed one-share additive returns, no spread/slippage model beyond an assumed cost, no partial exits, and no statistical significance analysis. Baseline choices are simple descriptive comparators, not tuned strategies. Replay does not show executed trades or real P&L.",
      "**Data limitations:** local snapshot ends on November 9, 2026, so current live data is stale; 36 records were excluded by the market calendar. Same-session missing intervals remain pending and may leave setups unresolved. No order book, bid/ask, actual fill, commissions, market impact, or calibrated outcome probabilities are available.",
      "", "## Reproduction", "",
      "From the repository root: `python reports/experiments/trading_analysis_engine_audit/replay.py` (use the documented virtual environment's Python on Windows). The script writes the CSV files, isolated `replay.sqlite3`, and `replay_metadata.json`; then run `python reports/experiments/trading_analysis_engine_audit/render_audit_report.py` to render this report.",
      "", "See `test_results.txt` for commands and test status.", ""]
    (HERE/"audit_report.md").write_text("\n".join(report),encoding="utf-8")

if __name__=="__main__": main()
