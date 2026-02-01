from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from alphafactory.config import load_config
from alphafactory.data.yfinance_io import load_yfinance_panel
from alphafactory.features import factors as factor_fns
from alphafactory.features.operators import winsorize_cs, zscore_cs, ewm_smooth
from alphafactory.labels import forward_return
from alphafactory.validation.splits import monthly_walk_forward_splits
from alphafactory.metrics.ic import rank_ic_daily, summarize_ic
from alphafactory.allocator.online_alm import OnlineALMAllocator
from alphafactory.portfolio.longshort import (
    long_short_weights_from_scores,
    staggered_holding_portfolio_returns,
    apply_linear_costs,
)
from alphafactory.reports.report import save_equity_curve_plot, write_report_md


FACTOR_REGISTRY = {
    "mom_12_1": lambda prices, vols: factor_fns.mom_12_1(prices),
    "rev_1m": lambda prices, vols: factor_fns.rev_1m(prices),
    "rev_5d": lambda prices, vols: factor_fns.rev_5d(prices),
    "vol_20d": lambda prices, vols: factor_fns.vol_20d(prices),
    "vol_change_20d": lambda prices, vols: factor_fns.vol_change_20d(prices),
    "dollar_volume": lambda prices, vols: factor_fns.dollar_volume(prices, vols),
    "volume_z_20d": lambda prices, vols: factor_fns.volume_z_20d(vols),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["reporting"]["output_dir"]) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- data -----------------
    tickers = cfg["data"]["tickers"]
    prices, volumes = load_yfinance_panel(
        tickers=tickers,
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        cache_dir=cfg["data"]["cache_dir"],
        price_field=cfg["data"].get("price_field", "Adj Close"),
        volume_field=cfg["data"].get("volume_field", "Volume"),
    )

    # daily returns (close-to-close proxy)
    daily_ret = prices.pct_change()

    # forward returns for IC / training signal quality
    fwd = forward_return(prices, cfg["label"]["delay_days"], cfg["label"]["horizon_days"])
    fwd = winsorize_cs(fwd, pct=float(cfg["label"].get("winsorize_pct", 0.0)))

    # ----------------- factors -----------------
    factor_names = list(cfg["factors"]["enabled"])
    factors = {}
    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            raise KeyError(f"Unknown factor: {name}. Available: {sorted(FACTOR_REGISTRY)}")
        raw = FACTOR_REGISTRY[name](prices, volumes)
        # optional smoothing to reduce turnover
        alpha = float(cfg["transforms"]["time_series"].get("ewm_alpha", 0.0))
        raw = ewm_smooth(raw, alpha=alpha)
        # cross-sectional normalization
        raw = winsorize_cs(raw, pct=0.01)
        raw = zscore_cs(raw)
        factors[name] = raw

    # ----------------- walk-forward -----------------
    splits = monthly_walk_forward_splits(
        dates=prices.index,
        train_years=int(cfg["validation"]["train_years"]),
        test_months=int(cfg["validation"]["test_months"]),
        embargo_days=int(cfg["validation"]["embargo_days"]),
    )
    if len(splits) < 6:
        raise RuntimeError("Too few splits. Expand date range or reduce train_years.")

    # Track results
    factor_rows = []
    portfolio_rows = []

    method = cfg["combination"]["method"]
    allocator = None
    if method == "online_alm":
        allocator = OnlineALMAllocator(
            n=len(factor_names),
            l1_budget=float(cfg["combination"]["online_alm"]["l1_budget"]),
            eta=float(cfg["combination"]["online_alm"]["eta"]),
            tau=float(cfg["combination"]["online_alm"]["tau"]),
        )

    # For “online” methods: only start after warmup splits
    warmup = int(cfg["combination"].get("online_alm", {}).get("warmup_splits", 0))

    all_port_rets = []

    for si, sp in enumerate(splits):
        train_mask = (prices.index >= sp.train_start) & (prices.index <= sp.train_end)
        test_mask = (prices.index >= sp.test_start) & (prices.index <= sp.test_end)

        # compute factor quality on train (IC mean)
        qualities = []
        oriented = []
        for name in factor_names:
            ic_train = rank_ic_daily(factors[name].loc[train_mask], fwd.loc[train_mask])
            summ = summarize_ic(ic_train)
            q = summ["mean"]
            # orient so "higher is better"
            sign = 1.0 if (q is not None and not np.isnan(q) and q >= 0) else -1.0
            qualities.append(abs(q) if q is not None else 0.0)
            oriented.append(sign)
            factor_rows.append(
                {
                    "split": si,
                    "factor": name,
                    "train_ic_mean": summ["mean"],
                    "train_ic_ir": summ["ir"],
                    "train_n": summ["n"],
                    "orientation": sign,
                }
            )

        qualities = np.array(qualities, dtype=float)
        oriented = np.array(oriented, dtype=float)

        # choose combination weights
        if method == "equal":
            w = np.full(len(factor_names), 1.0 / len(factor_names))
        elif method == "ic_weighted":
            w = qualities / qualities.sum() if qualities.sum() > 0 else np.full(len(factor_names), 1.0 / len(factor_names))
        elif method == "online_alm":
            if allocator is None:
                raise RuntimeError("Allocator not initialized")
            # During warmup, keep equal weights; afterwards update online
            if si < warmup:
                w = np.full(len(factor_names), 1.0 / len(factor_names))
            else:
                w = allocator.step(qualities)
        else:
            raise ValueError(f"Unknown combination method: {method}")

        # combined score on test
        combined = None
        for j, name in enumerate(factor_names):
            s = factors[name].loc[test_mask] * oriented[j]
            combined = s * w[j] if combined is None else combined.add(s * w[j], fill_value=0.0)

        assert combined is not None

        # portfolio construction on test
        sub_w = long_short_weights_from_scores(
            combined,
            long_q=float(cfg["portfolio"]["long_quantile"]),
            short_q=float(cfg["portfolio"]["short_quantile"]),
            gross_exposure=float(cfg["portfolio"]["gross_exposure"]),
            max_abs_weight=float(cfg["portfolio"]["max_abs_weight"]),
        )

        gross_ret = staggered_holding_portfolio_returns(
            sub_weights=sub_w,
            daily_returns=daily_ret,
            delay_days=int(cfg["label"]["delay_days"]),
            horizon_days=int(cfg["label"]["horizon_days"]),
        )

        # costs sweep
        for bps in cfg["costs"]["bps_list"]:
            net = apply_linear_costs(gross_ret, sub_w, cost_bps=float(bps))
            portfolio_rows.append(
                {
                    "split": si,
                    "test_start": str(sp.test_start.date()),
                    "test_end": str(sp.test_end.date()),
                    "cost_bps": float(bps),
                    "mean_daily": float(net.mean()),
                    "vol_daily": float(net.std(ddof=1)),
                }
            )

        all_port_rets.append(pd.DataFrame({"gross": gross_ret, "split": si}))

    factor_df = pd.DataFrame(factor_rows)
    port_df = pd.DataFrame(portfolio_rows)

    factor_df.to_csv(out_dir / "factor_ic_summary.csv", index=False)
    port_df.to_csv(out_dir / "portfolio_perf_by_split.csv", index=False)

    # Create a full-series return for cost=0 (concatenate splits)
    if len(all_port_rets) > 0:
        concat = pd.concat(all_port_rets).sort_index()
        gross = concat["gross"].copy()
        gross.to_csv(out_dir / "portfolio_daily_returns_gross.csv")

        # plot
        plots = {}
        p = out_dir / "plots" / "equity_gross.png"
        save_equity_curve_plot(gross.fillna(0.0), p, title="Gross cumulative growth (stitched splits)")
        plots["Gross equity curve"] = str(p.relative_to(out_dir))

        # summaries
        perf_summary = port_df.groupby("cost_bps", as_index=False).agg(
            mean_daily=("mean_daily", "mean"),
            vol_daily=("vol_daily", "mean"),
            n_splits=("split", "nunique"),
        )
        write_report_md(
            out_dir=out_dir,
            factor_summary=factor_df.groupby("factor", as_index=False).agg(
                train_ic_mean=("train_ic_mean", "mean"),
                train_ic_ir=("train_ic_ir", "mean"),
            ).sort_values("train_ic_mean", ascending=False),
            perf_summary=perf_summary,
            plots=plots,
        )

    # metadata
    (out_dir / "metadata.json").write_text(
        json.dumps({"config": cfg, "timestamp": ts}, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
